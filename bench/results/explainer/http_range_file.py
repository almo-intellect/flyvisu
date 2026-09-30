"""A minimal seekable, read-only file-like object over HTTP Range requests.

Lets Python's `zipfile` module open a remote ZIP archive and read only the
central directory plus a handful of selected members, without downloading
the whole archive. Buffers reads in chunks so that zipfile's many small
sequential reads (local file headers, compressed payload in 8KB pieces,
etc.) don't turn into one HTTP request per read() call.
"""
import time
import requests


class HTTPRangeFile:
    def __init__(self, url, chunk_size=4 * 1024 * 1024, session=None, max_retries=5,
                 timeout=60, verbose=False):
        self.url = url
        self.chunk_size = chunk_size
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.timeout = timeout
        self.verbose = verbose
        self.pos = 0
        self.total_bytes_downloaded = 0
        self.num_requests = 0
        self._buf_start = 0
        self._buf = b""

        r = self._request_with_retries("HEAD", headers={})
        if "Content-Length" not in r.headers:
            raise RuntimeError("Server did not report Content-Length; cannot proceed.")
        self.size = int(r.headers["Content-Length"])
        accept_ranges = r.headers.get("Accept-Ranges", "").lower()
        if accept_ranges != "bytes":
            # Some servers omit the header on HEAD but still honour Range on GET;
            # verify with an actual ranged GET below instead of failing here.
            if self.verbose:
                print(f"[HTTPRangeFile] warning: Accept-Ranges header is "
                      f"{accept_ranges!r}, will verify with a real Range GET")

        # Verify the server actually honours Range with a real 206 response.
        probe = self._raw_range_get(0, min(15, self.size - 1))
        if probe.status_code != 206:
            raise RuntimeError(
                f"Server does not honour HTTP Range requests (status "
                f"{probe.status_code} for a ranged GET). Refusing to fall back "
                f"to downloading the whole archive."
            )

    # -- low level -----------------------------------------------------
    def _request_with_retries(self, method, headers, stream=False):
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.request(
                    method, self.url, headers=headers, timeout=self.timeout,
                    allow_redirects=True, stream=stream,
                )
                self.num_requests += 1
                return resp
            except requests.exceptions.RequestException as e:
                last_exc = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Request failed after {self.max_retries} retries: {last_exc}")

    def _raw_range_get(self, start, end):
        headers = {"Range": f"bytes={start}-{end}"}
        resp = self._request_with_retries("GET", headers=headers)
        return resp

    def _fetch_range(self, start, end):
        """Fetch bytes [start, end] inclusive, with retries, return bytes."""
        end = min(end, self.size - 1)
        if end < start:
            return b""
        for attempt in range(self.max_retries):
            resp = self._raw_range_get(start, end)
            if resp.status_code == 206:
                data = resp.content
                self.total_bytes_downloaded += len(data)
                if self.verbose:
                    print(f"[HTTPRangeFile] GET bytes={start}-{end} -> "
                          f"{len(data)} bytes (req #{self.num_requests})")
                return data
            time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(
            f"Range request for bytes={start}-{end} did not return 206 "
            f"after {self.max_retries} attempts (last status {resp.status_code})."
        )

    # -- file-like API used by zipfile ----------------------------------
    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return False

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        if self.pos < 0:
            raise OSError("negative seek position")
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            want_end = self.size - 1
        else:
            if n == 0:
                return b""
            want_end = self.pos + n - 1
        want_end = min(want_end, self.size - 1)
        if want_end < self.pos:
            return b""

        buf_end = self._buf_start + len(self._buf) - 1
        if not (self._buf and self._buf_start <= self.pos <= buf_end + 1
                and want_end <= buf_end):
            # Cache miss (or exact-end append not applicable): fetch a fresh
            # chunk starting at pos, sized to cover at least the request.
            fetch_len = max(want_end - self.pos + 1, self.chunk_size)
            fetch_end = min(self.pos + fetch_len - 1, self.size - 1)
            self._buf = self._fetch_range(self.pos, fetch_end)
            self._buf_start = self.pos
            buf_end = self._buf_start + len(self._buf) - 1

        start_off = self.pos - self._buf_start
        end_off = min(want_end - self._buf_start + 1, len(self._buf))
        data = self._buf[start_off:end_off]
        self.pos += len(data)
        return data

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
