"""What does a fully trained model score under flyvis 1.2.0's OWN validation loss?

Our learning curves start at 1212.6 and the released ensemble's stored minimum validation loss
is 5.31. If the stored number was written by an older loss definition (per-element RMS instead
of the current root-of-summed-squares), the trained target on our scale is not 5.31 but
~5.31 * sqrt(19 * 2 * 721) = 879, and the "230x gap" is really a 27% gap. This evaluates
released checkpoints with the current code on the current validation set and reports both.

  python bench_eval_released.py            models 000 001 002, checkpoints best + last
  python bench_eval_released.py 000 005    chosen models
"""
import os, subprocess, sys, time, math
from pathlib import Path

os.environ.setdefault("FLYVIS_ROOT_DIR", "/content/flyvis_data")
import numpy as np
import torch
import flyvis
from flyvis import NetworkView
from flyvis.task.tasks import Task


def log(*a):
    print(*a, flush=True)


def ensure_pretrained():
    if not (Path(flyvis.results_dir) / "flow" / "0000" / "000").exists():
        log("[eval] downloading pretrained models ...")
        subprocess.run(["flyvis", "download-pretrained"], check=True)


def val_loss(network, decoder, task, t_pre=0.25):
    """Same computation as MultiTaskSolver.test(): mean over the validation loader of the task loss."""
    dl = task.val_data
    losses = []
    network.eval()                       # as MultiTaskSolver._eval(): no dropout / batch-norm updates
    for m in decoder.values():
        m.eval()
    with torch.no_grad(), task.dataset.augmentation(False):
        ss = network.steady_state(t_pre=t_pre, dt=task.dataset.dt, batch_size=dl.batch_size, value=0.5)
        for data in dl:
            n_samples, n_frames = data["lum"].shape[:2]
            network.stimulus.zero(n_samples, n_frames)
            network.stimulus.add_input(data["lum"])
            act = network(network.stimulus(), task.dataset.dt, state=ss)
            y_est = decoder["flow"](act)
            losses.append(float(task.loss(y_est, data["flow"], "flow", **data.get("loss_kwargs", {}))))
    return float(np.mean(losses))


def default_task():
    """The 1.2.0 default task: the same validation set and loss our learning curves use.
    (The released configs' task section has keys 1.2.0's Task no longer accepts.)"""
    from flyvis.utils.config_utils import get_default_config
    import flyvis.utils.config_utils as cu
    cfg = get_default_config(path=str(Path(cu.__file__).parent.parent / "config" / "solver.yaml"),
                             overrides=["task_name=flow", "ensemble_and_network_id=curve/evaltmp"])
    return Task(**cfg.task)


def main(models):
    ensure_pretrained()
    task = None
    scale = math.sqrt(19 * 2 * 721)
    for mid in models:
        nv = NetworkView(f"flow/0000/{mid}")
        cfg = nv.dir.config
        if task is None:
            task = default_task()
            log(f"[eval] validation loader: {len(task.val_data)} batches of {task.val_data.batch_size}, "
                f"n_frames {task.dataset.n_frames}, dt {task.dataset.dt}")
            try:
                log(f"[eval] released task config keys: {sorted(dict(cfg.task).keys())}; "
                    f"n_iters {cfg.task.get('n_iters')}, decoder {dict(cfg.task.get('decoder', {})).get('flow')}")
            except Exception as e:
                log(f"[eval] could not print released task config: {e}")
        try:
            stored = np.atleast_1d(np.asarray(nv.dir.validation.loss[()], dtype=float))
        except Exception as e:
            stored = np.array([float("nan")])
            log(f"[eval] {mid}: no stored validation loss ({e})")
        paths = nv.checkpoints.paths
        log(f"[eval] model {mid}: {len(paths)} checkpoints on disk; stored validation/loss = "
            f"{stored.tolist() if stored.size <= 6 else (stored.min(), stored.shape)}")
        for label, ck in (("best", "best"), ("last", len(paths) - 1), ("first", 0)):
            t0 = time.time()
            try:
                net = nv.init_network(checkpoint=ck)
                dec = nv.init_decoder(checkpoint=ck)
                v = val_loss(net, dec, task)
                log(f"RESULT model={mid} checkpoint={label} val_1.2.0={v:.3f} "
                    f"per_element_rms={v / scale:.4f} ({time.time() - t0:.0f} s)")
            except Exception as e:
                log(f"RESULT model={mid} checkpoint={label} FAILED {type(e).__name__}: {str(e)[:160]}")
    log(f"[eval] scale sqrt(19*2*721) = {scale:.2f}; stored 5.31 * scale = {5.3143 * scale:.1f}")
    log("EVAL DONE")


if __name__ == "__main__":
    main(sys.argv[1:] or ["000", "001", "002"])
