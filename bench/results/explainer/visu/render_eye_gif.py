"""Render plate 1 of the VISU explorer ("O olho tem 721 píxeis hexagonais") as a looping GIF.

Same geometry, colours and labels as the page (visu-anatomia.template.html, function eye()):
left lattice plays the real Sintel clip (alley_2, 19 frames) on the 721-hexal eye, right lattice
shows the fingerprint stimulus, a bright edge moving at 25 deg. One loop = 19 clip frames at
100 ms per GIF frame with the clip advancing every second frame (3.8 s); the edge crosses the eye
once per loop.
"""
import json, math, sys, urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SP = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else SP / "visu-olho-721.gif"
SS = 2                                    # supersampling for antialiased hexagons
W, H = 1200, 660                          # output size (page drawing is 900x470 + a title band)
SCALE = W / 900.0                         # page units -> output px
TOP = 80                                  # title band height (output px)

# --- colours (VISU tokens)
SURFACE = (0x08, 0x1B, 0x4D)
INK = (0xF9, 0xF9, 0xF9)
MUTED = (0xB9, 0xC2, 0xDA)
ACCENT = (0xF2, 0xB4, 0x4B)
HEX_DARK = (0x0B, 0x1F, 0x55)
HEX_LIGHT = (0xF4, 0xEF, 0xE4)


def mix(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


# --- fonts: Lexend Mega (title) and Lexend (labels), fetched once from Google Fonts' TTF endpoints
FONTS = SP / "fonts"
FONTS.mkdir(exist_ok=True)


GF = "https://github.com/google/fonts/raw/main/ofl/"


def gf_ttf(rel, dest):                     # variable TTFs from the google/fonts repository
    if not dest.exists():
        req = urllib.request.Request(GF + rel, headers={"User-Agent": "Mozilla/5.0"})
        dest.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return dest


def vfont(path, size, weight):
    f = ImageFont.truetype(str(path), size)
    try:
        f.set_variation_by_axes([weight])
    except Exception:
        pass
    return f


try:
    mega = gf_ttf("lexendmega/LexendMega%5Bwght%5D.ttf", FONTS / "LexendMega-VF.ttf")
    lex = gf_ttf("lexend/Lexend%5Bwght%5D.ttf", FONTS / "Lexend-VF.ttf")
    F_TITLE = vfont(mega, int(26 * SS), 700)
    F_STEP = vfont(mega, int(30 * SS), 700)
    F_LBL = vfont(lex, int(13 * SS), 400)
    F_LBL_S = vfont(lex, int(12 * SS), 400)
    print("fonts: Lexend Mega + Lexend (variable TTFs)")
except Exception as e:
    print("font download failed, falling back to Arial:", e)
    F_TITLE = ImageFont.truetype("arialbd.ttf", int(26 * SS)); F_STEP = F_TITLE
    F_LBL = ImageFont.truetype("arial.ttf", int(13 * SS)); F_LBL_S = ImageFont.truetype("arial.ttf", int(12 * SS))

# --- data
EX = next(d for d in (SP, SP / "explainer", SP.parent) if (d / "sintel_clip.json").exists())
sin = json.loads((EX / "sintel_clip.json").read_text())
uv = sin["uv"]; lum = sin["lum"]; NF = sin["frames"]

# --- geometry (page units, as in eye() with the Sintel clip present)
R0, G0 = 6.3, 0.6
S0 = R0 + G0
EX, EY, LX = 668, 222, 232


def hex_cells(E):
    return [(u, v) for u in range(-E, E + 1) for v in range(-E, E + 1) if max(abs(u), abs(v), abs(u + v)) <= E]


def centre(cx, cy, u, v):
    return cx + math.sqrt(3) * S0 * (u + v / 2), cy + 1.5 * S0 * v


def hex_poly(x, y, r):
    return [((x + r * math.cos(math.radians(60 * i - 30))), (y + r * math.sin(math.radians(60 * i - 30)))) for i in range(6)]


def P(x, y):                               # page units -> supersampled output px
    return (x * SCALE * SS, (TOP + y * SCALE) * SS)


right = [(u, v, *centre(EX, EY, u, v)) for u, v in hex_cells(15)]
left = [centre(LX, EY, u, v) for u, v in uv]
th = math.radians(25); dx, dy = math.cos(th), math.sin(th)
proj = [(x - EX) * dx + (y - EY) * dy for _, _, x, y in right]
lo, hi = min(proj), max(proj)
blur = 3 * S0
capY = EY + 16 * 1.5 * S0 + 12

frames = []
N = 2 * NF                                 # 38 frames, 100 ms each
for f in range(N):
    img = Image.new("RGB", (W * SS, H * SS), SURFACE)
    d = ImageDraw.Draw(img)
    # title band
    d.text(P(18, -58), "1", font=F_STEP, fill=INK)
    d.text(P(48, -56), "O olho tem 721 píxeis hexagonais", font=F_TITLE, fill=INK)
    # left: the Sintel clip
    k = (f // 2) % NF
    for (x, y), L in zip(left, lum[k]):
        d.polygon([P(px, py) for px, py in hex_poly(x, y, R0)], fill=mix(HEX_DARK, HEX_LIGHT, L / 255))
    # right: the moving edge, one crossing per loop
    edge = lo - 60 + (f / N) * (hi - lo + 120)
    for (u, v, x, y), pr in zip(right, proj):
        val = 1 - 1 / (1 + math.exp(-(pr - edge) / blur))
        d.polygon([P(px, py) for px, py in hex_poly(x, y, R0)], fill=mix(HEX_DARK, HEX_LIGHT, val))
    # central column outlined in gold + leader
    cx, cy = centre(EX, EY, 0, 0)
    d.polygon([P(px, py) for px, py in hex_poly(cx, cy, R0)], outline=ACCENT, width=int(2.2 * SCALE * SS))
    lx, ly = EX + 70, 36
    x1, y1 = P(cx + 7, cy - 5); x2, y2 = P(lx - 4, ly + 8)
    n_dash = 40
    for i in range(0, n_dash, 2):
        t0, t1 = i / n_dash, (i + 1) / n_dash
        d.line([(x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0), (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)], fill=ACCENT, width=SS)
    d.text(P(lx, ly - 10), "coluna central", font=F_LBL, fill=ACCENT)
    d.text(P(lx, ly + 3), "a sua T4c é a célula", font=F_LBL_S, fill=MUTED)
    d.text(P(lx, ly + 15), "de que tiramos a impressão digital", font=F_LBL_S, fill=MUTED)
    # motion arrow with marching dashes
    ax, ay, L = EX - 228, capY - 40, 56
    x1, y1 = P(ax, ay); x2, y2 = P(ax + L * dx, ay + L * dy)
    seg = 10
    shift = (f * 3) % seg
    total = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / total, (y2 - y1) / total
    s = -shift * SS
    while s < total:
        a, b = max(s, 0), min(s + 6 * SS, total)
        if b > a:
            d.line([(x1 + ux * a, y1 + uy * a), (x1 + ux * b, y1 + uy * b)], fill=ACCENT, width=int(2.5 * SCALE * SS))
        s += seg * SS
    tipx, tipy = x2, y2
    d.polygon([(tipx, tipy), (tipx - 9 * SS * ux + 5 * SS * uy, tipy - 9 * SS * uy - 5 * SS * ux), (tipx - 9 * SS * ux - 5 * SS * uy, tipy - 9 * SS * uy + 5 * SS * ux)], fill=ACCENT)
    d.text(P(ax - 4, ay - 24), "movimento · 25°", font=F_LBL, fill=ACCENT)
    # captions
    def centred(x, y, s, fnt, fill):
        w = d.textlength(s, font=fnt)
        px, py = P(x, y)
        d.text((px - w / 2, py - fnt.size), s, font=fnt, fill=fill)
    centred(LX, capY, f"entrada de treino · Sintel «{sin['sequence']}» · 19 fotogramas a 20 ms, 10× mais lento", F_LBL, MUTED)
    centred(LX, capY + 15, f"fotograma {k + 1} de {NF} · t = {k * 20} ms", F_LBL_S, MUTED)
    centred(EX, capY, "estímulo da impressão digital · uma aresta luminosa a mover-se a 25°", F_LBL, MUTED)
    frames.append(img.resize((W, H), Image.LANCZOS))

# one shared palette so the loop does not flicker; a swatch strip keeps the accent colours exact
pal_src = frames[N // 2].copy()
sw = ImageDraw.Draw(pal_src)
for i, col in enumerate([ACCENT, INK, MUTED, SURFACE, HEX_DARK, HEX_LIGHT]):
    sw.rectangle([i * 60, 0, i * 60 + 59, 40], fill=col)
pal_src = pal_src.quantize(colors=255, method=Image.MEDIANCUT)
q = [fr.quantize(palette=pal_src, dither=Image.NONE) for fr in frames]
q[0].save(OUT, save_all=True, append_images=q[1:], duration=100, loop=0, optimize=True, disposal=1)
print("wrote", OUT, OUT.stat().st_size // 1024, "KB,", len(q), "frames,", W, "x", H)
