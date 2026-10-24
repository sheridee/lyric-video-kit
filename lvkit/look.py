"""The look: blurred Ken Burns image, beat-reactive glow, word-by-word highlight (Chrome text PNGs),
mirrored spectrum bars, progress hairline. Pure numpy/PIL compositing per frame."""
import os, html, hashlib, subprocess, colorsys
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFilter
from .util import FONT, chrome_bin

FPS = 30
LAYOUTS = {  # all sizes in output px
    "16:9": dict(W=1920, H=1080, font=92, stack=722, maxw=1640, rows=1, nextf=30, secf=16,
                 nb=40, bw=9, gap=5, base=1030, hmax=110, title="bottom", tx=64, ty=44, titlef=22),
    "9:16": dict(W=1080, H=1920, font=96, stack=1250, maxw=940, rows=2, nextf=30, secf=18,
                 nb=26, bw=10, gap=6, base=1730, hmax=130, title="top", tx=80, ty=150, titlef=26),
    "1:1":  dict(W=1080, H=1080, font=80, stack=630, maxw=940, rows=2, nextf=26, secf=15,
                 nb=26, bw=8, gap=5, base=1010, hmax=80, title="top", tx=60, ty=56, titlef=20),
}
DEFAULT = dict(accent=(255, 62, 165), accent2=(90, 240, 255), accent3=(255, 118, 92))

def hex2rgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def rgb2hex(c): return "#%02x%02x%02x" % tuple(int(x) for x in c)

def neon(c, s=0.85, v=1.0):
    h, _, _ = colorsys.rgb_to_hsv(*[x / 255 for x in c])
    return tuple(int(round(x * 255)) for x in colorsys.hsv_to_rgb(h, s, v))

def _families(img_path):
    """Vivid hue families of the image: [(weight, hue_deg)] strongest first."""
    im = Image.open(img_path).convert("HSV").resize((320, 180))
    a = np.asarray(im).reshape(-1, 3).astype(np.float32) / 255; h, s, v = a.T
    m = (s > 0.45) & (v > 0.55)
    hist = np.bincount((h[m] * 24).astype(int) % 24, weights=(s * v * v)[m], minlength=24)
    sm = hist   # 15-degree bins, unsmoothed: keeps pink (330) apart from red/coral (0)
    fams = []
    for i in range(24):
        if sm[i] > 0 and sm[i] >= sm[i - 1] and sm[i] >= sm[(i + 1) % 24]:
            idx = [(i - 1) % 24, i, (i + 1) % 24]; w = hist[idx]
            ang = np.angle((w * np.exp(1j * (np.array(idx) + 0.5) / 24 * 2 * np.pi)).sum()) / (2 * np.pi) * 360 % 360
            fams.append((float(w.sum()), float(ang)))
    return sorted(fams, reverse=True)

def palette(img_path, accent=None):
    """Neon palette from the image: accent (current word, glow, centre bars) leads with the strongest
    pink/magenta/purple family when the image has one; accent2 is a contrasting family (section label,
    outer bars); accent3 a warm coral/orange family (middle bars). --accent overrides the lead."""
    fams = _families(img_path)
    hsv = lambda hd, s_, v: tuple(int(round(x * 255)) for x in colorsys.hsv_to_rgb((hd % 360) / 360, s_, v))
    dist = lambda x, y: min(abs(x - y) % 360, 360 - abs(x - y) % 360)
    if not fams and not accent: return dict(DEFAULT)
    top = fams[0][0] if fams else 0
    if accent:
        a = hex2rgb(accent); ha = colorsys.rgb_to_hsv(*[x / 255 for x in a])[0] * 360
    else:
        pink = [f for f in fams if 270 <= f[1] < 350 and f[0] >= 0.15 * top]
        ha = (pink[0] if pink else fams[0])[1]; a = hsv(ha, 0.76, 1.0)
    f2 = [f for f in fams if dist(f[1], ha) >= 90]
    cool = [f for f in f2 if 150 <= f[1] < 260]
    h2 = (cool[0] if cool and cool[0][0] >= 0.3 * f2[0][0] else f2[0])[1] if f2 else ha + 180
    f3 = [f for f in fams if (f[1] < 50 or f[1] >= 350) and dist(f[1], ha) >= 20 and dist(f[1], h2) >= 40]
    h3 = f3[0][1] if f3 else ha + 30
    return dict(accent=a, accent2=hsv(h2, 0.65, 1.0), accent3=hsv(h3, 0.64, 1.0))

def mix(c, d, t): return tuple(int(round(c[i] * (1 - t) + d[i] * t)) for i in range(3))

# ---------------------------------------------------------------- audio analysis
def analyse(song, dur, nb):
    a, sr = sf.read(song, dtype="float32", always_2d=True); a = a.mean(1)
    n = 4096; hop = sr / FPS; nfr = int(np.ceil(dur * FPS)) + 2
    a = np.pad(a, (n // 2, n + int(hop) * 4))
    win = np.hanning(n).astype(np.float32); idx = (np.arange(nfr) * hop).astype(int)
    f = np.fft.rfftfreq(n, 1 / sr)
    edges = np.geomspace(45, min(14000, sr / 2 - 100), nb + 1)
    bi = np.searchsorted(f, edges)
    for j in range(1, len(bi)): bi[j] = max(bi[j], bi[j - 1] + 1)
    bl = (f > 35) & (f < 140)
    bands = np.empty((nfr, nb), np.float32); bass = np.empty(nfr, np.float32)
    for i in range(0, nfr, 256):
        fr = np.stack([a[j:j + n] for j in idx[i:i + 256]]) * win
        sp = np.abs(np.fft.rfft(fr))
        bands[i:i + 256] = np.stack([sp[:, bi[j]:bi[j + 1]].mean(1) for j in range(nb)], 1)
        bass[i:i + 256] = sp[:, bl].mean(1)
    db = 20 * np.log10(bands + 1e-6) + np.linspace(0, 14, nb)
    lo, hi = np.percentile(db, 25), np.percentile(db, 99.7)
    lv = np.clip((db - lo) / (hi - lo + 1e-6), 0, 1) ** 2.0
    sm = np.zeros_like(lv); sm[0] = lv[0]
    for i in range(1, len(lv)):
        p = sm[i - 1]; sm[i] = np.where(lv[i] > p, p + (lv[i] - p) * 0.7, p + (lv[i] - p) * 0.18)
    bd = 20 * np.log10(bass + 1e-6); l0, l1 = np.percentile(bd, 20), np.percentile(bd, 99)
    bass = np.clip((bd - l0) / (l1 - l0 + 1e-6), 0, 1) ** 2
    return sm, bass

def ease(x): return x * x * (3 - 2 * x)

# ---------------------------------------------------------------- the look
class Look:
    def __init__(self, work, lyrics, proj, aspect="16:9", title=None, artist=None, accent=None, focus=None):
        self.work = work; self.L = lyrics["lines"]; self.dur = proj["duration"]
        self.lay = LAYOUTS[aspect]; self.W, self.H = self.lay["W"], self.lay["H"]; self.aspect = aspect
        self.song = os.path.join(work, proj["song"]); self.img_path = os.path.join(work, proj["image"])
        self.title = title if title is not None else (proj.get("title") or os.path.splitext(proj.get("song_source_name", "Untitled"))[0].replace("_", " "))
        self.artist = artist if artist is not None else proj.get("artist")
        self.pal = palette(self.img_path, accent)
        self.focus = focus   # (x, y, r) normalised to image width/height, r relative to width; or None
        self.txdir = os.path.join(work, "cache", "tx"); os.makedirs(self.txdir, exist_ok=True)
        self.win = self._windows()
        self._ready = False; self._cache = {}

    # ---------- per-process heavy setup
    def setup(self):
        if self._ready: return
        W, H, lay = self.W, self.H, self.lay
        self.sm, self.bass = analyse(self.song, self.dur, lay["nb"])
        im = Image.open(self.img_path).convert("RGB"); self.sw, self.sh = im.size
        cover = max(W / self.sw, H / self.sh)
        self.k = max(1.0, cover * 1.35)          # supersampled source so the slow zoom stays smooth
        big = im.resize((round(self.sw * self.k), round(self.sh * self.k)), Image.LANCZOS)
        self.big = big.filter(ImageFilter.GaussianBlur(4.1 * self.k / cover))   # ~4 output px of blur
        y = np.linspace(0, 1, H)[:, None]; x = np.linspace(-1, 1, W)[None, :]
        t0 = lay["stack"] / H - 0.12
        v = 1 - 0.55 * np.clip((x ** 2 * 0.6 + ((y - 0.45) * 2) ** 2 * 0.5) - 0.15, 0, 1)
        bot = 1 - 0.78 * np.clip((y - t0) / (1 - t0), 0, 1) ** 1.3
        top = 1 - 0.35 * np.clip((0.18 - y) / 0.18, 0, 1)
        if lay["title"] == "top": top = 1 - 0.55 * np.clip((0.16 - y) / 0.16, 0, 1)
        self.grade = (0.80 * v * bot * top)[..., None].astype(np.float32)
        # spectrum colour field: accent centre -> accent3 -> accent2 at the outer bars
        span = (2 * lay["nb"] * (lay["bw"] + lay["gap"])) / W
        xx = np.clip(np.abs(np.linspace(-1, 1, W)) / span, 0, 1)[None, :, None]
        c0, c1, c2 = (np.array(self.pal[k], np.float32) for k in ("accent", "accent3", "accent2"))
        m1 = np.clip(xx / 0.45, 0, 1); m2 = np.clip((xx - 0.45) / 0.45, 0, 1)
        g = c0 * (1 - m1) + c1 * m1 * (1 - m2) + c2 * m2
        yy = np.linspace(0, H - 1, H)[:, None, None]
        self.sgrad = (np.broadcast_to(g, (H, W, 3)) * (0.75 + 0.25 * (1 - np.clip((lay["base"] - yy) / lay["hmax"], 0, 1)))).astype(np.float32)
        # generic beat glow behind the lyric (used when no --focus point)
        cy = lay["stack"] + lay["font"] * 0.9
        gy, gx = np.mgrid[0:H, 0:W]
        self.bloom = np.exp(-(((gx - W / 2) / (W * 0.38)) ** 2 + ((gy - cy) / (lay["font"] * 1.6)) ** 2)).astype(np.float32)[..., None]
        self.ui = self._shot(self._ui_html())
        self._ready = True

    # ---------- background
    def kb(self, t):
        u = ease(min(1, max(0, t / self.dur))); z = 1.05 + 0.10 * u
        fx, fy = (self.focus[0] * self.sw, self.focus[1] * self.sh) if self.focus else (self.sw / 2, self.sh / 2)
        if self.W / self.H < self.sw / self.sh - 0.05:   # narrower frame than the image: keep the focus in frame
            cx = fx - 6 * np.sin(t / 9)
        else:
            cx = self.sw / 2 + (fx - self.sw / 2) * 0.25 * u - 10 * np.sin(t / 9)
        cy = self.sh / 2 + (fy - self.sh / 2) * 0.30 * u
        return cx, cy, z

    def bg(self, t):
        cx, cy, z = self.kb(t); W, H = self.W, self.H
        s = max(W / self.sw, H / self.sh) * z; w, h = W / s, H / s
        cx = min(max(cx, w / 2), self.sw - w / 2); cy = min(max(cy, h / 2), self.sh - h / 2)
        k = self.k
        fr = self.big.resize((W, H), Image.BICUBIC, box=[(cx - w / 2) * k, (cy - h / 2) * k, (cx + w / 2) * k, (cy + h / 2) * k])
        return np.asarray(fr, np.float32), (s, cx, cy)

    def glow(self, img, b, kbi):
        acc = np.array(self.pal["accent"], np.float32)
        if not self.focus:
            img += self.bloom * acc * (0.05 + 0.22 * b); return
        s, cx, cy = kbi; W, H = self.W, self.H
        x = W / 2 + (self.focus[0] * self.sw - cx) * s; y = H / 2 + (self.focus[1] * self.sh - cy) * s
        r = self.focus[2] * self.sw * s
        x0, x1, y0, y1 = int(max(0, x - 3 * r)), int(min(W, x + 3 * r)), int(max(0, y - 3 * r)), int(min(H, y + 3 * r))
        if x1 <= x0 or y1 <= y0: return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / r; rr = 1.02 + 0.10 * b
        gl = np.exp(-((d - rr) / (0.22 + 0.25 * b)) ** 2) * (0.25 + 0.75 * b) * 0.55 + np.exp(-(d / 1.6) ** 2) * 0.10 * (0.4 + b)
        img[y0:y1, x0:x1] += gl[..., None] * acc * 0.9

    def spectrum(self, img, lv):
        lay = self.lay; W, H = self.W, self.H; BW, GAP, BASE, HM = lay["bw"], lay["gap"], lay["base"], lay["hmax"]
        m = Image.new("L", (W, H)); d = ImageDraw.Draw(m)
        vals = np.concatenate([lv[::-1], lv]); total = len(vals) * (BW + GAP) - GAP; x0 = (W - total) / 2
        for i, v in enumerate(vals):
            h = 4 + v * HM; x = x0 + i * (BW + GAP)
            edge = 1 - abs(i - len(vals) / 2 + 0.5) / (len(vals) / 2); a = int(255 * (0.55 + 0.45 * edge ** 0.5))
            d.rounded_rectangle([x, BASE - h, x + BW, BASE], radius=BW // 2, fill=a)
            d.rounded_rectangle([x, BASE + 5, x + BW, BASE + 5 + h * 0.28], radius=BW // 2, fill=int(a * 0.16))
        y0 = max(0, int(BASE - HM - 60)); y1 = min(H, int(BASE + HM * 0.3 + 60))   # only touch the bar band
        band = m.crop((0, y0, W, y1))
        mk = np.asarray(band, np.float32)[..., None] / 255
        gl = np.asarray(band.resize((W // 4, (y1 - y0) // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(6))
                        .resize((W, y1 - y0), Image.BILINEAR), np.float32)[..., None] / 255
        g = self.sgrad[y0:y1]; sub = img[y0:y1]
        sub += g * gl * 1.3
        img[y0:y1] = sub * (1 - mk) + g * mk * 1.08 + 255 * mk * 0.12

    def progress(self, img, t):
        x1 = int(self.W * min(1, t / self.dur)); H = self.H
        img[H - 4:H - 1, :] *= 0.4
        img[H - 4:H - 1, :x1] = img[H - 4:H - 1, :x1] * 0.2 + np.array(self.pal["accent"], np.float32) * 0.9

    # ---------- text via Chrome
    def css(self):
        if getattr(self, "_css", None): return self._css
        self._css = self._make_css(); return self._css

    def _make_css(self):
        lay = self.lay; a = self.pal["accent"]; a2 = self.pal["accent2"]
        A = "%d,%d,%d" % a; A2 = "%d,%d,%d" % a2; cur = rgb2hex(mix(a, (255, 255, 255), 0.86)); sub = mix(a, (255, 255, 255), 0.55)
        pad = (self.W - lay["maxw"]) // 2
        tpos = (f"left:{lay['tx']}px;bottom:{lay['ty']}px" if lay["title"] == "bottom" else f"left:{lay['tx']}px;top:{lay['ty']}px")
        return f"""
@font-face{{font-family:I;src:url(file://{FONT});font-weight:100 900}}
html,body{{margin:0;width:{self.W}px;height:{self.H}px;background:transparent;font-family:I;overflow:hidden}}
.stack{{position:absolute;left:0;right:0;top:{lay['stack']}px;text-align:center}}
.sec{{font-weight:700;font-size:{lay['secf']}px;letter-spacing:.55em;text-transform:uppercase;color:rgba({A2},.88);
 text-shadow:0 0 12px rgba({A2},.6);height:{lay['secf']*2.4}px;line-height:{lay['secf']*2.4}px}}
.sec i{{display:inline-block;width:7px;height:7px;border-radius:50%;background:rgb({A});box-shadow:0 0 10px rgb({A});margin-right:14px;vertical-align:2px}}
.line{{font-weight:850;font-size:{lay['font']}px;letter-spacing:-0.025em;text-transform:uppercase;color:#fff;line-height:1.08;padding:0 {pad}px;margin-top:10px}}
#in{{display:inline-block;max-width:{lay['maxw']}px;text-wrap:balance}}
.w{{display:inline-block;margin:0 .13em}}
.p{{color:#fff;text-shadow:0 0 18px rgba({A},.35),0 4px 18px rgba(0,0,0,.55)}}
.f{{color:rgba(255,255,255,.34);text-shadow:0 4px 18px rgba(0,0,0,.4)}}
.c{{color:{cur};transform:scale(1.08);margin:0 .2em;
 text-shadow:0 0 4px #fff,0 0 12px rgb({A}),0 0 26px rgb({A}),0 0 52px rgb({A}),0 0 90px rgb({A}),0 0 160px rgba({A},.7)}}
.next{{font-weight:600;font-size:{lay['nextf']}px;letter-spacing:.16em;text-transform:uppercase;color:rgba(255,255,255,.30);
 margin-top:{lay['nextf']}px;padding:0 {pad}px}}
.title{{position:absolute;{tpos};color:#fff;max-width:{self.W - 2*lay['tx']}px}}
.title b{{display:block;font-weight:800;font-size:{lay['titlef']}px;letter-spacing:.28em;text-transform:uppercase}}
.title span{{display:block;font-weight:500;font-size:{round(lay['titlef']*0.6)}px;letter-spacing:.42em;text-transform:uppercase;color:rgba({sub[0]},{sub[1]},{sub[2]},.85);margin-top:8px}}
.title:before{{content:"";position:absolute;left:-20px;top:3px;bottom:3px;width:3px;border-radius:2px;background:linear-gradient(rgb({A2}),rgb({A}))}}
"""

    def _ui_html(self):
        sub = self.artist or "Lyric video"
        return f'<div class="title"><b>{html.escape(self.title or "")}</b><span>{html.escape(sub)}</span></div>'

    def text_html(self, li, cur):
        L = self.L[li]; lay = self.lay; sp = []
        for k, w in enumerate(L["words"]):
            cls = "p" if k < cur else ("c" if k == cur else "f")
            sp.append(f'<span class="w {cls}">{html.escape(w["text"])}</span>')
        nxt = self.L[li + 1]["text"] if li + 1 < len(self.L) and self.L[li + 1]["start"] - L["end"] < 6 else ""
        sec = f'<i></i>{html.escape(L["section"])}' if L.get("section") else ""
        # shrink until the line fits in `rows` rows and maxw
        fit = (f'<script>var e=document.getElementById("in"),l=e.parentNode,s={lay["font"]},lh=s*1.08;'
               f'while((e.offsetHeight>{lay["rows"]}*s*1.08*1.15||e.scrollWidth>{lay["maxw"]})&&s>30){{s-=2;l.style.fontSize=s+"px";}}</script>')
        return (f'<div class="stack"><div class="sec">{sec}</div><div class="line"><span id="in">{"".join(sp)}</span></div>'
                f'<div class="next">{html.escape(nxt)}</div></div>{fit}')

    def _doc(self, body):
        return f"<!doctype html><html><head><meta charset=utf-8><style>{self.css()}</style></head><body>{body}</body></html>"

    def _path(self, body):
        h = hashlib.sha1(self._doc(body).encode()).hexdigest()[:16]
        return os.path.join(self.txdir, h + ".png")

    def _shot(self, body):
        out = self._path(body)
        if os.path.exists(out): return out
        hp = out + ".html"; open(hp, "w").write(self._doc(body))
        for attempt in range(3):
            subprocess.run([chrome_bin(), "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--default-background-color=00000000",
                            f"--window-size={self.W},{self.H}", "--virtual-time-budget=1500", f"--screenshot={out}.tmp.png", "file://" + hp],
                           capture_output=True, timeout=90)
            if os.path.exists(out + ".tmp.png"): os.replace(out + ".tmp.png", out); break
        os.remove(hp)
        if not os.path.exists(out): raise RuntimeError("chrome screenshot failed")
        return out

    def text_jobs(self):
        return [self.text_html(i, k) for i, l in enumerate(self.L) for k in range(-1, len(l["words"]) + 1)]

    # ---------- timing of text states
    LEAD, HOLD, FADE_IN, FADE_OUT, RAMP = 0.30, 1.0, 0.22, 0.45, 0.09
    def _windows(self):
        W = []
        for i, l in enumerate(self.L):
            a = l["start"] - self.LEAD
            nxt = self.L[i + 1]["start"] - self.LEAD if i + 1 < len(self.L) else self.dur
            b = min(nxt, l["end"] + self.HOLD)
            W.append((a, b, b < nxt - 0.05))
        return W

    def text_state(self, t):
        for i, (a, b, gap) in enumerate(self.win):
            if a <= t < b:
                ws = self.L[i]["words"]; al = 1.0
                if i == 0 or self.win[i - 1][1] < a - 0.01: al = min(al, (t - a) / self.FADE_IN)
                if gap: al = min(al, (b - t) / self.FADE_OUT)
                al = max(0.0, min(1.0, al))
                k = -1
                for j, w in enumerate(ws):
                    if w["start"] <= t: k = j
                if k == len(ws) - 1 and t > ws[-1]["end"] + 0.05: k = len(ws)
                ts = ws[k]["start"] if 0 <= k < len(ws) else (ws[-1]["end"] + 0.05 if k == len(ws) else None)
                if ts is not None and t - ts < self.RAMP:
                    r = ease((t - ts) / self.RAMP)
                    return [(self.text_html(i, k - 1), al * (1 - r)), (self.text_html(i, k), al * r)]
                return [(self.text_html(i, k), al)]
        return []

    def _rgba(self, p):
        if p not in self._cache:
            if len(self._cache) > 6: self._cache.pop(next(iter(self._cache)))
            a = np.asarray(Image.open(p).convert("RGBA"), np.float32); m = a[..., 3:4] / 255
            self._cache[p] = (a[..., :3] * m, m)
        return self._cache[p]

    # ---------- one frame
    def frame(self, t):
        self.setup()
        f = min(int(round(t * FPS)), len(self.bass) - 1)
        img, kbi = self.bg(t); b = float(self.bass[f])
        img *= self.grade * (1 + 0.05 * b)
        self.glow(img, b, kbi)
        self.spectrum(img, self.sm[f]); self.progress(img, t)
        c, m = self._rgba(self.ui); img[:] = img * (1 - m) + c
        layers = [(self._path(b), al) for b, al in self.text_state(t)]
        if len(layers) == 2:
            (p0, a0), (p1, a1) = layers; tot = a0 + a1; r = a1 / tot if tot else 0
            c0, m0 = self._rgba(p0); c1, m1 = self._rgba(p1)
            img[:] = img * (1 - (m0 * (1 - r) + m1 * r) * tot) + (c0 * (1 - r) + c1 * r) * tot
        elif layers:
            (p, al), = layers; c, m = self._rgba(p); img[:] = img * (1 - m * al) + c * al
        return np.clip(img, 0, 255).astype(np.uint8)

    def best_still_time(self, t0=0.0, t1=None):
        """Mid-chorus moment with the line fully visible (falls back to the most repeated line, then the middle)."""
        t1 = self.dur if t1 is None else t1
        cand = [i for i, l in enumerate(self.L) if t0 + 0.5 <= l["start"] and l["end"] <= t1 - 0.3 and len(l["words"]) >= 3]
        if not cand: cand = [i for i, l in enumerate(self.L) if t0 <= l["start"] < t1]
        if not cand: return (t0 + t1) / 2
        ch = [i for i in cand if "chorus" in (self.L[i]["section"] or "").lower() and "pre" not in self.L[i]["section"].lower()]
        if not ch:
            from collections import Counter
            cnt = Counter(self.L[i]["text"].lower() for i in cand); top = max(cnt.values())
            ch = [i for i in cand if cnt[self.L[i]["text"].lower()] == top] if top > 1 else cand
        i = ch[len(ch) // 2] if len(ch) > 2 else ch[0]
        ws = self.L[i]["words"]; k = len(ws) // 2
        return ws[k]["start"] + self.RAMP + 0.03
