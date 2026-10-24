import os, glob, json, shutil, subprocess, sys

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = os.path.join(KIT, "fonts", "Inter-Variable.ttf")
CACHE = os.path.expanduser(os.environ.get("LVK_CACHE", "~/.cache/lyric-video-kit"))

def log(*a):
    print(*a, flush=True)

def die(msg):
    print("error: " + msg, file=sys.stderr); sys.exit(2)

def chrome_bin():
    """Standalone chrome-headless-shell (system Chrome can hang on headless screenshots)."""
    env = os.environ.get("LVK_CHROME")
    if env and os.path.exists(env): return env
    pats = [os.path.join(CACHE, "chrome-headless-shell", "*", "chrome-headless-shell"),
            os.path.join(CACHE, "chrome-headless-shell", "*", "*", "chrome-headless-shell"),
            os.path.expanduser("~/.cache/chs/chrome-headless-shell/*/chrome-headless-shell-linux64/chrome-headless-shell"),
            os.path.expanduser("~/.cache/puppeteer/chrome-headless-shell/*/chrome-headless-shell-linux64/chrome-headless-shell")]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits: return hits[-1]
    for b in ("chrome-headless-shell", "chromium", "chromium-browser", "google-chrome"):
        if shutil.which(b): return shutil.which(b)
    die("no headless Chrome found; run install.sh (or set LVK_CHROME)")

def ffprobe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
                       capture_output=True, text=True)
    if r.returncode: die(f"ffprobe failed on {path}: {r.stderr.strip()}")
    return json.loads(r.stdout)

def song_meta(path):
    d = ffprobe(path); fmt = d.get("format", {})
    tags = {}
    for src in [fmt.get("tags", {})] + [s.get("tags", {}) for s in d.get("streams", [])]:
        for k, v in (src or {}).items(): tags.setdefault(k.lower(), v)
    dur = float(fmt.get("duration") or max(float(s.get("duration", 0)) for s in d["streams"]))
    return {"duration": round(dur, 3), "title": tags.get("title"),
            "artist": tags.get("artist") or tags.get("album_artist") or tags.get("performer"), "album": tags.get("album")}

def load_json(p, default=None):
    if not os.path.exists(p): return default
    return json.load(open(p))

def save_json(p, obj):
    tmp = p + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1, ensure_ascii=False); os.replace(tmp, p)

def mmss(t):
    t = max(0, t); return "%02d:%02d" % (int(t) // 60, int(t) % 60)

def parse_time(s):
    if s is None: return None
    s = str(s)
    if ":" in s:
        m, sec = s.rsplit(":", 1); return int(m) * 60 + float(sec)
    return float(s)

def mem_avail_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"): return int(line.split()[1]) / 1e6
    except OSError: pass
    return 4.0
