"""preview + render: text PNGs via Chrome in parallel, frames composited in N worker processes,
each piping raw RGB into its own x264 encode; segments concatenated and muxed with the original audio."""
import os, sys, math, time, json, subprocess, shutil
from multiprocessing import Pool
import numpy as np
from PIL import Image
from .util import log, die, load_json, save_json, mem_avail_gb, parse_time, mmss
from .look import Look, FPS, LAYOUTS
from . import qa

_LOOK = None
def _init(look):
    global _LOOK
    _LOOK = look; _LOOK.setup()

def _shot(body): return _LOOK._shot(body)

def _seg(job):
    w, f0, f1, out, vb = job; L = _LOOK
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{L.W}x{L.H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-maxrate", f"{vb}k", "-bufsize", f"{2*vb}k",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-g", "60", "-threads", "2",
           "-x264-params", "log-level=error:rc-lookahead=20", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE); T = time.time()
    for f in range(f0, f1):
        p.stdin.write(L.frame(f / FPS).tobytes())
        if (f - f0) % 300 == 299: log(f"  worker {w}: {f-f0+1}/{f1-f0} frames, {time.time()-T:.0f}s")
    p.stdin.close(); p.wait()
    if p.returncode: raise RuntimeError(f"x264 failed for segment {w}")
    return out

def load(args):
    work = os.path.abspath(args.work)
    proj = load_json(os.path.join(work, "project.json")) or die("no project.json; run prep first")
    ly = load_json(os.path.join(work, "lyrics.json")) or die("no lyrics.json; run prep first")
    focus = tuple(float(x) for x in args.focus.split(",")) if getattr(args, "focus", None) else None
    if focus and len(focus) == 2: focus = focus + (0.07,)
    look = Look(work, ly, proj, aspect=args.aspect, title=args.title, artist=args.artist, accent=args.accent, focus=focus)
    return work, proj, look

def default_workers(n):
    if n: return n
    return max(1, min(4, (os.cpu_count() or 2) - 1, int(mem_avail_gb() / 0.6)))

def make_text(look, workers):
    T = time.time(); jobs = look.text_jobs()
    todo = [j for j in jobs if not os.path.exists(look._path(j))]
    log(f"text layers: {len(jobs)} ({len(todo)} new) via headless Chrome ...")
    if todo:
        with Pool(workers, initializer=_init_light, initargs=(look,)) as P: P.map(_shot, todo)
    look._shot(look._ui_html())
    log(f"text layers ready in {time.time()-T:.0f}s")

def _init_light(look):
    global _LOOK
    _LOOK = look

def preview(args):
    work, proj, look = load(args)
    t = parse_time(args.t) if args.t is not None else look.best_still_time()
    for body, _ in look.text_state(t): look._shot(body)     # only the layers this frame needs
    look._shot(look._ui_html())
    out = args.out or os.path.join(work, "out", f"preview_{args.aspect.replace(':','x')}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(look.frame(t)).save(out)
    log(f"preview at {t:.2f}s ({mmss(t)}): {out}")
    return out

def render(args):
    work, proj, look = load(args)
    t0 = parse_time(args.start) or 0.0; t1 = parse_time(args.end) if args.end is not None else proj["duration"]
    t1 = min(t1, proj["duration"])
    if t1 - t0 < 1: die("render range shorter than 1 s")
    nw = default_workers(args.workers)
    log(f"render {args.aspect} {look.W}x{look.H}, {mmss(t0)}-{mmss(t1)} ({t1-t0:.1f}s), {nw} workers, "
        f"accent {look.pal['accent']} title={look.title!r} artist={look.artist!r}")
    T = time.time(); make_text(look, nw)
    f0, f1 = int(round(t0 * FPS)), int(math.ceil(t1 * FPS)); NF = f1 - f0
    clipd = t1 - t0
    vb = int(max(1200, min(4000, 0.92 * (48e6 * 8 / clipd) / 1000 - 192)))   # keep files under ~50 MB
    tag = args.name or (f"lyric_video_{args.aspect.replace(':','x')}" + ("" if (t0 == 0 and t1 >= proj["duration"] - 0.01) else f"_{int(t0)}-{int(t1)}s"))
    outd = os.path.join(work, "out"); tmpd = os.path.join(outd, "_seg_" + tag); os.makedirs(tmpd, exist_ok=True)
    jobs = [(w, f0 + w * NF // nw, f0 + (w + 1) * NF // nw, os.path.join(tmpd, f"seg{w}.mp4"), vb) for w in range(nw)]
    log(f"compositing {NF} frames ...")
    with Pool(nw, initializer=_init, initargs=(look,)) as P: segs = P.map(_seg, jobs)
    lst = os.path.join(tmpd, "segs.txt"); open(lst, "w").write("".join(f"file '{s}'\n" for s in segs))
    final = os.path.join(outd, tag + ".mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", f"{t0:.3f}", "-t", f"{clipd:.3f}", "-i", look.song,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-t", f"{clipd:.3f}", "-movflags", "+faststart", final], check=True)
    shutil.rmtree(tmpd, ignore_errors=True)
    ts = look.best_still_time(t0, t1)
    still = os.path.join(outd, tag + "_still.png"); Image.fromarray(look.frame(ts)).save(still)
    log(f"video done in {time.time()-T:.0f}s: {final}\nstill at {ts:.2f}s: {still}")
    rep = qa.check(final, look.song, t0, clipd, os.path.join(outd, tag + "_qa.json"))
    rep["still"] = still; rep["still_time"] = round(ts, 2); rep["render_seconds"] = round(time.time() - T, 1)
    save_json(os.path.join(outd, tag + "_qa.json"), rep)
    log(json.dumps(rep, indent=1))
    return final
