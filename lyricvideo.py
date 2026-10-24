#!/usr/bin/env python3
"""Lyric Video Kit CLI.

  prep     song + image -> vocals, word timings, lyrics.json/lyrics.txt/low_confidence.txt
  preview  one PNG frame in the final look
  render   MP4 (H.264 + AAC, faststart) + a clean still + qa.json
  qa       re-run QA on an MP4
  doctor   check the tools are installed
"""
import argparse, os, sys, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def look_args(p):
    p.add_argument("--work", required=True, help="work folder from prep")
    p.add_argument("--aspect", default="16:9", choices=["16:9", "9:16", "1:1"])
    p.add_argument("--title", help="override the title (default: file tag, else file name)")
    p.add_argument("--artist", help="override the artist (default: file tag)")
    p.add_argument("--accent", help="accent colour '#rrggbb' (default: picked from the image)")
    p.add_argument("--focus", help="x,y[,r] point of interest in the image, 0-1 (e.g. 0.5,0.6,0.07): "
                                   "slow zoom drifts there and the beat glow rings it")
    p.add_argument("--workers", type=int, default=0, help="parallel workers (default: auto, max 4, ~0.6 GB each)")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep", help="separate vocals, transcribe, build/align lyrics")
    p.add_argument("--work", required=True); p.add_argument("--song"); p.add_argument("--image")
    p.add_argument("--lyrics", help="pasted lyric sheet (.txt); its words are aligned to the sung timing")
    p.add_argument("--relyric", action="store_true", help="re-align after editing WORK/lyrics.txt (or --lyrics FILE); no re-transcription")
    p.add_argument("--model", default="large-v3-turbo", help="faster-whisper model (large-v3-turbo, large-v3, medium.en, small.en)")
    p.add_argument("--no-separate", action="store_true", help="skip demucs and transcribe the full mix")
    p.add_argument("--retranscribe", action="store_true", help="ignore a cached whisper.json")
    p = sp.add_parser("preview", help="render one PNG frame"); look_args(p)
    p.add_argument("--t", help="time in seconds or m:ss (default: a mid-chorus moment)"); p.add_argument("--out")
    p = sp.add_parser("render", help="render the MP4, a still and qa.json"); look_args(p)
    p.add_argument("--start", help="start time (s or m:ss) for a partial render"); p.add_argument("--end", help="end time")
    p.add_argument("--name", help="output file stem (default lyric_video_<aspect>[_range])")
    p = sp.add_parser("qa", help="QA an MP4"); p.add_argument("mp4"); p.add_argument("--expect", type=float)
    sp.add_parser("doctor", help="check dependencies")
    a = ap.parse_args()
    if a.cmd == "prep":
        if not a.relyric and not (a.song or os.path.exists(os.path.join(a.work, "project.json"))): ap.error("prep needs --song and --image")
        from lvkit import prep; prep.run(a)
    elif a.cmd == "preview":
        from lvkit import render; render.preview(a)
    elif a.cmd == "render":
        from lvkit import render; render.render(a)
    elif a.cmd == "qa":
        import json; from lvkit import qa; print(json.dumps(qa.check(a.mp4, expect=a.expect), indent=1))
    elif a.cmd == "doctor":
        doctor()

def doctor():
    from lvkit.util import chrome_bin, FONT, mem_avail_gb
    ok = True
    def chk(name, cond, hint=""):
        nonlocal ok; print(("ok   " if cond else "MISS ") + name + ("" if cond else "  -> " + hint)); ok &= bool(cond)
    chk("ffmpeg", shutil.which("ffmpeg"), "apt-get install ffmpeg")
    chk("ffprobe", shutil.which("ffprobe"), "apt-get install ffmpeg")
    for m in ("numpy", "PIL", "soundfile", "faster_whisper", "demucs", "torch"):
        try: __import__(m); chk(m, True)
        except Exception as e: chk(m, False, f"pip install (see install.sh): {e}")
    chk("Inter font", os.path.exists(FONT), "fonts/Inter-Variable.ttf missing")
    try:
        c = chrome_bin(); r = subprocess.run([c, "--version"], capture_output=True, text=True, timeout=20)
        chk(f"headless chrome ({c})", r.returncode == 0, r.stderr[-200:])
    except SystemExit: chk("headless chrome", False, "run install.sh")
    print(f"info cpu={os.cpu_count()} mem_available={mem_avail_gb():.1f} GB")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
