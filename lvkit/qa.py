"""Built-in QA for a rendered MP4: duration, decode, black/frozen stretches, faststart, peak/loudness."""
import json, re, subprocess, sys
from .util import ffprobe, save_json, log

def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def loudness(path, ss=None, t=None):
    cmd = ["ffmpeg", "-hide_banner", "-nostats"]
    if ss is not None: cmd += ["-ss", f"{ss:.3f}"]
    if t is not None: cmd += ["-t", f"{t:.3f}"]
    cmd += ["-i", path, "-vn", "-af", "astats=measure_perchannel=none:measure_overall=Peak_level,ebur128=peak=true", "-f", "null", "-"]
    e = _run(cmd).stderr
    pk = re.findall(r"Peak level dB:\s*(-?[\d.]+|-inf)", e); tp = re.findall(r"Peak:\s*(-?[\d.]+|-inf)\s*dBFS", e)
    li = re.findall(r"I:\s*(-?[\d.]+) LUFS", e)
    f = lambda v: float(v[-1]) if v and v[-1] != "-inf" else None
    return dict(sample_peak_dbfs=f(pk), true_peak_dbfs=f(tp), integrated_lufs=f(li))

def check(mp4, source_audio=None, src_start=0.0, expect=None, out_json=None):
    d = ffprobe(mp4); v = [s for s in d["streams"] if s["codec_type"] == "video"]; a = [s for s in d["streams"] if s["codec_type"] == "audio"]
    r = {"file": mp4, "size_mb": round(int(d["format"]["size"]) / 1e6, 2), "duration": round(float(d["format"]["duration"]), 3)}
    issues = []
    if not v or not a: issues.append("missing video or audio stream")
    if v:
        r["video"] = dict(codec=v[0]["codec_name"], profile=v[0].get("profile"), size=f"{v[0]['width']}x{v[0]['height']}",
                          fps=v[0]["r_frame_rate"], frames=int(v[0].get("nb_frames", 0)), duration=float(v[0].get("duration", 0)),
                          pix_fmt=v[0].get("pix_fmt"))
        if v[0]["codec_name"] != "h264": issues.append("video is not H.264")
    if a:
        r["audio"] = dict(codec=a[0]["codec_name"], rate=a[0]["sample_rate"], channels=a[0]["channels"], duration=float(a[0].get("duration", 0)))
        if a[0]["codec_name"] != "aac": issues.append("audio is not AAC")
    if expect:
        r["expected_duration"] = round(expect, 3)
        for k in ("video", "audio"):
            if k in r and abs(r[k]["duration"] - expect) > 0.1: issues.append(f"{k} duration {r[k]['duration']:.3f}s vs expected {expect:.3f}s")
    head = open(mp4, "rb").read(1 << 20); mo, md = head.find(b"moov"), head.find(b"mdat")
    r["faststart"] = mo != -1 and (md == -1 or mo < md)
    if not r["faststart"]: issues.append("moov atom not at the front (no faststart)")
    e = _run(["ffmpeg", "-v", "error", "-i", mp4, "-f", "null", "-"]).stderr.strip()
    r["decode_errors"] = len(e.splitlines()) if e else 0
    if e: issues.append("decode errors: " + e.splitlines()[0][:200])
    e = _run(["ffmpeg", "-hide_banner", "-i", mp4, "-an", "-vf", "blackdetect=d=0.5:pix_th=0.06,freezedetect=n=0.001:d=6",
              "-f", "null", "-"]).stderr
    r["black_stretches"] = re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", e)
    r["frozen_stretches"] = re.findall(r"freeze_start: ([\d.]+)", e)
    if r["black_stretches"]: issues.append(f"black stretches: {r['black_stretches'][:3]}")
    if r["frozen_stretches"]: issues.append(f"frozen stretches from: {r['frozen_stretches'][:3]}")
    r["audio_out"] = loudness(mp4)
    if source_audio: r["audio_source"] = loudness(source_audio, src_start, expect)
    tp = r["audio_out"]["true_peak_dbfs"]
    if tp is not None and tp > -0.1: issues.append(f"output true peak {tp} dBFS: possible clipping")
    r["issues"] = issues; r["pass"] = not issues
    if out_json: save_json(out_json, r)
    return r

if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1]), indent=1))
