"""prep: copy inputs, separate vocals (demucs), transcribe (faster-whisper word timestamps),
build timed lines from the transcript or align user-pasted lyrics, flag low-confidence lines."""
import os, re, shutil, subprocess, sys, time, difflib
import numpy as np
from .util import log, die, song_meta, save_json, load_json, mmss

NONLATIN = re.compile(r"[^\x00-\x7F\u00C0-\u024F\u2019\u2018]")
JUNK = re.compile(r"(thanks? for watching|subscribe|we'?ll be right back|subtitles? by|displayed on screen)", re.I)

def norm(s): return re.sub(r"[^a-z0-9']", "", s.lower().replace("\u2019", "'"))

# ---------------------------------------------------------------- inputs
def copy_inputs(work, song, image):
    os.makedirs(work, exist_ok=True)
    proj = load_json(os.path.join(work, "project.json"), {})
    if song:
        if not os.path.exists(song): die(f"song not found: {song}")
        ext = os.path.splitext(song)[1].lower() or ".wav"
        dst = os.path.join(work, "song" + ext)
        if os.path.abspath(song) != os.path.abspath(dst): shutil.copyfile(song, dst)
        proj["song"] = os.path.basename(dst); proj["song_source_name"] = os.path.basename(song)
        proj.update({k: v for k, v in song_meta(dst).items()})
    if image:
        if not os.path.exists(image): die(f"image not found: {image}")
        from PIL import Image
        Image.open(image).convert("RGB").save(os.path.join(work, "image.png"))
        proj["image"] = "image.png"
    if "song" not in proj: die("no song yet: pass --song")
    if "image" not in proj: die("no image yet: pass --image")
    save_json(os.path.join(work, "project.json"), proj)
    return proj

# ---------------------------------------------------------------- audio
def decode16k(src, dst):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-c:a", "flac", dst], check=True)

def separate(work, song_path, skip=False):
    out = os.path.join(work, "vocals.wav")
    if skip:
        log("vocal separation skipped (--no-separate); transcribing the full mix"); return song_path
    if os.path.exists(out): log("vocals.wav exists, reusing"); return out
    log("separating vocals with demucs htdemucs (CPU, ~1 min per 3 min of song) ...")
    T = time.time(); tmp = os.path.join(work, "_sep")
    r = subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-j", "1", "--segment", "7",
                        "-o", tmp, song_path], capture_output=True, text=True)
    stem = os.path.splitext(os.path.basename(song_path))[0]
    v = os.path.join(tmp, "htdemucs", stem, "vocals.wav")
    if r.returncode or not os.path.exists(v):
        log("demucs failed, falling back to the full mix:\n" + r.stderr[-800:]); return song_path
    shutil.move(v, out); shutil.rmtree(tmp, ignore_errors=True)
    log(f"vocals separated in {time.time()-T:.0f}s"); return out

def transcribe(work, vocals, model="large-v3-turbo", prompt=None):
    import soundfile as sf
    from faster_whisper import WhisperModel
    flac = os.path.join(work, "vocals16k.flac"); decode16k(vocals, flac)
    audio = sf.read(flac, dtype="float32")[0]   # numpy in: avoids PyAV version issues
    log(f"transcribing with faster-whisper {model} (int8, CPU) ...")
    T = time.time()
    m = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    segs, info = m.transcribe(audio, language=None if os.environ.get("LVK_LANG") == "auto" else os.environ.get("LVK_LANG", "en"),
                              word_timestamps=True, vad_filter=True,
                              vad_parameters=dict(threshold=0.3, min_silence_duration_ms=400),
                              condition_on_previous_text=False, beam_size=5, initial_prompt=prompt)
    out = []
    for s in segs:
        out.append(dict(start=s.start, end=s.end, text=s.text.strip(),
                        words=[dict(w=w.word.strip(), start=round(w.start, 3), end=round(w.end, 3), p=round(w.probability, 3))
                               for w in s.words if w.word.strip()]))
    save_json(os.path.join(work, "whisper.json"), {"model": model, "segments": out})
    log(f"transcribed in {time.time()-T:.0f}s, {sum(len(s['words']) for s in out)} words")
    return out

def vocal_activity(work):
    """50 ms vocal RMS frames -> (times, active bool)."""
    import soundfile as sf
    p = os.path.join(work, "vocals16k.flac")
    if not os.path.exists(p): return None
    a, sr = sf.read(p, dtype="float32"); h = int(sr * 0.05)
    r = np.sqrt(np.convolve(a ** 2, np.ones(h) / h, "same"))[::h]; db = 20 * np.log10(r + 1e-6)
    return db > db.max() - 30

# ---------------------------------------------------------------- whisper -> words
def clean_words(segs):
    W = []
    for s in segs:
        if JUNK.search(s["text"]): continue
        for w in s["words"]:
            if NONLATIN.search(w["w"]) or not norm(w["w"]): continue
            W.append(dict(w))
    for i in range(1, len(W)):
        if W[i]["start"] < W[i - 1]["start"] + 0.04: W[i]["start"] = W[i - 1]["start"] + 0.04
    return W

def lines_from_transcript(segs, max_words=9, max_chars=40):
    W = clean_words(segs); lines = []; cur = []
    def flush():
        if cur: lines.append(dict(section="", toks=[w["w"] for w in cur], starts=[w["start"] for w in cur],
                                  probs=[w["p"] for w in cur], matched=[True] * len(cur)))
        cur.clear()
    for i, w in enumerate(W):
        if cur:
            gap = w["start"] - W[i - 1]["end"]; text = " ".join(x["w"] for x in cur)
            prev = cur[-1]["w"]
            if (gap > 0.7 or len(cur) >= max_words or len(text) + len(w["w"]) > max_chars or re.search(r"[.!?]$", prev)
                    or (re.search(r"[,;:]$", prev) and len(cur) >= 4)
                    or (w["w"][:1].isupper() and not re.match(r"^I('|$)", w["w"]) and len(cur) >= 3)):  # whisper capitalises new lyric lines
                flush()
        cur.append(w)
    flush()
    lines = [l for l in lines if max(l["probs"]) >= 0.3]   # drop lines made only of near-certain mishearings
    for l in lines:
        l["toks"][0] = l["toks"][0][:1].upper() + l["toks"][0][1:]
        l["toks"][-1] = l["toks"][-1].rstrip(".,;:")
    return lines

# ---------------------------------------------------------------- pasted lyrics
TS = re.compile(r"^\s*\[?\d{1,2}:\d{2}(?:[.:]\d+)?\]?\s*")
def parse_lyrics(text, title=None):
    """Lyric sheet -> [(section, line)]. Accepts [Section] headers, optional leading mm:ss stamps, '(?)' flags."""
    out = []; sec = ""
    for raw in text.splitlines():
        s = raw.strip()
        if not s: continue
        m = re.match(r"^\[([^\]]+)\]\s*(\d{1,2}:\d{2})?\s*$", s)
        if m and not re.match(r"^\d", m.group(1)):
            sec = m.group(1).strip()
            if sec.lower().startswith("instrumental"): sec = ""
            continue
        if s.startswith("#"): continue
        s = TS.sub("", s); s = re.sub(r"\s*\(\?\)\s*$", "", s).strip()
        if re.fullmatch(r"\(.*\)", s): continue                      # notes like "(artist: ...)", "(x2)"
        if not out and title and norm(s) == norm(title.replace(" ", "")): continue   # sheet starts with the song title
        if s: out.append((sec, s))
    return out

def _sim(a, b):
    if a == b: return 1.0
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def align(tokens, W):
    """Needleman-Wunsch between lyric tokens and whisper words. Returns idx[i] -> whisper index or None."""
    n, m = len(tokens), len(W)
    A = [norm(t) for t in tokens]; B = [norm(w["w"]) if w.get("p", 1) >= 0.08 else "\x00" for w in W]
    GAP = -0.45
    S = np.zeros((n + 1, m + 1), np.float32); P = np.zeros((n + 1, m + 1), np.int8)
    S[1:, 0] = GAP * np.arange(1, n + 1); S[0, 1:] = GAP * np.arange(1, m + 1); P[1:, 0] = 1; P[0, 1:] = 2
    for i in range(1, n + 1):
        ai = A[i - 1]
        sims = np.array([_sim(ai, b) for b in B], np.float32)
        mt = np.where(sims >= 0.6, 1.0 + sims, -0.6)
        for j in range(1, m + 1):
            d = S[i - 1, j - 1] + mt[j - 1]; u = S[i - 1, j] + GAP; l = S[i, j - 1] + GAP
            # on ties prefer skipping the later whisper word: repeated words ("pop, pop, ...") align to the earliest run
            if l >= d and l >= u: S[i, j] = l; P[i, j] = 2
            elif d >= u: S[i, j] = d; P[i, j] = 0
            else: S[i, j] = u; P[i, j] = 1
    idx = [None] * n; i, j = n, m
    while i > 0 and j > 0:
        if P[i, j] == 0:
            if _sim(A[i - 1], B[j - 1]) >= 0.6: idx[i - 1] = j - 1
            i -= 1; j -= 1
        elif P[i, j] == 1: i -= 1
        else: j -= 1
    return idx

def vocal_onsets(work):
    """Sung-syllable onsets (spectral flux peaks on the vocal stem) -> (times, strength), and last active time."""
    import soundfile as sf
    p = os.path.join(work, "vocals16k.flac")
    if not os.path.exists(p): return None
    a, sr = sf.read(p, dtype="float32"); h, n = 160, 1024
    fr = np.lib.stride_tricks.sliding_window_view(a, n)[::h]
    out_t, out_s = [], []
    for i in range(0, len(fr), 4000):
        blk = fr[i:i + 4001] * np.hanning(n)
        S = np.log1p(np.abs(np.fft.rfft(blk))[:, :200] * 10)
        fl = np.maximum(0, np.diff(S, axis=0)).sum(1); rms = np.sqrt((blk[1:] ** 2).mean(1))
        out_t.append((np.arange(len(fl)) + i + 1) * h / sr + n / 2 / sr); out_s.append(np.where(rms > 0.01, fl, 0))
    t = np.concatenate(out_t); f = np.concatenate(out_s)
    if not f.any(): return None
    thr = np.percentile(f[f > 0], 60)
    pk = [i for i in range(1, len(f) - 1) if f[i] > thr and f[i] >= f[i - 1] and f[i] >= f[i + 1]]
    T, Sg = [], []
    for i in pk:
        if T and t[i] - T[-1] < 0.12:
            if f[i] > Sg[-1]: T[-1], Sg[-1] = t[i], f[i]
            continue
        T.append(t[i]); Sg.append(f[i])
    act = vocal_activity(work); last = (np.where(act)[0][-1] * 0.05) if act is not None and act.any() else None
    return np.array(T), np.array(Sg), last

def lines_from_lyrics(sheet, segs, duration, title=None, onsets=None):
    W = clean_words(segs); L = parse_lyrics(sheet, title)
    if not L: die("lyrics file has no lines")
    toks, owner = [], []
    for li, (_, s) in enumerate(L):
        for t in s.split(): toks.append(t); owner.append(li)
    idx = align(toks, W)
    st = [W[k]["start"] if k is not None else None for k in idx]
    pr = [W[k]["p"] if k is not None else 0.0 for k in idx]
    known = [i for i, v in enumerate(st) if v is not None]
    if not known: die("could not match any lyric words to the vocals; check the lyrics belong to this song")
    # interpolate unmatched tokens between matched neighbours, in syllable-ish proportion
    for i in range(len(st)):
        if st[i] is None:
            a = max([k for k in known if k < i], default=None); b = min([k for k in known if k > i], default=None)
            ta = st[a] if a is not None else max(0.0, st[b] - 0.35 * (b - i + 1))
            tb = st[b] if b is not None else min(duration - 0.2, st[a] + 0.35 * (i - a + 1))
            ia = a if a is not None else i - 1; ib = b if b is not None else i + 1
            st[i] = ta + (tb - ta) * (i - ia) / (ib - ia)
    # runs of unheard words: place them on sung-syllable onsets inside the gap when there are enough
    if onsets is not None:
        OT, OS, last = onsets; i = 0
        while i < len(idx):
            if idx[i] is not None: i += 1; continue
            j = i
            while j < len(idx) and idx[j] is None: j += 1
            lo = st[i - 1] + 0.1 if i > 0 else 0.0
            hi = st[j] - 0.1 if j < len(idx) else (last + 0.1 if last else duration)
            sel = np.where((OT > lo) & (OT < hi))[0]
            if len(sel) >= (j - i):
                keep = np.sort(sel[np.argsort(-OS[sel])[: j - i]])
                for k, oi in zip(range(i, j), keep): st[k] = float(OT[oi])
            i = j
    for i in range(1, len(st)):
        if st[i] < st[i - 1] + 0.04: st[i] = st[i - 1] + 0.04
    lines = []
    for li, (sec, s) in enumerate(L):
        ks = [i for i, o in enumerate(owner) if o == li]
        lines.append(dict(section=sec, toks=[toks[i] for i in ks], starts=[st[i] for i in ks],
                          probs=[pr[i] for i in ks], matched=[idx[i] is not None for i in ks]))
    return lines

# ---------------------------------------------------------------- finish + write
def finalize(lines, duration, act=None):
    # whisper often starts a phrase-initial word long before it is sung: pull it next to the following word
    for l in lines:
        if len(l["starts"]) > 1 and l["starts"][1] - l["starts"][0] > 1.5: l["starts"][0] = l["starts"][1] - 0.35
    out = []
    for li, l in enumerate(lines):
        nxt = lines[li + 1]["starts"][0] if li + 1 < len(lines) else duration
        hard_end = nxt - 0.02
        ws = []
        for k, t in enumerate(l["toks"]):
            s = l["starts"][k]
            if k + 1 < len(l["toks"]): e = l["starts"][k + 1]
            else:
                e = s + 1.4
                if act is not None:   # hold the last word while the voice is still sounding (max 3 s)
                    f = int(s / 0.05) + 1
                    while f < len(act) and act[f] and f * 0.05 < s + 3.0: f += 1
                    e = max(s + 0.3, f * 0.05)
            e = min(e, hard_end); e = max(e, s + 0.08)
            ws.append(dict(text=t, start=round(s, 3), end=round(e, 3)))
        reasons = []
        p = [x for x, mt in zip(l["probs"], l["matched"]) if mt]
        nm = sum(1 for x in l["matched"] if not x)
        if nm: reasons.append(f"{nm}/{len(l['toks'])} words not heard (timing interpolated)")
        if p and np.mean(p) < 0.6: reasons.append(f"low ASR confidence ({np.mean(p):.2f})")
        elif p and min(p) < 0.25: reasons.append("a word with very low ASR confidence")
        dur = ws[-1]["end"] - ws[0]["start"]
        if len(ws) / max(dur, 0.1) > 7: reasons.append("very fast for its length (timing may be compressed)")
        out.append(dict(section=l["section"], text=" ".join(l["toks"]), start=ws[0]["start"], end=ws[-1]["end"],
                        unsure="; ".join(reasons) or None, words=ws))
    return out

def write_outputs(work, proj, lines, mode):
    save_json(os.path.join(work, "lyrics.json"), dict(title=proj.get("title"), artist=proj.get("artist"),
                                                      duration=proj["duration"], mode=mode, lines=lines))
    txt = ["# Lyric sheet. Edit the words, then run: lyricvideo.py prep --work DIR --relyric",
           "# [Section] headers are optional and show above the lyric; timestamps and (?) marks are ignored on re-read.", ""]
    prev = None
    for i, l in enumerate(lines):
        gap = i > 0 and l["start"] - lines[i - 1]["end"] > 4
        if l["section"] != prev:
            if gap: txt += ["", f"[Instrumental] {mmss(lines[i-1]['end'])}"]
            if l["section"]: txt += ["", f"[{l['section']}]"]
            prev = l["section"]
        elif gap: txt.append("")
        txt.append(f"{mmss(l['start'])}  {l['text']}{' (?)' if l['unsure'] else ''}")
    open(os.path.join(work, "lyrics.txt"), "w").write("\n".join(txt).strip() + "\n")
    low = [f"{mmss(l['start'])}  {l['text']}  -- {l['unsure']}" for l in lines if l["unsure"]]
    open(os.path.join(work, "low_confidence.txt"), "w").write("\n".join(low) + ("\n" if low else ""))
    return low

def run(args):
    work = os.path.abspath(args.work)
    if args.relyric:
        proj = load_json(os.path.join(work, "project.json")) or die("no project.json; run prep with --song/--image first")
        segs = (load_json(os.path.join(work, "whisper.json")) or die("no whisper.json; run a full prep first"))["segments"]
        src = args.lyrics or os.path.join(work, "lyrics.txt")
        lines = lines_from_lyrics(open(src).read(), segs, proj["duration"], proj.get("title"), vocal_onsets(work)); mode = "lyrics"
    else:
        proj = copy_inputs(work, args.song, args.image)
        song_path = os.path.join(work, proj["song"])
        log(f"song: {proj['song_source_name']}  duration {proj['duration']:.2f}s  title={proj.get('title')!r}  artist={proj.get('artist')!r}")
        vox = separate(work, song_path, skip=args.no_separate)
        wj = load_json(os.path.join(work, "whisper.json"))
        if wj and wj.get("model") == args.model and not args.retranscribe:
            log("whisper.json exists, reusing (use --retranscribe to redo)"); segs = wj["segments"]
            if not os.path.exists(os.path.join(work, "vocals16k.flac")): decode16k(vox, os.path.join(work, "vocals16k.flac"))
        else:
            segs = transcribe(work, vox, args.model, prompt=(f"Song lyrics: {proj['title']}." if proj.get("title") else None))
        if args.lyrics:
            shutil.copyfile(args.lyrics, os.path.join(work, "lyrics_pasted.txt"))
            lines = lines_from_lyrics(open(args.lyrics).read(), segs, proj["duration"], proj.get("title"), vocal_onsets(work)); mode = "lyrics"
        else:
            lines = lines_from_transcript(segs); mode = "transcript"
    if not lines: die("no vocals/words found")
    lines = finalize(lines, proj["duration"], vocal_activity(work))
    low = write_outputs(work, proj, lines, mode)
    log(f"\n{len(lines)} lines, {sum(len(l['words']) for l in lines)} words ({mode} mode)")
    log(open(os.path.join(work, "lyrics.txt")).read())
    log(f"low-confidence lines: {len(low)}" + ("".join("\n  " + x for x in low)))
    log(f"\nwrote {work}/lyrics.json, lyrics.txt, low_confidence.txt")
