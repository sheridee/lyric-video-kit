# Lyric Video Kit

Turn **any song + one image** into a word-timed lyric video: the image as a slowly zooming, blurred
full-bleed background, a beat-reactive glow, big centred lyrics highlighted word by word in a neon accent,
the next line faint below, mirrored spectrum bars, line fades, and a small title/artist tag.
CPU only. Outputs H.264 + AAC MP4 (faststart, sized to stay under ~50 MB), a clean still, and a QA report.

## Install
```bash
git clone https://github.com/sheridee/lyric-video-kit && cd lyric-video-kit
bash install.sh                    # idempotent; add --prefetch-models to download the models now (~1.7 GB)
python3 lyricvideo.py doctor       # checks ffmpeg, python packages, font, headless Chrome
```
install.sh adds what is missing: ffmpeg, CPU torch, demucs, faster-whisper, numpy, pillow, soundfile, and a
standalone `chrome-headless-shell` from Chrome for Testing in `~/.cache/lyric-video-kit/` (system Chrome can hang
on headless screenshots). Set `LVK_CHROME=/path/to/chrome-headless-shell` to use another build.

## Make a video
```bash
K=lyricvideo.py; W=/path/to/work/mysong

# 1. prep: separate vocals (demucs htdemucs), transcribe with word timing (faster-whisper large-v3-turbo)
python3 $K prep --song song.wav --image cover.png --work $W
#    ...or, better, give the real lyrics and the kit aligns those words to the sung timing:
python3 $K prep --song song.wav --image cover.png --work $W --lyrics lyrics.txt

# 2. check $W/lyrics.txt and $W/low_confidence.txt; fix any line in lyrics.txt, then re-align (no re-transcribe, ~1 s):
python3 $K prep --work $W --relyric

# 3. one frame to check the look (default: a mid-chorus moment)
python3 $K preview --work $W [--t 1:15] [--aspect 16:9|9:16|1:1]

# 4. render
python3 $K render --work $W [--aspect 16:9|9:16|1:1] [--title "Song"] [--artist "Artist"] \
    [--accent '#ff3ea5'] [--focus 0.5,0.6,0.07] [--start 0:30 --end 0:45] [--workers 4]
```
Outputs in `$W/out/`: `lyric_video_<aspect>[_<start>-<end>s].mp4`, `..._still.png`, `..._qa.json`.

## Lyrics sheet format (`--lyrics` and the generated `lyrics.txt`)
```
[Verse 1]
City lights are calling out my name
00:17  Every step I take, I'm not the same (?)

[Chorus]
...
```
`[Section]` headers are optional and show as a small label above the lyric (`[Instrumental]` hides it).
Leading `mm:ss` stamps, trailing `(?)`, lines that are only `(...)`, and a first line equal to the song title are
ignored. Write repeats out in full (no "x2"). One sheet line = one on-screen line.

## Options
| option | meaning |
|---|---|
| `--aspect` | `16:9` 1920x1080 (default), `9:16` 1080x1920 (lyrics may wrap to 2 rows), `1:1` 1080x1080 |
| `--title/--artist` | default from the file's tags (title falls back to the file name; artist line falls back to "Lyric video") |
| `--accent` | lead colour. Default: picked from the image (strongest pink/magenta/purple family if present, else the strongest vivid hue), plus a contrasting colour and a warm middle colour for bars and labels |
| `--focus x,y[,r]` | point of interest in the image (0-1 of width/height, r = radius / width). The slow zoom drifts toward it, 9:16 / 1:1 crops centre on it, and the beat glow becomes a ring around it. Without it the glow is a soft bloom behind the lyrics |
| `--start/--end` | render only part of the song (seconds or m:ss) |
| `--workers` | parallel compositors (default auto: up to 4, about 0.6 GB RAM each) |
| `prep --model` | `large-v3-turbo` (default), `large-v3`, `medium.en`, `small.en` |
| `prep --no-separate` | skip demucs (faster, worse timing on dense mixes) |
| `LVK_LANG` env | whisper language (default `en`, `auto` to detect) |

## How it works
- **prep**: demucs `htdemucs --two-stems=vocals` (segment 7, 1 job, ~1.5 GB RAM) -> faster-whisper int8 word timestamps
  on the vocal stem -> lines. With `--lyrics`, a Needleman-Wunsch alignment maps sheet words to heard words; unheard
  words are placed on sung-syllable onsets detected in the vocal stem (or interpolated). Lines are flagged in
  `low_confidence.txt` when words were not heard, ASR confidence is low, or timing is compressed.
- **render**: each lyric state (line x current word) is drawn once as a transparent PNG by headless Chrome (Inter,
  CSS glow, auto-shrink to fit); a pool of workers composites frames with numpy/PIL (Ken Burns crop of a
  pre-blurred image, glow, spectrum from an STFT of the song, text layer with fades and a 90 ms glow cross-fade) and
  pipes them to x264; segments are concatenated and muxed with the original audio as AAC 192k.
- **qa** (`qa.json`, also `lyricvideo.py qa file.mp4`): codecs, durations vs expected, faststart, full decode,
  black/frozen stretches, sample/true peak and integrated loudness of output and source.

Speed on 8 CPU cores: prep ~1-1.5 min for a 2.5 min song; render ~1.5 min per minute of 1080p video with 4 workers.

## Limits
- Word timing is only as good as the ASR on the vocal stem. Shouted ad-libs, stacked harmonies and repeated
  one-word hooks ("la, la, la") are the weak spots; check `low_confidence.txt` and use `--lyrics`.
- Transcript-only mode (no `--lyrics`) gives a draft lyric sheet with mishearings; edit it and run `--relyric`.
- English by default. Latin-script lyrics only (other scripts are filtered as hallucinations).
- `--focus` is manual; there is no automatic subject detection.

Font: Inter (SIL Open Font License 1.1, `fonts/OFL.txt`), downloaded by `install.sh`. Code: MIT (`LICENSE`).
