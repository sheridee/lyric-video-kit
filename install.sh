#!/usr/bin/env bash
# Lyric Video Kit installer. Idempotent: safe to re-run; only installs what is missing.
# CPU only. Options: --prefetch-models (download whisper large-v3-turbo + htdemucs now, ~1.7 GB)
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="${LVK_CACHE:-$HOME/.cache/lyric-video-kit}"
PREFETCH=0; for a in "$@"; do [[ "$a" == "--prefetch-models" ]] && PREFETCH=1; done
say(){ printf '\n== %s\n' "$*"; }
SUDO=""; if [[ $EUID -ne 0 ]] && command -v sudo >/dev/null && sudo -n true 2>/dev/null; then SUDO="sudo -n"; fi
apt_install(){ # apt_install pkg...
  if command -v apt-get >/dev/null && { [[ $EUID -eq 0 ]] || [[ -n "$SUDO" ]]; }; then
    $SUDO apt-get update -qq >/dev/null 2>&1 || true
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "$@" >/dev/null
  else echo "  cannot apt-get install $* (no root/sudo); install them manually"; return 1; fi
}

say "system packages"
need=(); command -v ffmpeg >/dev/null || need+=(ffmpeg); command -v unzip >/dev/null || need+=(unzip)
command -v python3 >/dev/null || need+=(python3 python3-pip); python3 -m pip --version >/dev/null 2>&1 || need+=(python3-pip)
command -v git >/dev/null || need+=(git); [[ -e /usr/lib/x86_64-linux-gnu/libsndfile.so.1 || -e /usr/lib/libsndfile.so.1 ]] || need+=(libsndfile1)
if ((${#need[@]})); then echo "  installing: ${need[*]}"; apt_install "${need[@]}" || true; else echo "  ok (ffmpeg $(ffmpeg -version | head -1 | awk '{print $3}'))"; fi

say "python packages"
PIP=(python3 -m pip install -q)
python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages && \
  python3 - <<'PY' && PIP+=(--break-system-packages) || true
import sys, sysconfig, os
sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["stdlib"], "EXTERNALLY-MANAGED")) else 1)
PY
[[ $EUID -ne 0 && -z "${VIRTUAL_ENV:-}" ]] && PIP+=(--user)
have(){ python3 -c "import $1" >/dev/null 2>&1; }
if ! have torch; then echo "  torch (CPU build)"; "${PIP[@]}" torch torchaudio --index-url https://download.pytorch.org/whl/cpu; fi
miss=(); for m in numpy:numpy PIL:pillow soundfile:soundfile faster_whisper:faster-whisper; do have "${m%%:*}" || miss+=("${m#*:}"); done
((${#miss[@]})) && { echo "  ${miss[*]}"; "${PIP[@]}" "${miss[@]}"; }
if ! have demucs.separate; then
  echo "  demucs"; "${PIP[@]}" demucs || true
  have demucs.separate || { "${PIP[@]}" --no-deps demucs openunmix dora-search julius lameenc && \
    "${PIP[@]}" einops pyyaml tqdm omegaconf retrying submitit treetable cloudpickle safetensors sphn; }
fi
echo "  ok"

say "headless Chrome (chrome-headless-shell from Chrome for Testing)"
CHS=""
for p in "${LVK_CHROME:-}" "$CACHE"/chrome-headless-shell/*/chrome-headless-shell-linux64/chrome-headless-shell \
         "$HOME"/.cache/chs/chrome-headless-shell/*/chrome-headless-shell-linux64/chrome-headless-shell; do
  [[ -n "$p" && -x "$p" ]] && CHS="$p" && break
done
if [[ -z "$CHS" ]]; then
  JSON=https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json
  read -r VER URL < <(curl -fsSL "$JSON" | python3 -c "import json,sys;s=json.load(sys.stdin)['channels']['Stable'];print(s['version'],[d['url'] for d in s['downloads']['chrome-headless-shell'] if d['platform']=='linux64'][0])")
  echo "  downloading chrome-headless-shell $VER"
  mkdir -p "$CACHE/chrome-headless-shell/$VER"; TMP="$(mktemp -d)"
  curl -fsSL "$URL" -o "$TMP/chs.zip"
  python3 -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$TMP/chs.zip" "$CACHE/chrome-headless-shell/$VER"
  rm -rf "$TMP"; CHS="$CACHE/chrome-headless-shell/$VER/chrome-headless-shell-linux64/chrome-headless-shell"; chmod +x "$CHS"
fi
missing_libs=$(ldd "$CHS" 2>/dev/null | awk '/not found/{print $1}' || true)
if [[ -n "$missing_libs" ]]; then
  echo "  chrome needs shared libraries: $(echo $missing_libs)"
  apt_install libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 libpango-1.0-0 libcairo2 fonts-liberation || \
  apt_install libasound2t64 || true
fi
echo "  $CHS"

say "font (Inter, SIL OFL)"
if [ ! -s "$KIT/fonts/Inter-Variable.ttf" ]; then
  mkdir -p "$KIT/fonts"
  curl -fsSL -o "$KIT/fonts/Inter-Variable.ttf" "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz,wght%5D.ttf"
fi
echo "  $KIT/fonts/Inter-Variable.ttf"

if ((PREFETCH)); then
  say "prefetching models (whisper large-v3-turbo, htdemucs)"
  python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')"
  python3 -c "from demucs.pretrained import get_model; get_model('htdemucs')"
fi

say "doctor"
LVK_CHROME="$CHS" python3 "$KIT/lyricvideo.py" doctor
