"""WER of Coqui's own XTTS-v2 (PyTorch, sampling with the config defaults)
on the sentences of wer.py, to tell model artifacts from port bugs.

    python scripts/ref_wer.py <checkpoint-dir> --out /tmp/wer-ref [--langs fr,en] [--seeds 2]
"""

import argparse
import io
import sys
from pathlib import Path

import torch
import wave

import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("--out", required=True)
p.add_argument("--langs", default="fr,en")
p.add_argument("--seeds", type=int, default=2)
p.add_argument("--voice", default="Damien Black")
p.add_argument("--asr", default="http://127.0.0.1:6767")
p.add_argument("--asr-model", default="parakeet-tdt-v3-cpu")
a = p.parse_args()

# Reuse the sentences, WER and ASR client of wer.py without running it.
src = Path(__file__).with_name("wer.py").read_text().split("\np = argparse.ArgumentParser()")[0]
w = type(sys)("w")
exec(src, w.__dict__)

torch.set_num_threads(16)
config = XttsConfig()
config.load_json(f"{a.ckpt}/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=a.ckpt, eval=True)
cond, spk = model.speaker_manager.speakers[a.voice].values()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for lang in a.langs.split(","):
    errs = words = 0
    for i, s in enumerate(w.SENTENCES[lang]):
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            wav = model.inference(s, lang, cond, spk)["wav"]
            buf = io.BytesIO()
            with wave.open(buf, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())
            data = buf.getvalue()
            (out / f"{lang}{i:02d}-s{seed}.wav").write_bytes(data)
            hyp = w.asr(a.asr, a.asr_model, data, lang)
            e, n = w.wer(s, hyp)
            errs += e
            words += n
            if e:
                print(f"  {lang}{i:02d} s{seed} ({e}/{n}): {hyp}", flush=True)
    print(f"{lang}: WER {100 * errs / words:.1f}% ({errs}/{words} words)", flush=True)
