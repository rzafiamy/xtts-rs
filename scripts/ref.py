"""Reference dump from Coqui's XTTS-v2 (coqui-tts) for the parity example.

Setup (CPU torch is enough):
    python3.11 -m venv /bank2/temporary/xtts-ref
    /bank2/temporary/xtts-ref/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    /bank2/temporary/xtts-ref/bin/pip install "coqui-tts[languages,codec]" "transformers>=4.57,<4.58" safetensors

Usage:
    ref.py <checkpoint-dir> <out.safetensors> --text "..." --lang fr --voice "Claribel Dervla"
           [--clone ref.wav]

Greedy decoding (do_sample=False) with the config's repetition penalty, so
the Rust port can be compared code for code. Dumps: text_ids, codes,
latents [1, N, 1024], wav [S] (24 kHz), cond [1, 32, 1024], spk [512].
"""

import argparse

import torch
from safetensors.torch import save_file
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("out")
p.add_argument("--text", required=True)
p.add_argument("--lang", default="en")
p.add_argument("--voice", default="Claribel Dervla")
p.add_argument("--clone", help="reference wav: clone it instead of a built-in voice")
p.add_argument("--rep", type=float, default=None, help="repetition penalty (default: config)")
a = p.parse_args()

torch.set_num_threads(16)
config = XttsConfig()
config.load_json(f"{a.ckpt}/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=a.ckpt, eval=True)

if a.clone:
    cond, spk = model.get_conditioning_latents(
        audio_path=a.clone,
        gpt_cond_len=config.gpt_cond_len,
        gpt_cond_chunk_len=config.gpt_cond_chunk_len,
        max_ref_length=config.max_ref_len,
    )
else:
    cond, spk = model.speaker_manager.speakers[a.voice].values()

rep = config.repetition_penalty if a.rep is None else a.rep
lang = a.lang.split("-")[0]
sent = a.text.strip().lower()
ids = model.tokenizer.encode(sent, lang=lang)
text_tokens = torch.IntTensor(ids).unsqueeze(0)
with torch.no_grad():
    codes = model.gpt.generate(
        cond_latents=cond,
        text_inputs=text_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=rep,
        length_penalty=1.0,
        output_attentions=False,
    )
    expected = torch.tensor([codes.shape[-1] * model.gpt.code_stride_len])
    latents = model.gpt(
        text_tokens,
        torch.tensor([text_tokens.shape[-1]]),
        codes,
        expected,
        cond_latents=cond,
        return_attentions=False,
        return_latent=True,
    )
    wav = model.hifigan_decoder(latents, g=spk).squeeze()

print("text ids", ids)
print("codes", codes.shape[-1], codes[0, :16].tolist(), "...", codes[0, -4:].tolist())
print("latents", tuple(latents.shape), "wav", wav.shape[0] / 24000, "s")
save_file(
    {
        "text_ids": torch.tensor(ids, dtype=torch.int64),
        "codes": codes[0].to(torch.int64),
        "latents": latents.float().contiguous(),
        "wav": wav.float().contiguous(),
        "cond": cond.float().contiguous(),
        "spk": spk.reshape(-1).float().contiguous(),
    },
    a.out,
)
