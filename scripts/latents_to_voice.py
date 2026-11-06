"""Writes a .voice file (see crates/xtts/src/voicefile.rs) from a Coqui latents.pt.

Usage: latents_to_voice.py latents.pt out.voice voice_name
"""
import struct
import sys

import numpy as np
import torch


def write_voice(path, name, cond, spk):
    cond = np.asarray(cond, dtype="<f4").reshape(32, 1024)
    spk = np.asarray(spk, dtype="<f4").reshape(512)
    n = name.encode()
    with open(path, "wb") as f:
        f.write(b"XTTSVOIC" + struct.pack("<II", 1, len(n)) + n + struct.pack("<III", 32, 1024, 512))
        f.write(cond.tobytes())
        f.write(spk.tobytes())


if __name__ == "__main__":
    src, out, voice = sys.argv[1:]
    d = torch.load(src, weights_only=False, map_location="cpu")
    write_voice(out, voice, d["gpt_cond_latent"].numpy(), d["speaker_embedding"].numpy())
