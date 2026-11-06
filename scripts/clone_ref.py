"""Reference for the cloning parity example: the intermediate values of
Coqui's get_conditioning_latents for one voice, on the CPU.

Usage: clone_ref.py <checkpoint-dir> <out.safetensors> <device> a.wav b.wav ...

Dumps: wav22k (all clips joined), wav16k0 (first clip, resampled again),
mel0 (first chunk), cond0 (conditioning encoder), perc0 (perceiver),
spkfeat0 (speaker features of the first clip), spk0 (its embedding),
cond [1, 32, 1024], spk [1, 512, 1], fb_gpt (mel filterbank).
"""
import sys

import soundfile as sf
import torch
import torchaudio
from safetensors.torch import save_file


def _sf_load(path, *args, **kwargs):  # torchaudio.load needs a backend otherwise
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(y.T.copy()), sr


torchaudio.load = _sf_load
from TTS.tts.configs.xtts_config import XttsConfig  # noqa: E402
from TTS.tts.models import xtts as X  # noqa: E402

ckpt, out, device, *wavs = sys.argv[1:]
torch.set_num_threads(16)
config = XttsConfig()
config.load_json(f"{ckpt}/config.json")
model = X.Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=ckpt, eval=True)
model = model.to(device)

d = {}
with torch.inference_mode():
    clips = []
    for p in wavs:
        a = X.load_audio(p, 22050)[:, : 22050 * 30]
        clips.append(a)
    full = torch.cat(clips, -1)
    d["wav22k"] = full[0]
    a16 = torchaudio.functional.resample(clips[0], 22050, 16000)
    d["wav16k0"] = a16[0]
    chunk = full[:, : 22050 * 4]
    mel = X.wav_to_mel_cloning(chunk, mel_norms=model.mel_stats.cpu(), n_fft=2048, hop_length=256, win_length=1024,
                               power=2, normalized=False, sample_rate=22050, f_min=0, f_max=8000, n_mels=80)
    d["mel0"] = mel[0]
    g = model.gpt
    ce = g.conditioning_encoder(mel.to(device))
    d["cond0"] = ce[0]
    d["perc0"] = g.conditioning_perceiver(ce.permute(0, 2, 1))[0]
    se = model.hifigan_decoder.speaker_encoder
    x = se.torch_spec(a16.to(device))
    x = (x + 1e-6).log()
    x = se.instancenorm(x)
    d["spkfeat0"] = x[0]
    d["spk0"] = se.forward(a16.to(device).clone(), l2_norm=True)[0]
    d["fb_gpt"] = torchaudio.functional.melscale_fbanks(1025, 0.0, 8000.0, 80, 22050, "slaney")
    cond, spk = model.get_conditioning_latents(audio_path=list(wavs), gpt_cond_len=30, gpt_cond_chunk_len=4,
                                               max_ref_length=30)
    d["cond"] = cond
    d["spk"] = spk
save_file({k: v.float().cpu().contiguous() for k, v in d.items()}, out)
print({k: tuple(v.shape) for k, v in d.items()})
