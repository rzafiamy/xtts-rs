"""Intelligibility check: synthesize sentences with a running `xtts serve`,
transcribe them with an ASR model behind an OpenAI-compatible endpoint
(zallama's parakeet-tdt-v3-cpu by default), print WER per language.

    python scripts/wer.py --tts http://127.0.0.1:18091 --out /tmp/wer-q8 [--seeds 2]

Sentences avoid digits: Parakeet writes numbers back as digits (inverse text
normalization), which would count as errors.
"""

import argparse
import json
import re
import unicodedata
import urllib.request
import uuid
from pathlib import Path

SENTENCES = {
    "fr": [
        "Bonjour à tous, et bienvenue dans cette démonstration de synthèse vocale.",
        "Le chat dort paisiblement sur le canapé du salon.",
        "Nous partirons demain matin vers la montagne, si le temps le permet.",
        "La réunion de ce soir a été reportée à la semaine prochaine.",
        "Elle a ouvert la fenêtre pour laisser entrer un peu d'air frais.",
        "Les enfants jouent dans le jardin pendant que leurs parents préparent le repas.",
        "Il faut absolument terminer ce rapport avant la fin de la journée.",
        "Le musée présente une exposition consacrée à un célèbre peintre impressionniste.",
        "Pouvez-vous me dire où se trouve la gare la plus proche ?",
        "Après la pluie, un magnifique arc-en-ciel est apparu au-dessus de la ville.",
    ],
    "en": [
        "Hello everyone, and welcome to this speech synthesis demonstration.",
        "The cat is sleeping peacefully on the living room couch.",
        "We will leave for the mountains tomorrow morning if the weather allows.",
        "Tonight's meeting has been moved to next week.",
        "She opened the window to let in some fresh air.",
        "The children are playing in the garden while their parents cook dinner.",
        "We really need to finish this report before the end of the day.",
        "The museum is showing an exhibition about impressionist painters.",
        "Could you tell me where the nearest train station is?",
        "After the rain, a beautiful rainbow appeared above the city.",
    ],
}


def norm(t):
    t = unicodedata.normalize("NFC", t.lower())
    t = re.sub(r"[’']", " ", t)
    t = re.sub(r"[^\w\s-]", " ", t).replace("-", " ")
    return t.split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def tts(base, text, lang, voice, seed, extra):
    body = json.dumps({"input": text, "language": lang, "voice": voice, "seed": seed, **extra}).encode()
    req = urllib.request.Request(f"{base}/v1/audio/speech", body, {"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=300).read()


def asr(base, model, wav, lang):
    b = uuid.uuid4().hex
    parts = [
        f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="language"\r\n\r\n{lang}\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode() + wav + b"\r\n",
        f"--{b}--\r\n".encode(),
    ]
    req = urllib.request.Request(
        f"{base}/v1/audio/transcriptions", b"".join(parts), {"content-type": f"multipart/form-data; boundary={b}"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["text"]


p = argparse.ArgumentParser()
p.add_argument("--tts", default="http://127.0.0.1:18091")
p.add_argument("--asr", default="http://127.0.0.1:6767")
p.add_argument("--asr-model", default="parakeet-tdt-v3-cpu")
p.add_argument("--out", required=True)
p.add_argument("--seeds", type=int, default=2)
p.add_argument("--voice", default="Damien Black,Claribel Dervla", help="comma-separated voices")
p.add_argument("--strip-period", action="store_true", help="send sentences without their final period")
p.add_argument("--langs", default="fr,en")
p.add_argument("--extra", default="{}", help='JSON merged into each request, e.g. {"temperature": 0.5}')
a = p.parse_args()
extra = json.loads(a.extra)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for lang in a.langs.split(","):
    sents = SENTENCES[lang]
    errs = words = 0
    for i, s in enumerate(sents):
        text = s.rstrip(".") if a.strip_period else s
        for seed in range(a.seeds):
            for vi, voice in enumerate(a.voice.split(",")):
                wav = tts(a.tts, text, lang, voice, seed, extra)
                (out / f"{lang}{i:02d}-v{vi}-s{seed}.wav").write_bytes(wav)
                hyp = asr(a.asr, a.asr_model, wav, lang)
                e, n = wer(s, hyp)
                errs += e
                words += n
                if e:
                    print(f"  {lang}{i:02d} v{vi} s{seed} ({e}/{n}): {hyp}")
    print(f"{lang}: WER {100 * errs / words:.1f}% ({errs}/{words} words)")
