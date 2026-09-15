import json
from pathlib import Path

import numpy as np
import torch
import torchaudio


TARGET_SR = 16000


def load_audio(item):
    path = item["path"]
    channel = item["channel"]

    wav, sr = torchaudio.load(path)

    # [C, T]
    if wav.size(0) == 1:
        wav = wav[0]
    else:
        wav = wav[channel]

    # 8k -> 16k on-the-fly
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(
            wav,
            orig_freq=sr,
            new_freq=TARGET_SR,
        )

    return wav


def extract_mfcc(wav):
    # fairseq HuBERT 1st iteration과 맞춰서
    # 13 MFCC + delta + delta-delta = 39 dim

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=13,
        melkwargs={
            "n_fft": 400,        # 25 ms @ 16k
            "win_length": 400,
            "hop_length": 160,   # 10 ms -> 100 Hz
            "n_mels": 40,
            "center": False,
        },
    )

    # [13, frames]
    mfcc = mfcc_transform(wav)

    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)

    # [39, frames]
    feat = torch.cat(
        [mfcc, delta, delta2],
        dim=0,
    )

    # [frames, 39]
    feat = feat.transpose(0, 1)

    return feat


manifest = "manifests/train.jsonl"

with open(manifest, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        wav = load_audio(item)

        feat = extract_mfcc(wav)

        print(
            item["sample_id"],
            wav.shape,
            feat.shape,
        )
