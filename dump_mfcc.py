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

    # [C, T] -> [T]
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


mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=13,
    melkwargs={
        "n_fft": 400,
        "win_length": 400,
        "hop_length": 160,
        "n_mels": 40,
        "center": False,
    },
)


def extract_mfcc(wav):
    # [13, T]
    mfcc = mfcc_transform(wav)

    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)

    # [39, T]
    feat = torch.cat(
        [mfcc, delta, delta2],
        dim=0,
    )

    # [T, 39]
    feat = feat.transpose(0, 1)

    return feat


def dump_mfcc(
    manifest_path,
    output_dir,
    split="train",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # fairseq simple_kmeans naming convention
    feat_path = output_dir / f"{split}_0_1.npy"
    len_path = output_dir / f"{split}_0_1.len"

    all_features = []
    lengths = []

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as f:

        for i, line in enumerate(f):
            item = json.loads(line)

            wav = load_audio(item)

            feat = extract_mfcc(wav)

            feat = feat.cpu().numpy().astype(
                np.float32
            )

            # 이 utterance의 MFCC frame 개수
            lengths.append(feat.shape[0])

            # 전체 feature에 이어 붙임
            all_features.append(feat)

            if i % 1000 == 0:
                print(
                    f"{i:,} | "
                    f"{item['sample_id']} | "
                    f"wav={wav.shape[0]} | "
                    f"mfcc={feat.shape}"
                )

    # [전체 MFCC frames, 39]
    all_features = np.concatenate(
        all_features,
        axis=0,
    )

    np.save(
        feat_path,
        all_features,
    )

    # utterance별 frame length
    with open(
        len_path,
        "w",
        encoding="utf-8",
    ) as f:

        for length in lengths:
            f.write(f"{length}\n")

    print()
    print(f"MFCC: {feat_path}")
    print(f"LEN : {len_path}")
    print(f"shape: {all_features.shape}")


if __name__ == "__main__":

    dump_mfcc(
        manifest_path="./manifests/train.jsonl",
        output_dir="./mfcc",
        split="train",
    )
