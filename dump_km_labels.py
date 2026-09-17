import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torchaudio


# python dump_km_labels.py \
#     --manifest ./manifests/train.jsonl \
#     --kmeans ./kmeans/km500.joblib \
#     --output ./labels/train.km

# python dump_km_labels.py \
#    --manifest ./manifests/valid.jsonl \
#     --kmeans ./kmeans/km500.joblib \
#     --output ./labels/valid.km

TARGET_SR = 16000


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
        help="train.jsonl or valid.jsonl",
    )

    parser.add_argument(
        "--kmeans",
        required=True,
        help="trained kmeans joblib file",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="output .km file",
    )

    return parser.parse_args()


def load_audio(item):
    path = item["path"]
    channel = item["channel"]

    wav, sr = torchaudio.load(path)

    # wav: [C, T]
    if wav.size(0) == 1:
        wav = wav[0]

    else:
        wav = wav[channel]

    # 8k -> 16k
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(
            wav,
            orig_freq=sr,
            new_freq=TARGET_SR,
        )

    return wav


mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=TARGET_SR,
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
    # [13, frames]
    mfcc = mfcc_transform(wav)

    delta = torchaudio.functional.compute_deltas(
        mfcc
    )

    delta2 = torchaudio.functional.compute_deltas(
        delta
    )

    # [39, frames]
    feat = torch.cat(
        [mfcc, delta, delta2],
        dim=0,
    )

    # [frames, 39]
    feat = feat.transpose(0, 1)

    return feat.cpu().numpy().astype(
        np.float32
    )


def main():
    args = parse_args()

    kmeans = joblib.load(
        args.kmeans
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_utts = 0
    total_frames = 0

    with open(
        args.manifest,
        "r",
        encoding="utf-8",
    ) as manifest_fp, open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_fp:

        for i, line in enumerate(manifest_fp):
            item = json.loads(line)

            #
            # waveform
            #
            wav = load_audio(item)

            #
            # MFCC
            #
            feat = extract_mfcc(wav)

            #
            # K-means cluster prediction
            #
            labels = kmeans.predict(
                feat
            )

            #
            # utterance 하나 = .km 한 줄
            #
            output_fp.write(
                " ".join(
                    map(str, labels.tolist())
                )
                + "\n"
            )

            num_utts += 1
            total_frames += len(labels)

            if i % 1000 == 0:
                print(
                    f"{i:,} | "
                    f"{item.get('sample_id')} | "
                    f"channel={item['channel']} | "
                    f"mfcc={feat.shape} | "
                    f"labels={len(labels)}"
                )

    print()
    print("Done")
    print(
        f"utterances : {num_utts:,}"
    )
    print(
        f"frames     : {total_frames:,}"
    )
    print(
        f"output     : {output_path}"
    )


if __name__ == "__main__":
    main()
