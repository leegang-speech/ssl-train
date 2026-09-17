# prepare_hubert_finetune.py

import argparse
import json
import hashlib
from pathlib import Path
from collections import Counter

import soundfile as sf
import torch
import torchaudio

# python prepare_hubert_finetune.py \
#     --train-jsonl train.jsonl \
#     --valid-jsonl valid.jsonl \
#     --output-dir /data/fairseq/hubert_ft

TARGET_SR = 16000


def text_to_ltr(text):
    # 연속 whitespace 정리
    text = " ".join(text.strip().split())

    # fairseq letter format
    # "뭐 학원" -> "뭐 | 학 원"
    tokens = []

    for ch in text:
        if ch == " ":
            tokens.append("|")
        else:
            tokens.append(ch)

    return " ".join(tokens)


def convert_split(jsonl_path, split, output_dir):
    output_dir = Path(output_dir)

    wav_root = output_dir / "wavs"
    wav_root.mkdir(parents=True, exist_ok=True)

    tsv_path = output_dir / f"{split}.tsv"
    ltr_path = output_dir / f"{split}.ltr"

    token_counter = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(tsv_path, "w", encoding="utf-8") as ftsv, \
         open(ltr_path, "w", encoding="utf-8") as fltr:

        # TSV 첫 줄 = audio root
        ftsv.write(str(wav_root.resolve()) + "\n")

        for i, line in enumerate(fin):
            item = json.loads(line)

            src_path = Path(item["audio_filepath"])
            text = item["text"]

            wav, sr = torchaudio.load(str(src_path))

            # stereo인 경우 mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # 8k -> 16k
            if sr != TARGET_SR:
                wav = torchaudio.functional.resample(
                    wav,
                    sr,
                    TARGET_SR,
                )

            # 파일명 충돌 방지
            key = hashlib.md5(
                str(src_path).encode()
            ).hexdigest()[:10]

            out_name = f"{src_path.stem}_{key}.wav"
            out_path = wav_root / out_name

            torchaudio.save(
                str(out_path),
                wav,
                TARGET_SR,
            )

            num_samples = wav.shape[1]

            # train.tsv
            ftsv.write(
                f"{out_name}\t{num_samples}\n"
            )

            # train.ltr
            ltr = text_to_ltr(text)
            fltr.write(ltr + "\n")

            for token in ltr.split():
                token_counter[token] += 1

            if i % 1000 == 0:
                print(i, src_path)

    return token_counter


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--valid-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    train_counter = convert_split(
        args.train_jsonl,
        "train",
        args.output_dir,
    )

    convert_split(
        args.valid_jsonl,
        "valid",
        args.output_dir,
    )

    # dictionary는 train 기준
    dict_path = Path(args.output_dir) / "dict.ltr.txt"

    with open(dict_path, "w", encoding="utf-8") as f:
        for token, count in train_counter.most_common():
            f.write(f"{token} {count}\n")

    print("done:", args.output_dir)


if __name__ == "__main__":
    main()
