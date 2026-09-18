# decode_ctc.py

import argparse
import torch
import torchaudio

from fairseq import checkpoint_utils
from fairseq.data.data_utils import post_process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--wav", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.checkpoint],
        arg_overrides={
            "data": args.data,
            "label_dir": args.data,
        },
    )

    model = models[0].to(device)
    model.eval()

    wav, sr = torchaudio.load(args.wav)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != 16000:
        wav = torchaudio.functional.resample(
            wav,
            sr,
            16000,
        )

    source = wav.squeeze(0).unsqueeze(0).to(device)
    padding_mask = torch.zeros(
        source.shape,
        dtype=torch.bool,
        device=device,
    )

    with torch.no_grad():
        net_output = model(
            source=source,
            padding_mask=padding_mask,
        )

        # [T, B, vocab]
        lprobs = model.get_normalized_probs(
            net_output,
            log_probs=True,
        )

    # [T]
    tokens = lprobs[:, 0].argmax(dim=-1)

    # CTC repeat 제거
    tokens = tokens.unique_consecutive()

    dictionary = task.target_dictionary

    # fairseq CTC blank 기본값은 0
    blank_idx = (
        dictionary.index(task.blank_symbol)
        if hasattr(task, "blank_symbol")
        else 0
    )

    tokens = tokens[tokens != blank_idx]

    units = dictionary.string(tokens)

    # "안 녕 | 하 세 요" -> "안녕 하세요"
    text = post_process(units, "letter")

    print("units:", units)
    print("text :", text)


if __name__ == "__main__":
    main()
