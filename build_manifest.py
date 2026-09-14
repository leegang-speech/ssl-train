#!/usr/bin/env python3

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio-dirs",
        nargs="+",
        required=True,
        help="WAV 파일이 들어있는 디렉토리들",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--valid-percent",
        type=float,
        default=1.0,
        help="validation 비율. 기본 1%%",
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="이보다 짧은 파일은 제외",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=0.0,
        help="0이면 제한 없음. 실제 crop은 학습 Dataset에서 수행",
    )

    parser.add_argument(
        "--stereo-map",
        choices=["rx-tx", "tx-rx"],
        default="rx-tx",
        help=(
            "stereo channel mapping. "
            "rx-tx: ch0=rx,ch1=tx / "
            "tx-rx: ch0=tx,ch1=rx"
        ),
    )

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".wav"],
    )

    return parser.parse_args()


def stable_split(path, valid_percent):
    """
    원본 파일 path를 기반으로 deterministic split.

    stereo RX/TX가 반드시 같은 split으로 감.
    """

    key = str(path).encode("utf-8")

    digest = hashlib.md5(key).hexdigest()

    value = int(digest[:8], 16) % 10000

    threshold = int(valid_percent * 100)

    if value < threshold:
        return "valid"

    return "train"


def write_jsonl(fp, item):
    fp.write(
        json.dumps(
            item,
            ensure_ascii=False,
        )
        + "\n"
    )


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    extensions = {
        ext.lower()
        if ext.startswith(".")
        else "." + ext.lower()
        for ext in args.extensions
    }

    #
    # RX / TX mapping
    #
    if args.stereo_map == "rx-tx":
        stereo_map = {
            0: "rx",
            1: "tx",
        }

    else:
        stereo_map = {
            0: "tx",
            1: "rx",
        }

    #
    # 입력 파일 수집
    #
    files = []

    source_map = {}

    for audio_dir_str in args.audio_dirs:

        root = Path(audio_dir_str).resolve()

        if not root.exists():
            raise FileNotFoundError(
                f"audio directory does not exist: {root}"
            )

        source_name = root.name

        print(f"Scanning: {root}")

        for path in root.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in extensions:
                continue

            path = path.resolve()

            files.append(path)

            source_map[str(path)] = source_name

    #
    # 중복 경로 제거
    #
    files = sorted(set(files))

    print()
    print(f"Found WAV files: {len(files):,}")
    print()

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    stats = Counter()

    hours = defaultdict(float)

    with (
        train_path.open("w", encoding="utf-8") as train_fp,
        valid_path.open("w", encoding="utf-8") as valid_fp,
    ):

        for index, path in enumerate(files):

            if index % 10000 == 0:
                print(
                    f"[{index:,}/{len(files):,}] "
                    f"{path}"
                )

            stats["total_files"] += 1

            #
            # header만 읽는다.
            # waveform 전체를 읽지 않음.
            #
            try:
                info = sf.info(str(path))

            except Exception as e:

                stats["broken_files"] += 1

                print(
                    f"[BROKEN] {path}: {e}"
                )

                continue

            sr = info.samplerate
            channels = info.channels
            frames = info.frames

            if frames <= 0:
                stats["empty_files"] += 1
                continue

            duration = frames / sr

            #
            # 8 kHz 검사
            #
            if sr != args.sample_rate:

                stats["wrong_sample_rate"] += 1
                hours["wrong_sample_rate"] += duration / 3600

                continue

            #
            # 너무 짧은 파일 제외
            #
            if duration < args.min_duration:

                stats["too_short"] += 1
                hours["too_short"] += duration / 3600

                continue

            #
            # max-duration은 필요하면 사용할 수 있지만
            # 보통 SSL에서는 긴 파일도 manifest에는 살려두고
            # Dataset에서 random crop 하는 걸 추천.
            #
            if (
                args.max_duration > 0
                and duration > args.max_duration
            ):

                stats["too_long"] += 1
                hours["too_long"] += duration / 3600

                continue

            #
            # mono / stereo 외 채널은 우선 제외
            #
            if channels not in (1, 2):

                stats["unsupported_channels"] += 1

                continue

            split = stable_split(
                path,
                args.valid_percent,
            )

            output_fp = (
                train_fp
                if split == "train"
                else valid_fp
            )

            source = source_map[str(path)]

            #
            # 공통 metadata
            #
            common = {
                "path": str(path),

                # 원본 waveform 정보
                "sample_rate": sr,
                "num_frames": frames,
                "duration": duration,

                # 데이터 provenance
                "source": source,

                # train / valid
                "split": split,
            }

            #
            # mono
            #
            if channels == 1:

                item = {
                    **common,

                    "channel": 0,
                    "channel_name": "mono",

                    "num_channels": 1,

                    # 원본 파일 identification
                    "utterance_id": path.stem,

                    # 실제 학습 sample identification
                    "sample_id": f"{path.stem}:mono",
                }

                write_jsonl(
                    output_fp,
                    item,
                )

                stats["mono_files"] += 1
                stats["virtual_samples"] += 1

                stats[f"{split}_samples"] += 1

                hours["mono"] += duration / 3600
                hours[f"{split}_virtual"] += duration / 3600

            #
            # stereo
            #
            elif channels == 2:

                stats["stereo_files"] += 1
                hours["stereo_original"] += duration / 3600

                #
                # 동일한 wav를 두 번 등록.
                #
                # 파일 복제 없음.
                #
                # channel=0
                # channel=1
                #
                for channel in (0, 1):

                    channel_name = stereo_map[channel]

                    item = {
                        **common,

                        "channel": channel,
                        "channel_name": channel_name,

                        "num_channels": 2,

                        "utterance_id": path.stem,

                        "sample_id": (
                            f"{path.stem}:{channel_name}"
                        ),
                    }

                    write_jsonl(
                        output_fp,
                        item,
                    )

                    stats["virtual_samples"] += 1
                    stats[f"{channel_name}_samples"] += 1
                    stats[f"{split}_samples"] += 1

                    hours[channel_name] += duration / 3600

                    hours[
                        f"{split}_virtual"
                    ] += duration / 3600

            stats["valid_files"] += 1

            hours["original_audio"] += (
                duration / 3600
            )

    #
    # stats
    #
    stats_output = {
        "config": {
            "audio_dirs": args.audio_dirs,
            "sample_rate": args.sample_rate,
            "valid_percent": args.valid_percent,
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "stereo_map": args.stereo_map,
        },

        "counts": dict(stats),

        "hours": {
            k: round(v, 3)
            for k, v in hours.items()
        },
    }

    stats_path = (
        output_dir / "stats.json"
    )

    with stats_path.open(
        "w",
        encoding="utf-8",
    ) as fp:

        json.dump(
            stats_output,
            fp,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("Manifest generation complete")
    print("=" * 70)

    print(
        f"original wav files : "
        f"{stats['valid_files']:,}"
    )

    print(
        f"mono files         : "
        f"{stats['mono_files']:,}"
    )

    print(
        f"stereo files       : "
        f"{stats['stereo_files']:,}"
    )

    print(
        f"virtual samples    : "
        f"{stats['virtual_samples']:,}"
    )

    print()

    print(
        f"train samples      : "
        f"{stats['train_samples']:,}"
    )

    print(
        f"valid samples      : "
        f"{stats['valid_samples']:,}"
    )

    print()

    print(
        f"original hours     : "
        f"{hours['original_audio']:,.1f} h"
    )

    print(
        f"training hours*    : "
        f"{hours['train_virtual']:,.1f} h"
    )

    print(
        f"validation hours*  : "
        f"{hours['valid_virtual']:,.1f} h"
    )

    print()
    print(
        "* stereo RX/TX를 각각 하나의 sample로 "
        "계산한 virtual hours"
    )

    print()

    print(
        f"wrong sample rate  : "
        f"{stats['wrong_sample_rate']:,}"
    )

    print(
        f"broken             : "
        f"{stats['broken_files']:,}"
    )

    print(
        f"too short          : "
        f"{stats['too_short']:,}"
    )

    print(
        f"unsupported ch     : "
        f"{stats['unsupported_channels']:,}"
    )

    print()
    print(f"train : {train_path}")
    print(f"valid : {valid_path}")
    print(f"stats : {stats_path}")


if __name__ == "__main__":
    main()
