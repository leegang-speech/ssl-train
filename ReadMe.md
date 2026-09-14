입력 디렉토리 2개 이상 가능
하위 디렉토리까지 .wav 재귀 탐색
기본적으로 8 kHz만 허용
mono → sample 1개
stereo → RX/TX sample 2개
실제 wav split/복사 안 함
channel만 manifest에 저장
RX/TX channel 순서 지정 가능
train/valid는 원본 wav 단위 split
깨진 wav / 8k 아닌 wav / 3채널 이상 등의 통계 출력
너무 짧은 파일 제외 가능
나중에 HuBERT Dataset에서 바로 사용할 수 있도록 JSONL 생성


hubert_ssl/
├── build_manifest.py
└── manifests/
    ├── train.jsonl
    ├── valid.jsonl
    └── stats.json

예를 들어 데이터가:

/data/unlabeled/callbot/
/data/unlabeled/counsel/
/nas/legacy_call/

에 있다고 하면:

python build_manifest.py \
    --audio-dirs \
        /data/unlabeled/callbot \
        /data/unlabeled/counsel \
        /nas/legacy_call \
    --output-dir ./manifests \
    --sample-rate 8000 \
    --valid-percent 1 \
    --min-duration 1.0 \
    --stereo-map rx-tx

그러면:

manifests/
├── train.jsonl
├── valid.jsonl
└── stats.json

이 생긴다.
