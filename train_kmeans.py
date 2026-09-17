import argparse
import joblib
import numpy as np

from sklearn.cluster import MiniBatchKMeans


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mfcc",
        required=True,
        help="train_0_1.npy",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="저장할 kmeans 모델",
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10_000_000,
        help="K-means 학습에 사용할 MFCC frame 수",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    args = parser.parse_args()

    #
    # 전체 npy를 RAM에 올리지 않음
    #
    features = np.load(
        args.mfcc,
        mmap_mode="r",
    )

    print("MFCC shape:", features.shape)

    num_frames = features.shape[0]

    rng = np.random.default_rng(args.seed)

    num_samples = min(
        args.num_samples,
        num_frames,
    )

    print(
        f"Sampling {num_samples:,} "
        f"from {num_frames:,} frames"
    )

    indices = rng.choice(
        num_frames,
        size=num_samples,
        replace=False,
    )

    #
    # numpy memmap에서 random indexing
    #
    train_features = np.asarray(
        features[indices],
        dtype=np.float32,
    )

    print(
        "Training features:",
        train_features.shape,
    )

    #
    # K-means
    #
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,

        batch_size=args.batch_size,

        random_state=args.seed,

        n_init=10,

        max_iter=200,

        verbose=1,
    )

    kmeans.fit(train_features)

    print()
    print("K-means complete")
    print(
        "centroids:",
        kmeans.cluster_centers_.shape,
    )

    #
    # 모델 저장
    #
    joblib.dump(
        kmeans,
        args.output,
    )

    print(
        "saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
