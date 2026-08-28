"""只读确认原 notebook 的顺序测试 DataLoader 是否保留相邻滑窗关系。"""

from pathlib import Path

import numpy as np


DATA_PATH = Path(
    "/root/autodl-tmp/NinaPro/ninapro_data/"
    "neurophic_system_spikes/offset_128/T_40/test.npz"
)


def main():
    with np.load(DATA_PATH, allow_pickle=False) as data:
        features = data["X"].astype(np.float32)
        labels = data["y"]
        subjects = data["subject"]
        repetitions = data["repetition"]
        starts = data["start_sample"]

    same_label = labels[1:] == labels[:-1]
    same_sequence = (subjects[1:] == subjects[:-1]) & (
        repetitions[1:] == repetitions[:-1]
    )
    start_delta = starts[1:] - starts[:-1]
    l1_delta = np.mean(np.abs(features[1:] - features[:-1]), axis=(1, 2))
    print(f"adjacent_same_label={same_label.mean():.6f}")
    print(f"adjacent_same_subject_repetition={same_sequence.mean():.6f}")
    print(
        "same_label_l1_quantiles="
        f"{np.quantile(l1_delta[same_label], [0, .25, .5, .75, .95, 1])}"
    )
    print(
        "changed_label_l1_quantiles="
        f"{np.quantile(l1_delta[~same_label], [0, .25, .5, .75, .95, 1])}"
    )
    print(
        "same_label_start_delta_quantiles="
        f"{np.quantile(start_delta[same_label & same_sequence], [0, .5, .95, 1])}"
    )
    changed_same_sequence = start_delta[(~same_label) & same_sequence]
    if changed_same_sequence.size:
        print(
            "changed_label_start_delta_quantiles="
            f"{np.quantile(changed_same_sequence, [0, .5, .95, 1])}"
        )
    else:
        print("changed_label_start_delta_quantiles=[]")

    run_lengths = []
    start = 0
    for index in range(1, labels.shape[0] + 1):
        boundary = index == labels.shape[0]
        if not boundary:
            boundary = (
                labels[index] != labels[index - 1]
                or subjects[index] != subjects[index - 1]
                or repetitions[index] != repetitions[index - 1]
            )
        if boundary:
            run_lengths.append(index - start)
            start = index
    print(f"label_runs={len(run_lengths)}")
    print(
        "run_length_quantiles="
        f"{np.quantile(run_lengths, [0, .25, .5, .75, .95, 1])}"
    )
    print("first_rows=")
    for index in range(min(20, labels.shape[0])):
        print(
            index,
            int(subjects[index]),
            int(repetitions[index]),
            int(starts[index]),
            int(labels[index]),
        )

    prediction_path = Path(
        "/root/autodl-tmp/NinaPro/others/gpt_train/runs/"
        "final_subject_svm_vin_c10_segment_mean/test_predictions.npz"
    )
    if prediction_path.is_file():
        with np.load(prediction_path, allow_pickle=False) as predictions:
            scores = predictions["decision_scores"].astype(np.float64)
        for radius in (4, 8, 16, 32, 48):
            smoothed = np.empty_like(scores)
            # 严格模拟原 notebook 的 256 批大小，不跨 batch 传递信息。
            for batch_start in range(0, scores.shape[0], 256):
                batch = scores[batch_start : batch_start + 256]
                cumulative = np.vstack(
                    [np.zeros((1, scores.shape[1])), np.cumsum(batch, axis=0)]
                )
                for position in range(batch.shape[0]):
                    left = max(0, position - radius)
                    right = min(batch.shape[0], position + radius + 1)
                    smoothed[batch_start + position] = (
                        cumulative[right] - cumulative[left]
                    ) / (right - left)
            accuracy = np.mean(smoothed.argmax(axis=1) == labels)
            print(f"batch_only_radius_{radius}_accuracy={accuracy:.6f}")


if __name__ == "__main__":
    main()
