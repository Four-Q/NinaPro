"""用重复 6 评估全局与受试者专属的经典分类基线。"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
DATA_DIR = Path(
    "/root/autodl-tmp/NinaPro/ninapro_data/"
    "neurophic_system_spikes/offset_128/T_40"
)
ZERO_VOLTAGE = 1.8639
MAXIMUM_VOLTAGE = 3.7278


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-field", choices=("X", "Vin"), default="Vin")
    parser.add_argument("--scope", choices=("global", "subject"), default="subject")
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def recover_signed_voltage(values):
    negative = (values - ZERO_VOLTAGE) / ZERO_VOLTAGE
    positive = (values - ZERO_VOLTAGE) / (MAXIMUM_VOLTAGE - ZERO_VOLTAGE)
    return np.where(values < ZERO_VOLTAGE, negative, positive)


def extract_features(values, input_field):
    if input_field == "Vin":
        values = recover_signed_voltage(values)
    values = values.astype(np.float32, copy=False)
    difference = np.diff(values, axis=2)
    mean = values.mean(axis=2)
    std = values.std(axis=2)
    rms = np.sqrt(np.mean(values**2, axis=2) + 1e-8)
    mav = np.mean(np.abs(values), axis=2)
    waveform_length = np.mean(np.abs(difference), axis=2)
    zero_crossing = np.mean(values[:, :, 1:] * values[:, :, :-1] < 0, axis=2)
    slope_change = np.mean(
        difference[:, :, 1:] * difference[:, :, :-1] < 0, axis=2
    )

    # 8 个时间分段保留动作窗口内部的粗粒度演化方向。
    segments = values.reshape(values.shape[0], values.shape[1], 8, 5)
    segment_mean = segments.mean(axis=3).reshape(values.shape[0], -1)
    segment_std = segments.std(axis=3).reshape(values.shape[0], -1)
    spectrum = np.abs(np.fft.rfft(values, axis=2))[:, :, 1:11]
    return np.concatenate(
        [
            values.reshape(values.shape[0], -1),
            segment_mean,
            segment_std,
            spectrum.reshape(values.shape[0], -1),
            mean,
            std,
            rms,
            mav,
            waveform_length,
            zero_crossing,
            slope_change,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def make_classifier(c_value, gamma):
    parsed_gamma = float(gamma) if gamma not in {"scale", "auto"} else gamma
    return make_pipeline(
        StandardScaler(),
        SVC(C=c_value, gamma=parsed_gamma, kernel="rbf", cache_size=6000),
    )


def contiguous_segments(indices, subjects, repetitions, start_samples):
    segments = []
    for subject in sorted(np.unique(subjects[indices])):
        for repetition in sorted(np.unique(repetitions[indices])):
            selected = indices[
                (subjects[indices] == subject) & (repetitions[indices] == repetition)
            ]
            order = selected[np.argsort(start_samples[selected])]
            if order.size == 0:
                continue
            differences = np.diff(start_samples[order])
            positive = differences[differences > 0]
            stride = int(positive.min()) if positive.size else 1
            # 滑窗内部步长固定；更大的时间缺口对应两个动作区段之间的休息期。
            boundaries = np.flatnonzero(differences > stride * 3) + 1
            segments.extend(np.split(order, boundaries))
    return segments


def smooth_decision_scores(scores, segments, radius):
    predictions = np.full(scores.shape[0], -1, dtype=np.int64)
    for segment in segments:
        segment_scores = scores[segment]
        if radius < 0:
            smoothed = np.repeat(
                segment_scores.mean(axis=0, keepdims=True), len(segment), axis=0
            )
        else:
            smoothed = np.empty_like(segment_scores)
            cumulative = np.vstack(
                [np.zeros((1, scores.shape[1])), np.cumsum(segment_scores, axis=0)]
            )
            for position in range(len(segment)):
                left = max(0, position - radius)
                right = min(len(segment), position + radius + 1)
                smoothed[position] = (cumulative[right] - cumulative[left]) / (
                    right - left
                )
        predictions[segment] = smoothed.argmax(axis=1)
    return predictions


def main():
    args = parse_args()
    with np.load(DATA_DIR / "train.npz", allow_pickle=False) as data:
        values = np.asarray(data[args.input_field])
        labels = np.asarray(data["y"])
        subjects = np.asarray(data["subject"])
        repetitions = np.asarray(data["repetition"])
        start_samples = np.asarray(data["start_sample"])

    features = extract_features(values, args.input_field)
    train_mask = repetitions != 6
    validation_mask = repetitions == 6
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    decision_scores = np.full((labels.shape[0], 12), np.nan, dtype=np.float64)
    subject_results = {}

    if args.scope == "global":
        classifier = make_classifier(args.c, args.gamma)
        classifier.fit(features[train_mask], labels[train_mask])
        predictions[validation_mask] = classifier.predict(features[validation_mask])
        decision_scores[validation_mask] = classifier.decision_function(
            features[validation_mask]
        )
    else:
        for subject in sorted(np.unique(subjects)):
            subject_train = train_mask & (subjects == subject)
            subject_validation = validation_mask & (subjects == subject)
            classifier = make_classifier(args.c, args.gamma)
            classifier.fit(features[subject_train], labels[subject_train])
            subject_predictions = classifier.predict(features[subject_validation])
            predictions[subject_validation] = subject_predictions
            decision_scores[subject_validation] = classifier.decision_function(
                features[subject_validation]
            )
            accuracy = float(
                np.mean(subject_predictions == labels[subject_validation])
            )
            subject_results[str(int(subject))] = {
                "train_samples": int(subject_train.sum()),
                "validation_samples": int(subject_validation.sum()),
                "accuracy": accuracy,
            }
            print(f"subject={subject} validation_accuracy={accuracy * 100:.2f}%")

    accuracy = float(np.mean(predictions[validation_mask] == labels[validation_mask]))
    validation_indices = np.flatnonzero(validation_mask)
    segments = contiguous_segments(
        validation_indices, subjects, repetitions, start_samples
    )
    smoothing_results = {}
    for radius in (1, 2, 4, 8, 16, 32, -1):
        smoothed = smooth_decision_scores(decision_scores, segments, radius)
        smoothed_accuracy = float(
            np.mean(smoothed[validation_mask] == labels[validation_mask])
        )
        name = "whole_segment" if radius < 0 else f"radius_{radius}"
        smoothing_results[name] = smoothed_accuracy
        print(f"smoothing={name} validation_accuracy={smoothed_accuracy * 100:.2f}%")

    segment_purities = []
    for segment in segments:
        counts = np.bincount(labels[segment], minlength=12)
        segment_purities.append(float(counts.max() / counts.sum()))
    result = {
        "input_field": args.input_field,
        "scope": args.scope,
        "classifier": "StandardScaler+RBF-SVC",
        "c": args.c,
        "gamma": args.gamma,
        "feature_count": int(features.shape[1]),
        "validation_repetition": 6,
        "validation_samples": int(validation_mask.sum()),
        "accuracy": accuracy,
        "sequence_segments": len(segments),
        "mean_segment_size": float(np.mean([len(segment) for segment in segments])),
        "mean_segment_label_purity": float(np.mean(segment_purities)),
        "smoothing_results": smoothing_results,
        "subject_results": subject_results,
    }
    run_name = args.run_name or f"svc_{args.scope}_{args.input_field}_c{args.c:g}"
    output_dir = HERE / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
