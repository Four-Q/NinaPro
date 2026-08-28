"""按已冻结的验证方案训练受试者专家并执行一次最终测试。"""

import json
from pathlib import Path

import joblib
import numpy as np

from classical_baseline import (
    DATA_DIR,
    contiguous_segments,
    extract_features,
    make_classifier,
    smooth_decision_scores,
)


HERE = Path(__file__).resolve().parent
RUN_NAME = "final_subject_svm_vin_c10_segment_mean"


def classification_metrics(labels, predictions, num_classes=12):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    correct = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    recall = correct / np.maximum(support, 1)
    precision = correct / np.maximum(predicted, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(correct.sum() / confusion.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_accuracy": recall.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def load_split(name):
    with np.load(DATA_DIR / f"{name}.npz", allow_pickle=False) as data:
        return {
            "features": extract_features(np.asarray(data["Vin"]), "Vin"),
            "labels": np.asarray(data["y"]).copy(),
            "subjects": np.asarray(data["subject"]).copy(),
            "repetitions": np.asarray(data["repetition"]).copy(),
            "start_samples": np.asarray(data["start_sample"]).copy(),
        }


def main():
    output_dir = HERE / "runs" / RUN_NAME
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    test = load_split("test")
    raw_predictions = np.full(test["labels"].shape, -1, dtype=np.int64)
    decision_scores = np.full(
        (test["labels"].shape[0], 12), np.nan, dtype=np.float64
    )
    subject_results = {}

    for subject in sorted(np.unique(train["subjects"])):
        train_mask = train["subjects"] == subject
        test_mask = test["subjects"] == subject
        classifier = make_classifier(c_value=10.0, gamma="scale")
        classifier.fit(train["features"][train_mask], train["labels"][train_mask])
        predictions = classifier.predict(test["features"][test_mask])
        scores = classifier.decision_function(test["features"][test_mask])
        raw_predictions[test_mask] = predictions
        decision_scores[test_mask] = scores
        joblib.dump(classifier, model_dir / f"subject_{int(subject):02d}.joblib")
        subject_results[str(int(subject))] = {
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "raw_accuracy": float(np.mean(predictions == test["labels"][test_mask])),
        }
        print(
            f"subject={subject} raw_test_accuracy="
            f"{subject_results[str(int(subject))]['raw_accuracy'] * 100:.2f}%",
            flush=True,
        )

    test_indices = np.arange(test["labels"].shape[0])
    segments = contiguous_segments(
        test_indices,
        test["subjects"],
        test["repetitions"],
        test["start_samples"],
    )
    # 验证阶段已固定整段平均，不再根据测试结果选择平滑半径。
    smoothed_predictions = smooth_decision_scores(
        decision_scores, segments, radius=-1
    )
    raw_metrics = classification_metrics(test["labels"], raw_predictions)
    smoothed_metrics = classification_metrics(
        test["labels"], smoothed_predictions
    )
    result = {
        "run_name": RUN_NAME,
        "selection_protocol": {
            "selection_split": "train.npz repetition 6",
            "input_field": "Vin",
            "scope": "one RBF-SVM expert per known subject",
            "classifier": "StandardScaler+RBF-SVC",
            "c": 10.0,
            "gamma": "scale",
            "postprocessing": "mean decision scores within metadata-derived segment",
            "test_evaluations": 1,
        },
        "training_repetitions": [1, 3, 4, 6],
        "test_repetitions": [2, 5],
        "train_samples": int(train["labels"].shape[0]),
        "test_samples": int(test["labels"].shape[0]),
        "sequence_segments": len(segments),
        "mean_segment_size": float(np.mean([len(segment) for segment in segments])),
        "raw_window_metrics": raw_metrics,
        "segment_smoothed_metrics": smoothed_metrics,
        "subject_results": subject_results,
    }
    (output_dir / "test_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        raw_predictions=raw_predictions,
        smoothed_predictions=smoothed_predictions,
        decision_scores=decision_scores.astype(np.float32),
        subject=test["subjects"],
        repetition=test["repetitions"],
        start_sample=test["start_samples"],
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
