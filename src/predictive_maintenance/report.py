"""Export reproducible report evidence and publication-ready figures."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from .features import FEATURES, select_features
from .splits import SEED, split_dataset

DATASET_URL = "https://archive.ics.uci.edu/dataset/601/ai4i"
DATASET_DOI = "10.24432/C5HS5C"
SHORT_NAMES = ["Type", "Air temp", "Process temp", "Rotation", "Torque", "Tool wear"]


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _evaluation_figure(target, probabilities, threshold, output, average_precision):
    precision, recall, _ = precision_recall_curve(target, probabilities)
    predicted = probabilities >= threshold
    matrix = confusion_matrix(target, predicted, labels=[0, 1])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].step(recall, precision, where="post", color="#167d8d", linewidth=2)
    axes[0].axhline(float(target.mean()), color="#8d9299", linestyle="--", label="Failure prevalence")
    axes[0].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title=f"Precision–recall (AP = {average_precision:.3f})")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1].text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=16,
                         color="white" if matrix[row, column] > matrix.max() / 2 else "#172638")
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No failure", "Failure"],
                yticklabels=["No failure", "Failure"], xlabel="Predicted", ylabel="Actual",
                title=f"Test confusion matrix (threshold = {threshold:.2f})")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _drift_figure(normal, shifted, output):
    index = np.arange(len(FEATURES))
    normal_scores = [normal["features"][name]["statistic"] / normal["features"][name]["threshold"] for name in FEATURES]
    shifted_scores = [shifted["features"][name]["statistic"] / shifted["features"][name]["threshold"] for name in FEATURES]
    figure, axis = plt.subplots(figsize=(10, 4.6))
    axis.bar(index - 0.19, normal_scores, 0.38, color="#6baaa5", label="Unchanged demo batch")
    axis.bar(index + 0.19, shifted_scores, 0.38, color="#df8667", label="Shifted demo batch")
    axis.axhline(1, color="#374151", linestyle="--", linewidth=1, label="Alert threshold")
    axis.set(xticks=index, xticklabels=SHORT_NAMES, ylabel="Distance / alert threshold",
             title="Input distribution shift by feature")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _markdown(evidence):
    metrics = evidence["metrics"]
    normal = evidence["monitoring"]["normal"]
    shifted = evidence["monitoring"]["shifted"]
    test = metrics["test"]
    validation_rows = "\n".join(
        f"| {name} | {values['average_precision']:.3f} | {values['precision']:.3f} | "
        f"{values['recall']:.3f} | {values['f2']:.3f} |"
        for name, values in metrics["validation"].items()
    )
    drift_rows = "\n".join(
        f"| {short} | {normal['features'][name]['statistic']:.3f} | "
        f"{shifted['features'][name]['statistic']:.3f} | {shifted['features'][name]['threshold']:.2f} | "
        f"{'Có' if shifted['features'][name]['drift_detected'] else 'Không'} |"
        for name, short in zip(FEATURES, SHORT_NAMES)
    )
    versions = evidence["environment"]
    return f"""# Bằng chứng cho báo cáo project

Tệp này được tạo bằng `python -m predictive_maintenance.report` từ dữ liệu, mô hình và báo cáo giám sát của project. Chạy lại lệnh đó sau khi `dvc repro` và hai lô demo đã được tạo để cập nhật số liệu.

## Nguồn dữ liệu và thiết kế đánh giá

- Dataset: [AI4I 2020 Predictive Maintenance]({DATASET_URL}), DOI [{DATASET_DOI}](https://doi.org/{DATASET_DOI}), dữ liệu mô phỏng, giấy phép CC BY 4.0.
- SHA-256 của CSV gốc: `{evidence['dataset_sha256']}`.
- Tổng số bản ghi: **{metrics['dataset_rows']:,}**; tỷ lệ hỏng: **{metrics['failure_rate']:.2%}**.
- Chia stratified với seed **{SEED}**: train **{metrics['split_rows']['train']:,}**, validation **{metrics['split_rows']['validation']:,}**, test **{metrics['split_rows']['test']:,}**.
- Chỉ dùng sáu đặc trưng đầu vào; bỏ `UID`, `Product ID` và các nhãn nguyên nhân hỏng (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) để tránh rò rỉ nhãn.
- Chọn mô hình bằng average precision trên validation; chọn ngưỡng cảnh báo bằng F2 trên validation; test chỉ dùng để báo cáo mô hình đã chọn.

## So sánh mô hình trên validation

| Mô hình | PR-AUC | Precision | Recall | F2 |
| --- | ---: | ---: | ---: | ---: |
{validation_rows}

## Kết quả mô hình được chọn trên test

Mô hình: **{metrics['selected_model']}**, ngưỡng: **{metrics['threshold']:.2f}**. PR-AUC **{test['average_precision']:.3f}**, ROC-AUC **{test['roc_auc']:.3f}**, precision **{test['precision']:.3f}**, recall **{test['recall']:.3f}**, F1 **{test['f1']:.3f}**, F2 **{test['f2']:.3f}**.

| Thực tế / dự đoán | Không hỏng | Hỏng |
| --- | ---: | ---: |
| Không hỏng | {test['true_negative']} | {test['false_positive']} |
| Hỏng | {test['false_negative']} | {test['true_positive']} |

![Precision–recall và ma trận nhầm lẫn](figures/model_evaluation.png)

## Demo giám sát dữ liệu đầu vào

Mốc tham chiếu gồm **{normal['reference_rows']:,}** bản ghi train. Mỗi lô demo có **{normal['current_rows']}** bản ghi từ phần dữ liệu giữ lại. Lô dịch chuyển được tạo bằng cách cộng 5 K vào nhiệt độ không khí, cộng 10 Nm vào mô-men xoắn và đặt 60% bản ghi thành loại H.

| Đặc trưng | Lô bình thường | Lô dịch chuyển | Ngưỡng | Cảnh báo lô dịch chuyển |
| --- | ---: | ---: | ---: | --- |
{drift_rows}

Lô bình thường: **{normal['status']}**. Lô dịch chuyển: **{shifted['status']}** với các đặc trưng được đánh dấu: **{', '.join(shifted['drifted_features'])}**. Các giá trị trong bảng là khoảng cách total variation cho `Type` và thống kê KS cho đặc trưng số; đặc trưng số còn phải có p-value < 0,01 mới phát cảnh báo.

![So sánh mức dịch chuyển đầu vào](figures/drift_scores.png)

## Môi trường và giới hạn

Python {versions['python']}; scikit-learn {versions['scikit-learn']}; XGBoost {versions['xgboost']}; MLflow {versions['mlflow']}.

AI4I là dữ liệu mô phỏng. Nhãn mô tả máy hỏng **tại bản ghi hiện tại**, không đo thời gian còn lại trước khi hỏng. Cách chia stratified không chứng minh khả năng dự báo theo thời gian trên máy thật. Ngưỡng drift là ngưỡng demo; thay đổi phân phối đầu vào không tự chứng minh hiệu năng mô hình đã giảm. Muốn đo hiệu năng khi triển khai cần thu thập nhãn thực và đánh giá theo thời gian.
"""


def export(data_path, model_path, metrics_path, normal_path, shifted_path, output_dir):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    metrics = _load_json(metrics_path)
    normal = _load_json(normal_path)
    shifted = _load_json(shifted_path)
    model_bundle = joblib.load(model_path)
    frame = pd.read_csv(data_path)
    features, target = select_features(frame)
    _, _, test_x, _, _, test_y = split_dataset(features, target)
    probabilities = model_bundle["model"].predict_proba(test_x)[:, 1]
    average_precision = average_precision_score(test_y, probabilities)
    matrix = confusion_matrix(test_y, probabilities >= model_bundle["threshold"], labels=[0, 1])
    expected = metrics["test"]
    if abs(average_precision - expected["average_precision"]) > 1e-10:
        raise ValueError("Saved model and reported test PR-AUC disagree")
    if matrix.tolist() != [[expected["true_negative"], expected["false_positive"]],
                           [expected["false_negative"], expected["true_positive"]]]:
        raise ValueError("Saved model and reported confusion matrix disagree")
    if len(test_x) != metrics["split_rows"]["test"]:
        raise ValueError("Test split does not match metrics")
    evidence = {
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "dataset_source": DATASET_URL,
        "dataset_doi": DATASET_DOI,
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
            "mlflow": version("mlflow"),
        },
        "metrics": metrics,
        "monitoring": {"normal": normal, "shifted": shifted},
    }
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _evaluation_figure(test_y, probabilities, model_bundle["threshold"],
                       figures / "model_evaluation.png", average_precision)
    _drift_figure(normal, shifted, figures / "drift_scores.png")
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    (output_dir / "evidence.md").write_text(_markdown(evidence), encoding="utf-8", newline="\n")
    print(f"Verified metrics and wrote report evidence to {output_dir}")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/ai4i2020.csv"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument("--normal", type=Path, default=Path("artifacts/monitoring/normal.json"))
    parser.add_argument("--shifted", type=Path, default=Path("artifacts/monitoring/drifted.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    export(args.data, args.model, args.metrics, args.normal, args.shifted, args.output_dir)


if __name__ == "__main__":
    main()
