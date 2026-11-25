"""Create report figures and LaTeX table rows from verified project artifacts."""

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import confusion_matrix, fbeta_score

from predictive_maintenance.features import FEATURES, NUMERIC_FEATURES, select_features
from predictive_maintenance.splits import SEED, split_dataset

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
GENERATED = HERE / "generated"
DATA = ROOT / "data/raw/ai4i2020.csv"
METRICS = ROOT / "artifacts/metrics.json"
MODEL = ROOT / "artifacts/model.joblib"
EVIDENCE = ROOT / "reports/evidence.json"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "savefig.facecolor": "white",
})


def dec(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def integer(value):
    return f"{int(value):,}".replace(",", ".")


def write_rows(name, rows):
    (GENERATED / name).write_text("\n".join(row + r" \\" for row in rows) + "\n\\bottomrule\n", encoding="utf-8", newline="\n")


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=190)
    plt.close()


def box(axis, xy, label, color, width=2.1, height=0.78):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.06,rounding_size=0.1",
                           linewidth=1.3, edgecolor="#294054", facecolor=color)
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, label, ha="center", va="center",
              color="#142538", fontsize=9, fontweight="bold")


def arrow(axis, start, end):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                   linewidth=1.5, color="#4b647a"))


def architecture():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, (0.3, 3.8), "UCI AI4I CSV", "#d9ebf6")
    box(ax, (3.2, 3.8), "DVC pipeline", "#d9ebf6")
    box(ax, (6.2, 3.8), "Train + validation", "#e8e2fb")
    box(ax, (9.2, 3.8), "MLflow tracking", "#e8e2fb")
    box(ax, (6.2, 2.2), "Model artifact", "#d8f0df")
    box(ax, (9.2, 2.2), "FastAPI + Docker", "#d8f0df")
    box(ax, (3.2, 0.6), "Training reference", "#fff0d5")
    box(ax, (6.2, 0.6), "Batch drift check", "#fff0d5")
    box(ax, (9.2, 0.6), "JSON drift report", "#fff0d5")
    arrow(ax, (2.45, 4.2), (3.14, 4.2))
    arrow(ax, (5.35, 4.2), (6.14, 4.2))
    arrow(ax, (8.35, 4.2), (9.14, 4.2))
    arrow(ax, (7.25, 3.75), (7.25, 3.05))
    arrow(ax, (8.35, 2.6), (9.14, 2.6))
    arrow(ax, (6.35, 2.25), (4.35, 1.43))
    arrow(ax, (5.35, 1.0), (6.14, 1.0))
    arrow(ax, (8.35, 1.0), (9.14, 1.0))
    ax.text(0.4, 0.1, "GitHub Actions checks source, API contract and monitoring logic on each push.",
            color="#4b647a", fontsize=9)
    save("architecture.png")


def generate():
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA)
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if digest != evidence["dataset_sha256"]:
        raise ValueError("Dataset hash differs from report evidence")
    if metrics != evidence["metrics"]:
        raise ValueError("Metrics differ from report evidence")
    features, target = select_features(frame)
    train_x, validation_x, test_x, train_y, validation_y, test_y = split_dataset(features, target)
    bundle = joblib.load(MODEL)
    if len(train_x) != metrics["split_rows"]["train"]:
        raise ValueError("Split size differs from metrics")

    split_rows = []
    for label, values in (("Train", train_y), ("Validation", validation_y), ("Test", test_y)):
        split_rows.append(f"{label} & {integer(len(values))} & {integer(values.sum())} & {dec(values.mean()*100)}\\%")
    write_rows("split_rows.tex", split_rows)

    feature_labels = ["Air temperature (K)", "Process temperature (K)", "Rotational speed (rpm)",
                      "Torque (Nm)", "Tool wear (min)"]
    data_rows = []
    for column, label in zip(NUMERIC_FEATURES, feature_labels):
        series = frame[column]
        data_rows.append(f"{label} & {dec(series.mean())} & {dec(series.std())} & "
                         f"{dec(series.min())} & {dec(series.max())}")
    write_rows("data_rows.tex", data_rows)

    type_rows = []
    for category in ("L", "M", "H"):
        subset = frame[frame["Type"] == category]
        type_rows.append(f"{category} & {integer(len(subset))} & {integer(subset['Machine failure'].sum())} & "
                         f"{dec(subset['Machine failure'].mean()*100)}\\%")
    write_rows("type_rows.tex", type_rows)

    mode_rows = []
    for mode in ("TWF", "HDF", "PWF", "OSF", "RNF"):
        mode_rows.append(f"{mode} & {integer(frame[mode].sum())}")
    write_rows("mode_rows.tex", mode_rows)

    model_labels = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
    validation_rows = []
    for key, value in metrics["validation"].items():
        validation_rows.append(f"{model_labels[key]} & {dec(value['average_precision'],3)} & "
                               f"{dec(value['precision'],3)} & {dec(value['recall'],3)} & "
                               f"{dec(value['f2'],3)}")
    write_rows("validation_rows.tex", validation_rows)

    drift_rows = []
    normal = evidence["monitoring"]["normal"]
    shifted = evidence["monitoring"]["shifted"]
    for column, label in zip(FEATURES, ["Type", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]):
        a = normal["features"][column]
        b = shifted["features"][column]
        drift_rows.append(f"{label} & {dec(a['statistic'],3)} & {dec(b['statistic'],3)} & "
                          f"{dec(b['threshold'],2)} & {'Có' if b['drift_detected'] else 'Không'}")
    write_rows("drift_rows.tex", drift_rows)

    test_probability = bundle["model"].predict_proba(test_x)[:, 1]
    test_prediction = test_probability >= bundle["threshold"]
    tn, fp, fn, tp = confusion_matrix(test_y, test_prediction, labels=[0, 1]).ravel()
    expected = metrics["test"]
    if (tn, fp, fn, tp) != (expected["true_negative"], expected["false_positive"],
                            expected["false_negative"], expected["true_positive"]):
        raise ValueError("Model predictions differ from recorded test confusion matrix")
    test_details = test_x.copy()
    test_details["label"] = test_y
    test_details["prediction"] = test_prediction.astype(int)
    test_details["probability"] = test_probability
    subgroup_rows = []
    for category in ("L", "M", "H"):
        subset = test_details[test_details["Type"] == category]
        a, b, c, d = confusion_matrix(subset["label"], subset["prediction"], labels=[0, 1]).ravel()
        subgroup_rows.append(f"{category} & {integer(len(subset))} & {integer(subset['label'].sum())} & "
                             f"{integer(d)} & {integer(c)} & {integer(b)}")
    write_rows("subgroup_rows.tex", subgroup_rows)
    error_rows = []
    for label, actual, predicted in (("TP", 1, 1), ("FN", 1, 0), ("FP", 0, 1), ("TN", 0, 0)):
        subset = test_details[(test_details["label"] == actual) & (test_details["prediction"] == predicted)]
        error_rows.append(f"{label} & {integer(len(subset))} & {dec(subset['probability'].mean(),3)} & "
                          f"{dec(subset['Torque [Nm]'].mean())} & {dec(subset['Tool wear [min]'].mean())}")
    write_rows("error_rows.tex", error_rows)

    counts = target.value_counts().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(["No failure", "Failure"], counts.values, color=["#5b9cad", "#d97c61"])
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, integer(count), ha="center")
    ax.set(ylabel="Number of records", title="Class distribution in AI4I 2020")
    ax.set_ylim(0, 10600)
    save("class_balance.png")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for ax, column, label in zip(axes.flat, NUMERIC_FEATURES, feature_labels):
        ax.hist(frame.loc[target == 0, column], bins=28, density=True, alpha=0.55,
                color="#5b9cad", label="No failure")
        ax.hist(frame.loc[target == 1, column], bins=28, density=True, alpha=0.55,
                color="#d97c61", label="Failure")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Density")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.87, 0.13))
    save("sensor_distributions.png")

    rates = frame.groupby("Type")["Machine failure"].mean().reindex(["L", "M", "H"]) * 100
    fig, ax = plt.subplots(figsize=(6.7, 4))
    bars = ax.bar(rates.index, rates.values, color=["#5b9cad", "#7fae9a", "#d97c61"])
    for bar, rate in zip(bars, rates.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.06, f"{rate:.2f}%", ha="center")
    ax.set(xlabel="Product type", ylabel="Failure rate (%)", title="Observed failure rate by product type")
    ax.set_ylim(0, max(rates.values)*1.22)
    save("type_failure_rate.png")

    mode_counts = frame[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(mode_counts.index, mode_counts.values, color="#789ab4")
    for bar, count in zip(bars, mode_counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, str(int(count)), ha="center")
    ax.set(ylabel="Records with mode label = 1", title="Failure-mode labels (overlap is possible)")
    ax.set_ylim(0, max(mode_counts.values)*1.2)
    save("failure_modes.png")

    names = [model_labels[key] for key in metrics["validation"]]
    aps = [value["average_precision"] for value in metrics["validation"].values()]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, aps, color=["#a6a8ad", "#78a8b6", "#277d8d"])
    for bar, value in zip(bars, aps):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012, f"{value:.3f}", ha="center")
    ax.set(ylabel="Validation average precision", title="Model selection on validation set", ylim=(0, 0.9))
    save("validation_comparison.png")

    probabilities = bundle["model"].predict_proba(validation_x)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    f2 = [fbeta_score(validation_y, probabilities >= value, beta=2, zero_division=0) for value in thresholds]
    if abs(thresholds[int(np.argmax(f2))] - bundle["threshold"]) > 1e-10:
        raise ValueError("Saved alert threshold differs from validation optimum")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, f2, color="#277d8d", linewidth=2)
    ax.axvline(bundle["threshold"], color="#d97c61", linestyle="--", label=f"Chosen threshold = {bundle['threshold']:.2f}")
    ax.set(xlabel="Probability threshold", ylabel="Validation F2", title="Threshold selection on validation set")
    ax.legend()
    save("threshold_f2.png")

    preprocess = bundle["model"].named_steps["preprocess"]
    estimator = bundle["model"].named_steps["model"]
    importances = np.asarray(estimator.feature_importances_)
    feature_names = preprocess.get_feature_names_out()
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.barh(np.asarray(feature_names)[order], importances[order], color="#6e9eb0")
    ax.set(xlabel="XGBoost feature importance", title="Model-specific feature importance (not causal)")
    save("feature_importance.png")

    architecture()
    summary = {
        "dataset_sha256": digest,
        "rows": len(frame),
        "columns": list(frame.columns),
        "missing_values": int(frame.isna().sum().sum()),
        "failures": int(target.sum()),
        "type_counts": {key: int(value) for key, value in frame["Type"].value_counts().items()},
        "seed": SEED,
    }
    (GENERATED / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    print("Generated report figures and verified table rows")


if __name__ == "__main__":
    generate()
