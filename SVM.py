"""
=============================================================
  AI for Interpretable Heart Disease Prediction
  Support Vector Machine (SVM) Model
=============================================================

Dataset expected: Book1.csv
  - 13 clinical features (Cleveland Heart Disease dataset format)
  - Binary target column named 'target' (0 = No Disease, 1 = Disease)

Key difference from Random Forest:
  SVM finds the optimal hyperplane that best separates the two
  classes (disease / no disease) with the maximum margin.
  It REQUIRES feature scaling — raw values like cholesterol (200+)
  would dominate over binary features like sex (0/1) otherwise.

Feature Reference:
  age, sex, cp (chest pain type), trestbps (resting BP),
  chol (cholesterol), fbs (fasting blood sugar), restecg,
  thalach (max heart rate), exang (exercise-induced angina),
  oldpeak (ST depression), slope, ca (vessels colored), thal
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = "/kaggle/input/datasets/saeedulhussain/heart-1/Book1.csv"

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

TARGET_CANDIDATES = ["target", "condition", "num", "heart_disease", "output",
                     "diagnosis", "label", "class", "disease"]


# ── 1. DATA LOADING ───────────────────────────────────────────────────────────
def _find_header_row(filepath: str) -> int:
    """Scan first 10 rows to find the real header (handles title rows)."""
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            cells = [c.strip().lower() for c in line.split(",")]
            if "age" in cells or "target" in cells:
                return i
    return 0


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV, skip title rows, auto-detect target column, and clean."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    header_row = _find_header_row(filepath)
    if header_row > 0:
        print(f"ℹ  Detected title row(s); real headers at row {header_row}.")
    df = pd.read_csv(filepath, header=header_row)
    df.columns = df.columns.str.strip()

    # Auto-detect target column
    col_map = {c.lower(): c for c in df.columns}
    target_col = None
    for candidate in TARGET_CANDIDATES:
        if candidate in col_map:
            target_col = col_map[candidate]
            break
    if target_col is None:
        target_col = df.columns[-1]
        print(f"⚠  Falling back to last column as target: '{target_col}'")

    df = df.rename(columns={target_col: "target"})
    print(f"✓ Using '{target_col}' as target column.")

    # Force numeric, binarise target, drop bad rows
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    if df["target"].nunique(dropna=True) > 2:
        df["target"] = (df["target"] > 0).astype(int)
    for col in df.columns:
        if col != "target":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.drop_duplicates().dropna()
    dropped = before - len(df)
    if dropped:
        print(f"ℹ  Dropped {dropped} rows (duplicates / non-numeric values).")

    print(f"✓ Loaded {df.shape[0]} samples | {df.shape[1]-1} features | "
          f"Disease prevalence: {df['target'].mean()*100:.1f}%")
    return df


# ── 2. EXPLORATORY SUMMARY ────────────────────────────────────────────────────
def explore(df: pd.DataFrame) -> None:
    print("\n── Data Overview ──")
    print(df.describe().round(2).to_string())
    print("\nMissing values:", df.isnull().sum().sum())
    print("Class distribution:\n", df["target"].value_counts())


# ── 3. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    X = df[FEATURE_NAMES]
    y = df["target"]
    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )


# ── 4. MODEL — SVM inside a scaling Pipeline ──────────────────────────────────
def build_model() -> Pipeline:
    """
    SVM wrapped in a Pipeline with StandardScaler.

    WHY SCALING IS MANDATORY FOR SVM:
      SVM computes distances between data points to find the margin.
      If 'chol' ranges 100–600 and 'sex' is 0 or 1, SVM will treat
      cholesterol as ~500x more important purely due to scale — not
      because it actually is. StandardScaler fixes this by converting
      every feature to mean=0, std=1.

    KERNEL CHOICE — RBF (Radial Basis Function):
      • Linear kernel: draws a straight line between classes (fast,
        interpretable via coefficients, but limited to linearly
        separable data).
      • RBF kernel: maps data into higher dimensions to find a curved
        boundary — handles non-linear patterns in heart disease data
        much better in practice.
      We use RBF here for accuracy, and use permutation importance
      for interpretability (since RBF has no direct coefficients).

    KEY PARAMETERS:
      C     → regularisation: high C = fits training data tightly
              (risk of overfitting); low C = wider margin, more errors
              allowed (more generalisation). C=1 is a safe default.
      gamma → controls how far each training point's influence reaches.
              'scale' = 1/(n_features * X.var()) — sklearn's smart default.
      class_weight='balanced' → compensates for class imbalance
              automatically, same as in the RF model.
      probability=True → enables predict_proba() for ROC-AUC scoring.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,       # needed for ROC-AUC & predict_proba
            random_state=RANDOM_STATE,
        ))
    ])


# ── 5. EVALUATION ─────────────────────────────────────────────────────────────
def evaluate(model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv  = cross_val_score(
        model, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", n_jobs=-1
    )

    print(f"\n── Model Performance ──")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Test ROC-AUC  : {auc:.4f}")
    print(f"  5-Fold CV AUC : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"\nClassification Report:\n"
          f"{classification_report(y_test, y_pred, target_names=['No Disease','Disease'])}")

    return dict(y_pred=y_pred, y_proba=y_proba, acc=acc, auc=auc, cv=cv)


# ── 6. INTERPRETABILITY ───────────────────────────────────────────────────────
def permutation_importance_df(model, X_test, y_test) -> pd.DataFrame:
    """
    Permutation Importance — the RIGHT way to interpret SVM.

    Unlike Random Forest, SVM (with RBF kernel) has no built-in
    feature importance scores. Instead, we shuffle each feature one
    at a time and measure how much the ROC-AUC drops. A big drop
    means that feature was critical to the model's decisions.
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=30, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    return (
        pd.DataFrame({
            "feature":    list(X_test.columns),
            "importance": result.importances_mean,
            "std":        result.importances_std
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def support_vector_summary(model, X_train) -> None:
    """Print a summary of the support vectors found by SVM."""
    svm_clf = model.named_steps["svm"]
    n_sv = svm_clf.n_support_          # [n_sv_class0, n_sv_class1]
    total_sv = sum(n_sv)
    ratio = total_sv / len(X_train) * 100
    print(f"\n── Support Vector Summary ──")
    print(f"  Total support vectors : {total_sv} / {len(X_train)} "
          f"training samples ({ratio:.1f}%)")
    print(f"  Class 0 (No Disease)  : {n_sv[0]} support vectors")
    print(f"  Class 1 (Disease)     : {n_sv[1]} support vectors")
    print(f"  (Fewer SVs = cleaner, more confident margin)")


# ── 7. PREDICTION INTERFACE ───────────────────────────────────────────────────
def predict_patient(model, patient_values: list, threshold: float = 0.5) -> None:
    """
    Predict heart disease risk for a single patient.
    patient_values: 13 values in FEATURE_NAMES order.
    """
    x_df = pd.DataFrame([patient_values], columns=FEATURE_NAMES)
    # Note: Pipeline automatically scales x_df before passing to SVM
    proba   = model.predict_proba(x_df)[0, 1]
    outcome = "⚠  HEART DISEASE DETECTED" if proba >= threshold else "✓  No Heart Disease"

    print(f"\n── Single Patient Prediction ──")
    for name, val in zip(FEATURE_NAMES, patient_values):
        print(f"  {name:12s}: {val}")
    print(f"\n  Disease Probability : {proba*100:.1f}%")
    print(f"  Prediction          : {outcome}")


# ── 8. VISUALISATIONS ─────────────────────────────────────────────────────────
def plot_all(model, results, X_test, y_test, pi_df) -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "AI for Interpretable Heart Disease Prediction\nSVM (RBF Kernel) Analysis",
        fontsize=16, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── (a) Confusion Matrix ──
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test, results["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax1,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax1.set_title("Confusion Matrix")
    ax1.set_ylabel("Actual"); ax1.set_xlabel("Predicted")

    # ── (b) ROC Curve ──
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, results["y_proba"])
    ax2.plot(fpr, tpr, lw=2, color="purple",
             label=f"SVM AUC = {results['auc']:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random Guess")
    ax2.set(title="ROC Curve", xlabel="False Positive Rate",
            ylabel="True Positive Rate")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # ── (c) CV Score Distribution ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(range(1, 6), results["cv"], color="purple", alpha=0.7, edgecolor="white")
    ax3.axhline(results["cv"].mean(), color="tomato", lw=2,
                linestyle="--", label=f"Mean = {results['cv'].mean():.3f}")
    ax3.set(title="5-Fold CV ROC-AUC", xlabel="Fold", ylabel="AUC",
            ylim=(0.5, 1.05), xticks=range(1, 6))
    ax3.legend()

    # ── (d) Permutation Importance (full) ──
    ax4 = fig.add_subplot(gs[1, 0:2])
    colors = ["#6c3483" if i == 0 else "mediumpurple" for i in range(len(pi_df))]
    ax4.barh(pi_df["feature"][::-1], pi_df["importance"][::-1],
             xerr=pi_df["std"][::-1], color=colors[::-1],
             align="center", capsize=3, alpha=0.85)
    ax4.set(title="Feature Importance — Permutation Method\n(ROC-AUC drop when feature is shuffled)",
            xlabel="Mean AUC Decrease")
    ax4.axvline(0, color="black", lw=0.8, linestyle="--")
    ax4.grid(axis="x", alpha=0.3)

    # ── (e) Probability Distribution by Class ──
    ax5 = fig.add_subplot(gs[1, 2])
    y_test_arr = np.array(y_test)
    proba_disease    = results["y_proba"][y_test_arr == 1]
    proba_no_disease = results["y_proba"][y_test_arr == 0]
    ax5.hist(proba_no_disease, bins=20, alpha=0.6, color="steelblue",
             label="No Disease", density=True)
    ax5.hist(proba_disease, bins=20, alpha=0.6, color="tomato",
             label="Disease", density=True)
    ax5.axvline(0.5, color="black", lw=1.5, linestyle="--", label="Threshold=0.5")
    ax5.set(title="Predicted Probability Distribution",
            xlabel="P(Heart Disease)", ylabel="Density")
    ax5.legend()

    plt.savefig("heart_disease_svm_report.png", dpi=150, bbox_inches="tight")
    print("\n✓ Report saved → heart_disease_svm_report.png")
    plt.show()


# ── 9. MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load
    df = load_data(DATA_PATH)
    explore(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Build & evaluate
    model = build_model()
    results = evaluate(model, X_train, X_test, y_train, y_test)

    # Support vector summary (unique to SVM)
    support_vector_summary(model, X_train)

    # Interpretability via permutation importance
    print("\n⏳ Computing permutation importance (this takes ~30 seconds)...")
    pi_df = permutation_importance_df(model, X_test, y_test)

    print("\n── Top-5 Most Important Features ──")
    print(pi_df.head(5).to_string(index=False))

    # Single-patient demo (63-year-old male, typical angina)
    sample_patient = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
    predict_patient(model, sample_patient)

    # Plots
    plot_all(model, results, X_test, y_test, pi_df)
