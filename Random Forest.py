"""
=============================================================
  AI for Interpretable Heart Disease Prediction
  Random Forest Model — Improved & Production-Ready
=============================================================

Dataset expected: Book1.csv
  - 13 clinical features (Cleveland Heart Disease dataset format)
  - Binary target column named 'target' (0 = No Disease, 1 = Disease)

Feature Reference:
  age, sex, cp (chest pain type), trestbps (resting BP),
  chol (cholesterol), fbs (fasting blood sugar), restecg,
  thalach (max heart rate), exang (exercise-induced angina),
  oldpeak (ST depression), slope, ca (vessels colored), thal
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ── 1. DATA LOADING & VALIDATION ─────────────────────────────────────────────
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

DATA_PATH = "/kaggle/input/datasets/saeedulhussain/heart-1/Book1.csv"

# Known target-column name variants (case-insensitive)
TARGET_CANDIDATES = ["target", "condition", "num", "heart_disease", "output",
                     "diagnosis", "label", "class", "disease"]

def _find_header_row(filepath: str) -> int:
    """
    Scan the first 10 rows to find which row contains the real column headers.
    Looks for a row where 'age' or 'target' appears (case-insensitive).
    Returns 0 if no title row detected (normal CSV).
    """
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            cells = [c.strip().lower() for c in line.split(",")]
            if "age" in cells or "target" in cells:
                return i
    return 0


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV, skip any title rows, auto-detect target column, and clean."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    # ── Skip decorative title rows ────────────────────────────────────────
    header_row = _find_header_row(filepath)
    if header_row > 0:
        print(f"ℹ  Detected title row(s); real headers at row {header_row}.")
    df = pd.read_csv(filepath, header=header_row)
    df.columns = df.columns.str.strip()

    # ── Auto-detect target column ─────────────────────────────────────────
    col_map = {c.lower(): c for c in df.columns}
    target_col = None
    for candidate in TARGET_CANDIDATES:
        if candidate in col_map:
            target_col = col_map[candidate]
            break

    if target_col is None:
        target_col = df.columns[-1]
        print(f"⚠  No known target column found. Falling back to: '{target_col}'")

    df = df.rename(columns={target_col: "target"})
    print(f"✓ Using '{target_col}' as target column.")

    # ── Force numeric types (handles stray strings after title-row shift) ─
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    if df["target"].nunique(dropna=True) > 2:
        df["target"] = (df["target"] > 0).astype(int)

    for col in df.columns:
        if col != "target":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Drop duplicates & rows with any NaN ──────────────────────────────
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


# ── 4. MODEL — sklearn RandomForestClassifier ─────────────────────────────────
def build_model() -> RandomForestClassifier:
    """
    Uses sklearn's production-grade Random Forest instead of manual bagging.
    Key improvements over the original script:
      • bootstrap=True  → row sampling with replacement (same idea as before)
      • max_features='sqrt' → standard column sub-sampling per split
      • class_weight='balanced' → handles class imbalance automatically
      • n_estimators=200 → much more stable than 3 trees
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,            # trees grow fully; pruned via min_samples
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",       # √13 ≈ 3-4 features per split
        bootstrap=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ── 5. EVALUATION ─────────────────────────────────────────────────────────────
def evaluate(model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_proba)
    cv    = cross_val_score(
        model, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", n_jobs=-1
    )

    print(f"\n── Model Performance ──")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Test ROC-AUC  : {auc:.4f}")
    print(f"  5-Fold CV AUC : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['No Disease','Disease'])}")

    return dict(y_pred=y_pred, y_proba=y_proba, acc=acc, auc=auc, cv=cv)


# ── 6. INTERPRETABILITY ───────────────────────────────────────────────────────
def feature_importance_df(model, feature_names) -> pd.DataFrame:
    """Mean-decrease impurity importance (built-in) + std across trees."""
    importances = model.feature_importances_
    stds = np.std(
        [tree.feature_importances_ for tree in model.estimators_], axis=0
    )
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances, "std": stds})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance_df(model, X_test, y_test) -> pd.DataFrame:
    """
    Permutation importance: shuffle each feature and measure accuracy drop.
    More reliable than MDI for correlated / high-cardinality features.
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=30, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    return (
        pd.DataFrame({
            "feature": X_test.columns,
            "importance": result.importances_mean,
            "std": result.importances_std
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ── 7. PREDICTION INTERFACE ───────────────────────────────────────────────────
def predict_patient(model, patient_values: list, threshold: float = 0.5) -> None:
    """
    Make an interpretable prediction for a single patient.

    patient_values: list of 13 values in FEATURE_NAMES order.
    """
    x = np.array(patient_values).reshape(1, -1)
    x_df = pd.DataFrame(x, columns=FEATURE_NAMES)

    proba   = model.predict_proba(x_df)[0, 1]
    outcome = "⚠  HEART DISEASE DETECTED" if proba >= threshold else "✓  No Heart Disease"

    print(f"\n── Single Patient Prediction ──")
    for name, val in zip(FEATURE_NAMES, patient_values):
        print(f"  {name:12s}: {val}")
    print(f"\n  Disease Probability : {proba*100:.1f}%")
    print(f"  Prediction          : {outcome}")


# ── 8. VISUALISATIONS ─────────────────────────────────────────────────────────
def plot_all(model, results, X_test, y_test, fi_df, pi_df) -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "AI for Interpretable Heart Disease Prediction\nRandom Forest Analysis",
        fontsize=16, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── (a) Confusion Matrix ──
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test, results["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax1.set_title("Confusion Matrix")
    ax1.set_ylabel("Actual"); ax1.set_xlabel("Predicted")

    # ── (b) ROC Curve ──
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, results["y_proba"])
    ax2.plot(fpr, tpr, lw=2, color="steelblue",
             label=f"AUC = {results['auc']:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set(title="ROC Curve", xlabel="False Positive Rate",
            ylabel="True Positive Rate")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # ── (c) CV Score Distribution ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(range(1, 6), results["cv"], color="steelblue", alpha=0.7, edgecolor="white")
    ax3.axhline(results["cv"].mean(), color="tomato", lw=2,
                linestyle="--", label=f"Mean={results['cv'].mean():.3f}")
    ax3.set(title="5-Fold CV ROC-AUC", xlabel="Fold", ylabel="AUC",
            ylim=(0.5, 1.05), xticks=range(1, 6))
    ax3.legend()

    # ── (d) MDI Feature Importance ──
    ax4 = fig.add_subplot(gs[1, 0:2])
    colors = ["#c0392b" if i == 0 else "steelblue" for i in range(len(fi_df))]
    ax4.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
             xerr=fi_df["std"][::-1], color=colors[::-1],
             align="center", capsize=3)
    ax4.set(title="Feature Importance (Mean Decrease Impurity)",
            xlabel="Importance Score")
    ax4.grid(axis="x", alpha=0.3)

    # ── (e) Permutation Importance ──
    ax5 = fig.add_subplot(gs[1, 2])
    top_pi = pi_df.head(10)
    ax5.barh(top_pi["feature"][::-1], top_pi["importance"][::-1],
             xerr=top_pi["std"][::-1], color="darkorange",
             align="center", capsize=3, alpha=0.8)
    ax5.set(title="Permutation Importance\n(Top 10, ROC-AUC Drop)",
            xlabel="Mean AUC Decrease")
    ax5.grid(axis="x", alpha=0.3)

    plt.savefig("heart_disease_rf_report.png", dpi=150, bbox_inches="tight")
    print("\n✓ Report saved → heart_disease_rf_report.png")
    plt.show()


# ── 9. MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load  (path is set via DATA_PATH above)
    df = load_data(DATA_PATH)
    explore(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Train & evaluate
    model = build_model()
    results = evaluate(model, X_train, X_test, y_train, y_test)

    # Interpretability
    fi_df = feature_importance_df(model, FEATURE_NAMES)
    pi_df = permutation_importance_df(model, X_test, y_test)

    print("\n── Top-5 Features (MDI) ──")
    print(fi_df.head(5).to_string(index=False))

    print("\n── Top-5 Features (Permutation) ──")
    print(pi_df.head(5).to_string(index=False))

    # Single-patient demo (Cleveland sample: 63-year-old male, typical angina)
    sample_patient = [63,1,3,145,233,1,0,150,0,2.3,0,0,1]
    predict_patient(model, sample_patient)

    # Plots
    plot_all(model, results, X_test, y_test, fi_df, pi_df)
