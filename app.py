"""
=============================================================
  AI for Interpretable Heart Disease Prediction
  Flask Deployment — Loads Pre-trained Models
=============================================================

Architecture:
  DEVELOPMENT (Feature_Selection_Pipeline.py):
    Dataset → Cleaning → RFE / Boruta → Train Models → Save

  DEPLOYMENT (this file):
    User → Flask → Load Saved Models → Prediction → SHAP → LIME → PDF

Key change from previous version:
  Before: trained all 3 models from scratch at every startup (~2 min)
  Now:    loads pre-saved models instantly (~3 seconds)
  
  The feature list is also loaded from metadata.pkl so Flask
  automatically uses whatever feature subset won (All / RFE / Boruta).
"""

import os
import io
import uuid
import pickle
import base64
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import (Flask, render_template, request,
                   jsonify, send_file, abort)

import shap
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage,
                                 HRFlowable)
from reportlab.lib.enums import TA_CENTER

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["REPORT_FOLDER"] = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Put Book1.csv in the same folder as app.py
DATA_PATH  = "heart_disease_combined.csv"
MODEL_DIR  = "saved_models"

# Full 13 feature labels (for UI display regardless of subset used)
FEATURE_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex (1=Male, 0=Female)",
    "cp":       "Chest Pain Type (0–3)",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol":     "Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar > 120 (1=Yes)",
    "restecg":  "Resting ECG (0–2)",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise Induced Angina (1=Yes)",
    "oldpeak":  "ST Depression (oldpeak)",
    "slope":    "Slope of ST Segment (0–2)",
    "ca":       "Coronary Arteries Coloured (0–4)",
    "thal":     "Thalassemia (0–3)",
}

# Hints for each feature (shown in the input form)
FEATURE_HINTS = {
    "age":      "Patient age in years",
    "sex":      "1 = Male, 0 = Female",
    "cp":       "0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic",
    "trestbps": "mmHg on admission",
    "chol":     "Serum cholesterol in mg/dl",
    "fbs":      "1 if fasting blood sugar > 120 mg/dl",
    "restecg":  "0=Normal, 1=ST-T abnormality, 2=LV hypertrophy",
    "thalach":  "Maximum heart rate achieved",
    "exang":    "1 = Yes, 0 = No",
    "oldpeak":  "ST depression induced by exercise relative to rest",
    "slope":    "0=Upsloping, 1=Flat, 2=Downsloping",
    "ca":       "Number of major vessels coloured by fluoroscopy (0–4)",
    "thal":     "0=Normal, 1=Fixed defect, 2=Reversible defect, 3=Other",
}

TARGET_CANDIDATES = ["target", "condition", "num", "heart_disease", "output",
                     "diagnosis", "label", "class", "disease"]

# Global model store
MODELS = {}


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_saved_models():
    """
    Load pre-trained models saved by Feature_Selection_Pipeline.py.
    Falls back to training from scratch if saved models not found.
    """
    meta_path = os.path.join(MODEL_DIR, "metadata.pkl")

    if not os.path.exists(meta_path):
        print(f"⚠  No saved models found in '{MODEL_DIR}/'")
        print("   Run Feature_Selection_Pipeline.py first to train and save models.")
        print("   Falling back to training from scratch...")
        train_from_scratch()
        return

    print(f"✓ Found saved models in '{MODEL_DIR}/' — loading...")

    # Load metadata (tells us which features the best model used)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    feature_names = metadata["feature_names"]
    feature_set   = metadata["feature_set"]
    print(f"  Feature set : '{feature_set}' ({len(feature_names)} features)")
    print(f"  Features    : {feature_names}")
    print(f"  Saved AUC   : {metadata['ensemble_auc']:.4f}")
    print(f"  Saved Acc   : {metadata['ensemble_acc']*100:.2f}%")

    # Load RF
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
        rf = pickle.load(f)

    # Load SVM
    with open(os.path.join(MODEL_DIR, "svm_model.pkl"), "rb") as f:
        svm = pickle.load(f)

    # Load ANN
    ann = keras.models.load_model(
        os.path.join(MODEL_DIR, "ann_model.keras")
    )

    # Load scaler
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # Build SHAP explainer on a background sample
    # We need training data for this — load and prepare a sample
    df = _load_data_for_explainers(feature_names)
    X_bg = scaler.transform(df[feature_names].values)
    bg_idx = np.random.choice(X_bg.shape[0],
                               size=min(100, X_bg.shape[0]),
                               replace=False)
    shap_explainer = shap.DeepExplainer(ann, X_bg[bg_idx])

    # Build LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_bg,
        feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE
    )

    MODELS.update({
        "rf": rf, "svm": svm, "ann": ann,
        "scaler": scaler,
        "feature_names":   feature_names,
        "feature_set":     feature_set,
        "shap_explainer":  shap_explainer,
        "lime_explainer":  lime_explainer,
        "metadata":        metadata,
    })

    print(f"✓ All models loaded. Flask is ready.")


def _load_data_for_explainers(feature_names):
    """Load and clean CSV — used to build SHAP/LIME background."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\n❌ CSV not found at: {os.path.abspath(DATA_PATH)}\n"
            f"   Place Book1.csv in the same folder as app.py.\n"
        )
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    encoding = "latin-1"
    for enc in encodings:
        try:
            with open(DATA_PATH, "r", encoding=enc) as f:
                f.read()
            encoding = enc
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    # Find header row
    header_row = 0
    with open(DATA_PATH, "r", encoding=encoding, errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            cells = [c.strip().lower() for c in line.split(",")]
            if "age" in cells or "target" in cells:
                header_row = i
                break

    df = pd.read_csv(DATA_PATH, header=header_row, encoding=encoding)
    df.columns = df.columns.str.strip()

    col_map = {c.lower(): c for c in df.columns}
    target_col = None
    for c in TARGET_CANDIDATES:
        if c in col_map:
            target_col = col_map[c]
            break
    if target_col is None:
        target_col = df.columns[-1]
    df = df.rename(columns={target_col: "target"})
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    for col in df.columns:
        if col != "target":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates().dropna()
    return df


def train_from_scratch():
    """
    Fallback: train models from scratch if no saved models found.
    Mirrors the finalized model configs.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from tensorflow.keras import layers, callbacks

    ALL_FEATURES = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal"
    ]

    print("⏳ Training models from scratch (no saved models found)...")
    df      = _load_data_for_explainers(ALL_FEATURES)
    X       = df[ALL_FEATURES].values
    y       = df["target"].values
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(
        n_estimators=200, min_samples_split=5, min_samples_leaf=2,
        max_features="sqrt", bootstrap=True, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                    class_weight="balanced", probability=True,
                    random_state=RANDOM_STATE))
    ])
    svm.fit(X_train, y_train)

    ann = keras.Sequential([
        layers.Input(shape=(len(ALL_FEATURES),)),
        layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_initializer="he_normal"),
        layers.Dense(1, activation="sigmoid")
    ])
    ann.compile(optimizer=keras.optimizers.Adam(0.001),
                loss="binary_crossentropy",
                metrics=["accuracy", keras.metrics.AUC(name="auc")])
    ann.fit(X_train_sc, y_train, epochs=200, batch_size=32,
            validation_split=0.15, verbose=0,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                        restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                            patience=10, min_lr=1e-6)
            ])

    bg_idx = np.random.choice(X_train_sc.shape[0], size=100, replace=False)
    shap_explainer = shap.DeepExplainer(ann, X_train_sc[bg_idx])
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_sc, feature_names=ALL_FEATURES,
        class_names=["No Disease", "Disease"],
        mode="classification", discretize_continuous=True,
        random_state=RANDOM_STATE
    )

    MODELS.update({
        "rf": rf, "svm": svm, "ann": ann,
        "scaler": scaler,
        "feature_names":  ALL_FEATURES,
        "feature_set":    "All Features (fallback)",
        "shap_explainer": shap_explainer,
        "lime_explainer": lime_explainer,
        "metadata":       {"feature_names": ALL_FEATURES,
                           "feature_set": "All Features (fallback)"},
    })
    print("✓ Fallback training complete. Flask is ready.")


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION + EXPLANATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(patient_values_full: dict) -> dict:
    """
    patient_values_full: dict of ALL 13 feature values from the form.
    We extract only the features the best model needs.
    """
    feature_names = MODELS["feature_names"]
    rf   = MODELS["rf"]
    svm  = MODELS["svm"]
    ann  = MODELS["ann"]
    sc   = MODELS["scaler"]
    shap_exp  = MODELS["shap_explainer"]
    lime_exp  = MODELS["lime_explainer"]

    # Extract only the features this model was trained on
    patient_values = [float(patient_values_full[f]) for f in feature_names]

    x_raw    = np.array(patient_values, dtype=float).reshape(1, -1)
    x_scaled = sc.transform(x_raw)

    # ── Soft Voting ───────────────────────────────────────────────────────
    p_rf  = float(rf.predict_proba(x_raw)[0, 1])
    p_svm = float(svm.predict_proba(x_raw)[0, 1])
    p_ann = float(ann.predict(x_scaled, verbose=0)[0][0])
    p_ens = (p_rf + p_svm + p_ann) / 3.0

    prediction = int(p_ens >= 0.5)
    outcome    = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    risk_level = (
        "High Risk"     if p_ens >= 0.70 else
        "Moderate Risk" if p_ens >= 0.45 else
        "Low Risk"
    )

    # ── SHAP ─────────────────────────────────────────────────────────────
    shap_vals_raw = shap_exp.shap_values(x_scaled)
    if isinstance(shap_vals_raw, list):
        shap_vals_raw = shap_vals_raw[0]
    if shap_vals_raw.ndim == 3:
        shap_vals_raw = shap_vals_raw[:, :, 0]
    shap_vals = shap_vals_raw[0]

    try:
        raw_base = shap_exp.expected_value
        if hasattr(raw_base, "numpy"):
            raw_base = raw_base.numpy()
        base_value = float(np.array(raw_base).flatten()[0])
    except Exception:
        base_value = p_ann

    shap_bar_img       = _plot_shap_bar(shap_vals, feature_names, patient_values)
    shap_waterfall_img = _plot_shap_waterfall(shap_vals, feature_names,
                                               patient_values)

    # ── LIME ─────────────────────────────────────────────────────────────
    def predict_proba_fn(X):
        p = ann.predict(X, verbose=0).flatten()
        return np.column_stack([1 - p, p])

    lime_result = lime_exp.explain_instance(
        data_row=x_scaled[0],
        predict_fn=predict_proba_fn,
        num_features=len(feature_names),
        num_samples=3000,
        labels=(1,)
    )
    lime_img = _plot_lime_bar(lime_result, feature_names)

    # ── Feature table (SHAP ranked) ───────────────────────────────────────
    original = sc.inverse_transform(x_scaled)[0]
    feature_table = []
    for i, fname in enumerate(feature_names):
        feature_table.append({
            "feature":   fname,
            "label":     FEATURE_LABELS.get(fname, fname),
            "value":     round(float(original[i]), 2),
            "shap":      round(float(shap_vals[i]), 4),
            "direction": "↑ Risk" if shap_vals[i] > 0 else "↓ Risk"
        })
    feature_table.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "p_rf":    round(p_rf * 100, 1),
        "p_svm":   round(p_svm * 100, 1),
        "p_ann":   round(p_ann * 100, 1),
        "p_ens":   round(p_ens * 100, 1),
        "prediction":  prediction,
        "outcome":     outcome,
        "risk_level":  risk_level,
        "feature_set": MODELS["feature_set"],
        "n_features":  len(feature_names),
        "shap_bar_img":       shap_bar_img,
        "shap_waterfall_img": shap_waterfall_img,
        "lime_img":           lime_img,
        "feature_table":      feature_table,
        "patient_values":     patient_values_full,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _plot_shap_bar(shap_vals, feature_names, patient_values) -> str:
    idx_sorted = np.argsort(np.abs(shap_vals))
    feats  = [feature_names[i] for i in idx_sorted]
    vals   = [shap_vals[i]     for i in idx_sorted]
    colors_bar = ["#e74c3c" if v > 0 else "#2980b9" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feats)*0.5)),
                            facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.barh(feats, vals, color=colors_bar, edgecolor="none", alpha=0.9)
    ax.axvline(0, color="#8b949e", lw=1)
    ax.set_xlabel("SHAP Value", color="#c9d1d9", fontsize=10)
    ax.set_title("SHAP Feature Contributions", color="#f0f6fc",
                 fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="#c9d1d9", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(axis="x", color="#30363d", alpha=0.5, linestyle="--")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_shap_waterfall(shap_vals, feature_names, patient_values) -> str:
    n_show = min(8, len(feature_names))
    idx    = np.argsort(np.abs(shap_vals))[::-1][:n_show]
    feats  = [f"{feature_names[i]}={patient_values[i]:.1f}" for i in idx]
    vals   = [shap_vals[i] for i in idx]
    colors_bar = ["#e74c3c" if v > 0 else "#2980b9" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.barh(feats[::-1], vals[::-1], color=colors_bar[::-1],
            edgecolor="none", alpha=0.9)
    ax.axvline(0, color="#8b949e", lw=1)
    ax.set_xlabel("SHAP Value", color="#c9d1d9", fontsize=9)
    ax.set_title("Top Feature Drivers (Waterfall)", color="#f0f6fc",
                 fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors="#c9d1d9", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(axis="x", color="#30363d", alpha=0.5, linestyle="--")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_lime_bar(lime_result, feature_names) -> str:
    feat_weights = lime_result.as_list(label=1)
    feat_weights.sort(key=lambda x: x[1])
    feats   = [f[0] for f in feat_weights]
    weights = [f[1] for f in feat_weights]
    colors_bar = ["#e74c3c" if w > 0 else "#2980b9" for w in weights]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feats)*0.5)),
                            facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.barh(feats, weights, color=colors_bar, edgecolor="none", alpha=0.9)
    ax.axvline(0, color="#8b949e", lw=1)
    ax.set_xlabel("LIME Weight", color="#c9d1d9", fontsize=10)
    ax.set_title("LIME Local Explanation", color="#f0f6fc",
                 fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="#c9d1d9", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(axis="x", color="#30363d", alpha=0.5, linestyle="--")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf_report(result: dict, report_id: str) -> str:
    path   = os.path.join(app.config["REPORT_FOLDER"], f"{report_id}.pdf")
    doc    = SimpleDocTemplate(path, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    title_s = ParagraphStyle("T", parent=styles["Heading1"], fontSize=17,
                              textColor=colors.HexColor("#1a1a2e"),
                              spaceAfter=4, alignment=TA_CENTER)
    sub_s   = ParagraphStyle("S", parent=styles["Normal"], fontSize=9,
                              textColor=colors.HexColor("#666666"),
                              spaceAfter=2, alignment=TA_CENTER)
    sec_s   = ParagraphStyle("Sec", parent=styles["Heading2"], fontSize=12,
                              textColor=colors.HexColor("#16213e"),
                              spaceBefore=12, spaceAfter=5)
    body_s  = ParagraphStyle("B", parent=styles["Normal"], fontSize=9,
                              textColor=colors.HexColor("#333333"), spaceAfter=4)
    risk_c  = "#c0392b" if result["prediction"] == 1 else "#27ae60"

    story = []

    # Header
    story.append(Paragraph("AI for Interpretable Heart Disease Prediction", title_s))
    story.append(Paragraph("Clinical Prediction Report", sub_s))
    story.append(Paragraph(
        f"Generated: {result['timestamp']}  |  Report ID: {report_id}  |  "
        f"Feature Set: {result['feature_set']} ({result['n_features']} features)",
        sub_s))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#16213e"), spaceAfter=10))

    # Prediction summary
    story.append(Paragraph("Prediction Summary", sec_s))
    out_s = ParagraphStyle("Out", parent=styles["Normal"], fontSize=15,
                            fontName="Helvetica-Bold",
                            textColor=colors.HexColor(risk_c),
                            spaceAfter=8, alignment=TA_CENTER)
    story.append(Paragraph(
        f"▶  {result['outcome']}  —  {result['risk_level']}", out_s))

    pred_data = [
        ["Model", "P(Disease)", "Weight"],
        ["Random Forest",  f"{result['p_rf']}%",  "33%"],
        ["SVM (RBF)",      f"{result['p_svm']}%", "33%"],
        ["ANN (MLP)",      f"{result['p_ann']}%", "33%"],
        ["Ensemble (Avg)", f"{result['p_ens']}%", "Final"],
    ]
    pt = Table(pred_data, colWidths=[6*cm, 5*cm, 5*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0),  colors.HexColor("#16213e")),
        ("TEXTCOLOR",      (0,0),(-1,0),  colors.white),
        ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,-1), 9),
        ("ALIGN",          (0,0),(-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1),(-1,-2), [colors.HexColor("#f8f9fa"),
                                            colors.white]),
        ("BACKGROUND",     (0,-1),(-1,-1), colors.HexColor("#ffeeba")),
        ("FONTNAME",       (0,-1),(-1,-1), "Helvetica-Bold"),
        ("GRID",           (0,0),(-1,-1), 0.5, colors.HexColor("#dee2e6")),
        ("ROWHEIGHT",      (0,0),(-1,-1), 20),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.3*cm))

    # Feature table
    story.append(Paragraph("SHAP Clinical Feature Analysis", sec_s))
    feat_data = [["Feature", "Value", "SHAP", "Direction"]]
    for row in result["feature_table"]:
        feat_data.append([
            row["label"], str(row["value"]),
            f"{row['shap']:+.4f}", row["direction"]
        ])
    ft = Table(feat_data, colWidths=[7*cm, 2.5*cm, 3*cm, 3.5*cm])
    ft.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0),  colors.HexColor("#16213e")),
        ("TEXTCOLOR",      (0,0),(-1,0),  colors.white),
        ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,-1), 8),
        ("ALIGN",          (1,0),(-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.HexColor("#f8f9fa"),
                                            colors.white]),
        ("GRID",           (0,0),(-1,-1), 0.4, colors.HexColor("#dee2e6")),
        ("ROWHEIGHT",      (0,0),(-1,-1), 16),
    ]))
    story.append(ft)
    story.append(Spacer(1, 0.3*cm))

    # SHAP plots
    story.append(Paragraph("SHAP Explainability", sec_s))
    story.append(Paragraph(
        "Red bars push prediction toward disease. "
        "Blue bars push away from disease.", body_s))
    story.append(RLImage(io.BytesIO(base64.b64decode(result["shap_bar_img"])),
                          width=14*cm, height=8*cm))
    story.append(Spacer(1, 0.2*cm))
    story.append(RLImage(io.BytesIO(base64.b64decode(result["shap_waterfall_img"])),
                          width=14*cm, height=6*cm))

    # LIME plot
    story.append(Paragraph("LIME Local Explanation", sec_s))
    story.append(Paragraph(
        "Locally fitted linear approximation around this patient's "
        "neighbourhood. Positive weights push toward disease.", body_s))
    story.append(RLImage(io.BytesIO(base64.b64decode(result["lime_img"])),
                          width=14*cm, height=8*cm))

    # Disclaimer
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#dee2e6")))
    disc_s = ParagraphStyle("D", parent=styles["Normal"], fontSize=7,
                             textColor=colors.HexColor("#999999"))
    story.append(Paragraph(
        "⚠ DISCLAIMER: This report is generated by an AI research system "
        "for educational and research purposes only. It is NOT a substitute "
        "for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified physician.", disc_s))

    doc.build(story)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════
ALL_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


@app.route("/")
def index():
    active_features = MODELS.get("feature_names", ALL_FEATURES)
    return render_template(
        "index.html",
        all_features=ALL_FEATURES,
        active_features=active_features,
        feature_labels=FEATURE_LABELS,
        feature_hints=FEATURE_HINTS,
        feature_set=MODELS.get("feature_set", "All Features"),
        n_features=len(active_features)
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect all 13 values from the form
        patient_values_full = {
            f: float(request.form.get(f, 0)) for f in ALL_FEATURES
        }
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    result    = run_pipeline(patient_values_full)
    report_id = str(uuid.uuid4())[:8]
    generate_pdf_report(result, report_id)
    result["report_id"] = report_id

    return render_template("result.html", result=result,
                           feature_labels=FEATURE_LABELS)


@app.route("/report/<report_id>")
def download_report(report_id):
    safe_id = "".join(c for c in report_id if c.isalnum() or c == "-")
    path    = os.path.join(app.config["REPORT_FOLDER"], f"{safe_id}.pdf")
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True,
                     download_name=f"heart_disease_report_{safe_id}.pdf",
                     mimetype="application/pdf")


@app.route("/model-info")
def model_info():
    """Simple JSON endpoint to check which models are loaded."""
    meta = MODELS.get("metadata", {})
    return jsonify({
        "status":       "ready",
        "feature_set":  MODELS.get("feature_set", "unknown"),
        "n_features":   len(MODELS.get("feature_names", [])),
        "features":     MODELS.get("feature_names", []),
        "saved_auc":    meta.get("ensemble_auc", "N/A"),
        "saved_acc":    meta.get("ensemble_acc", "N/A"),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("  Heart Disease Prediction — Flask App")
    print("="*60)
    load_saved_models()
    app.run(debug=False, host="0.0.0.0", port=5000)