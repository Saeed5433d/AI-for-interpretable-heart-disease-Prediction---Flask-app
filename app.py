"""
=============================================================
  AI for Interpretable Heart Disease Prediction
  Flask Web Application
=============================================================

Routes:
  GET  /              → Patient input form
  POST /predict       → Run ensemble prediction + SHAP + LIME
  GET  /report/<id>   → Download PDF clinical report
"""

import os
import io
import uuid
import base64
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required for Flask
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from flask import (Flask, render_template, request,
                   jsonify, send_file, abort)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap
import lime
import lime.lime_tabular

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage,
                                 HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["REPORT_FOLDER"] = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ── IMPORTANT: Update this to YOUR local CSV path ──
# Put Book1.csv in the same folder as app.py, or give the full path
DATA_PATH = "Book1.csv"

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

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

TARGET_CANDIDATES = ["target", "condition", "num", "heart_disease",
                     "output", "diagnosis", "label", "class", "disease"]

# ── Global model store (loaded once at startup) ───────────────────────────────
MODELS = {}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _detect_encoding(filepath):
    """
    Try common encodings in order. Many CSVs exported from Excel on
    Windows use 'cp1252' or 'latin-1' instead of pure UTF-8, which
    causes UnicodeDecodeError on bytes like 0x96 (en-dash character).
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings_to_try:
        try:
            with open(filepath, "r", encoding=enc) as f:
                f.read()
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "latin-1"   # latin-1 never raises — ultimate fallback


def _find_header_row(filepath, encoding="utf-8"):
    with open(filepath, "r", encoding=encoding, errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            cells = [c.strip().lower() for c in line.split(",")]
            if "age" in cells or "target" in cells:
                return i
    return 0


def load_and_prepare(filepath):
    encoding = _detect_encoding(filepath)
    print(f"ℹ  Detected file encoding: {encoding}")
    header_row = _find_header_row(filepath, encoding=encoding)
    df = pd.read_csv(filepath, header=header_row, encoding=encoding)
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
    if df["target"].nunique(dropna=True) > 2:
        df["target"] = (df["target"] > 0).astype(int)
    for col in df.columns:
        if col != "target":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates().dropna()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING (runs once at startup)
# ══════════════════════════════════════════════════════════════════════════════
def build_and_train_models(df):
    """Train RF, SVM, ANN on the full dataset split."""
    X = df[FEATURE_NAMES].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt", bootstrap=True,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # SVM
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                    class_weight="balanced", probability=True,
                    random_state=RANDOM_STATE))
    ])
    svm.fit(X_train, y_train)

    # ANN
    ann = keras.Sequential([
        layers.Input(shape=(len(FEATURE_NAMES),)),
        layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_initializer="he_normal"),
        layers.Dense(1, activation="sigmoid")
    ], name="MLP_HeartDisease")
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

    # SHAP background (100 training samples)
    bg_idx  = np.random.choice(X_train_sc.shape[0], size=100, replace=False)
    shap_explainer = shap.DeepExplainer(ann, X_train_sc[bg_idx])

    # LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_sc,
        feature_names=FEATURE_NAMES,
        class_names=["No Disease", "Disease"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE
    )

    return {
        "rf": rf, "svm": svm, "ann": ann,
        "scaler": scaler,
        "shap_explainer": shap_explainer,
        "lime_explainer": lime_explainer,
        "X_train": X_train,
        "X_train_sc": X_train_sc,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION + EXPLANATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(patient_values: list) -> dict:
    """
    Full pipeline for one patient:
      1. Soft voting ensemble prediction
      2. SHAP values + plots
      3. LIME explanation + plot
      4. Return everything as base64 images + structured data
    """
    rf   = MODELS["rf"]
    svm  = MODELS["svm"]
    ann  = MODELS["ann"]
    sc   = MODELS["scaler"]
    shap_exp  = MODELS["shap_explainer"]
    lime_exp  = MODELS["lime_explainer"]

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
        "High Risk"    if p_ens >= 0.70 else
        "Moderate Risk" if p_ens >= 0.45 else
        "Low Risk"
    )

    # ── SHAP ─────────────────────────────────────────────────────────────
    shap_vals_raw = shap_exp.shap_values(x_scaled)
    if isinstance(shap_vals_raw, list):
        shap_vals_raw = shap_vals_raw[0]
    if shap_vals_raw.ndim == 3:
        shap_vals_raw = shap_vals_raw[:, :, 0]
    shap_vals = shap_vals_raw[0]   # shape (13,)

    # Get base value
    try:
        raw_base = shap_exp.expected_value
        if hasattr(raw_base, "numpy"):
            raw_base = raw_base.numpy()
        base_value = float(np.array(raw_base).flatten()[0])
    except Exception:
        base_value = p_ann

    shap_img    = _plot_shap_bar(shap_vals, patient_values, sc)
    shap_waterfall = _plot_shap_waterfall(shap_vals, base_value,
                                          x_scaled[0], patient_values, sc)

    # ── LIME ─────────────────────────────────────────────────────────────
    def predict_proba_fn(X):
        p = ann.predict(X, verbose=0).flatten()
        return np.column_stack([1 - p, p])

    lime_result = lime_exp.explain_instance(
        data_row=x_scaled[0],
        predict_fn=predict_proba_fn,
        num_features=13,
        num_samples=3000,
        labels=(1,)
    )
    lime_img = _plot_lime_bar(lime_result)

    # ── Build structured feature table ───────────────────────────────────
    original = sc.inverse_transform(x_scaled)[0]
    feature_table = []
    for i, fname in enumerate(FEATURE_NAMES):
        feature_table.append({
            "feature":   fname,
            "label":     FEATURE_LABELS[fname],
            "value":     round(original[i], 2),
            "shap":      round(float(shap_vals[i]), 4),
            "direction": "↑ Risk" if shap_vals[i] > 0 else "↓ Risk"
        })
    feature_table.sort(key=lambda x: abs(x["shap"]), reverse=True)

    lime_weights = dict(lime_result.as_list(label=1))

    return {
        "p_rf":    round(p_rf * 100, 1),
        "p_svm":   round(p_svm * 100, 1),
        "p_ann":   round(p_ann * 100, 1),
        "p_ens":   round(p_ens * 100, 1),
        "prediction": prediction,
        "outcome":    outcome,
        "risk_level": risk_level,
        "shap_bar_img":       shap_img,
        "shap_waterfall_img": shap_waterfall,
        "lime_img":           lime_img,
        "feature_table":      feature_table,
        "lime_weights":       lime_weights,
        "patient_values":     patient_values,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── Plot helpers — return base64 PNG strings ──────────────────────────────────
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _plot_shap_bar(shap_vals, patient_values, scaler) -> str:
    original = scaler.inverse_transform(
        np.array(patient_values, dtype=float).reshape(1, -1)
    )[0]
    idx_sorted = np.argsort(np.abs(shap_vals))
    feats  = [FEATURE_NAMES[i] for i in idx_sorted]
    vals   = [shap_vals[i] for i in idx_sorted]
    colors_bar = ["#e74c3c" if v > 0 else "#2980b9" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1117")
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


def _plot_shap_waterfall(shap_vals, base_val, x_scaled,
                          patient_values, scaler) -> str:
    original = scaler.inverse_transform(
        np.array(patient_values, dtype=float).reshape(1, -1)
    )[0]
    idx = np.argsort(np.abs(shap_vals))[::-1][:8]   # top 8
    feats = [f"{FEATURE_NAMES[i]}={original[i]:.1f}" for i in idx]
    vals  = [shap_vals[i] for i in idx]
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


def _plot_lime_bar(lime_result) -> str:
    feat_weights = lime_result.as_list(label=1)
    feat_weights.sort(key=lambda x: x[1])
    feats   = [f[0] for f in feat_weights]
    weights = [f[1] for f in feat_weights]
    colors_bar = ["#e74c3c" if w > 0 else "#2980b9" for w in weights]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1117")
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
    """Generate a clinical PDF report and return its file path."""
    path = os.path.join(app.config["REPORT_FOLDER"], f"{report_id}.pdf")

    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                  fontSize=18, textColor=colors.HexColor("#1a1a2e"),
                                  spaceAfter=4, alignment=TA_CENTER)
    sub_style   = ParagraphStyle("Sub", parent=styles["Normal"],
                                  fontSize=10, textColor=colors.HexColor("#666666"),
                                  spaceAfter=2, alignment=TA_CENTER)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"],
                                    fontSize=13, textColor=colors.HexColor("#16213e"),
                                    spaceBefore=14, spaceAfter=6)
    body_style  = ParagraphStyle("Body", parent=styles["Normal"],
                                  fontSize=10, textColor=colors.HexColor("#333333"),
                                  spaceAfter=4)
    risk_color  = ("#c0392b" if result["prediction"] == 1 else "#27ae60")

    story = []

    # ── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("AI for Interpretable Heart Disease Prediction", title_style))
    story.append(Paragraph("Clinical Prediction Report", sub_style))
    story.append(Paragraph(f"Generated: {result['timestamp']}  |  Report ID: {report_id}",
                            sub_style))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#16213e"), spaceAfter=12))

    # ── Prediction Summary ────────────────────────────────────────────────
    story.append(Paragraph("Prediction Summary", section_style))

    outcome_style = ParagraphStyle("Outcome", parent=styles["Normal"],
                                    fontSize=16, fontName="Helvetica-Bold",
                                    textColor=colors.HexColor(risk_color),
                                    spaceAfter=8, alignment=TA_CENTER)
    story.append(Paragraph(f"▶  {result['outcome']}  ({result['risk_level']})", outcome_style))

    pred_data = [
        ["Model", "Disease Probability", "Contribution"],
        ["Random Forest",  f"{result['p_rf']}%",  "33%"],
        ["SVM (RBF)",      f"{result['p_svm']}%", "33%"],
        ["ANN (MLP)",      f"{result['p_ann']}%", "33%"],
        ["Ensemble (Avg)", f"{result['p_ens']}%", "Final Decision"],
    ]
    pred_table = Table(pred_data, colWidths=[6*cm, 5*cm, 5*cm])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2),
         [colors.HexColor("#f8f9fa"), colors.white]),
        ("BACKGROUND",   (0, -1), (-1, -1), colors.HexColor("#ffeeba")),
        ("FONTNAME",     (0, -1), (-1, -1), "Helvetica-Bold"),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("ROWHEIGHT",    (0, 0), (-1, -1), 22),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Patient Features ──────────────────────────────────────────────────
    story.append(Paragraph("Patient Clinical Features", section_style))
    feat_data = [["Feature", "Value", "SHAP Impact", "Direction"]]
    for row in result["feature_table"]:
        feat_data.append([
            row["label"],
            str(row["value"]),
            f"{row['shap']:+.4f}",
            row["direction"]
        ])
    feat_table = Table(feat_data, colWidths=[7*cm, 2.5*cm, 3*cm, 3.5*cm])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f8f9fa"), colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
        ("ROWHEIGHT",    (0, 0), (-1, -1), 18),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 0.4*cm))

    # ── SHAP Plot ─────────────────────────────────────────────────────────
    story.append(Paragraph("SHAP Explainability", section_style))
    story.append(Paragraph(
        "SHAP (SHapley Additive exPlanations) shows how each clinical feature "
        "contributed to this prediction. Red bars push toward disease; "
        "blue bars push away from disease.", body_style))

    shap_img_data = base64.b64decode(result["shap_bar_img"])
    shap_img_buf  = io.BytesIO(shap_img_data)
    story.append(RLImage(shap_img_buf, width=14*cm, height=9*cm))
    story.append(Spacer(1, 0.3*cm))

    wf_img_data = base64.b64decode(result["shap_waterfall_img"])
    wf_img_buf  = io.BytesIO(wf_img_data)
    story.append(RLImage(wf_img_buf, width=14*cm, height=7*cm))

    # ── LIME Plot ─────────────────────────────────────────────────────────
    story.append(Paragraph("LIME Explainability", section_style))
    story.append(Paragraph(
        "LIME (Local Interpretable Model-Agnostic Explanations) fits a "
        "linear model around this patient's neighbourhood to explain the "
        "prediction locally. Positive weights push toward disease; "
        "negative weights push away.", body_style))

    lime_img_data = base64.b64decode(result["lime_img"])
    lime_img_buf  = io.BytesIO(lime_img_data)
    story.append(RLImage(lime_img_buf, width=14*cm, height=9*cm))

    # ── Disclaimer ────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#dee2e6")))
    disclaimer_style = ParagraphStyle("Disc", parent=styles["Normal"],
                                       fontSize=8,
                                       textColor=colors.HexColor("#999999"),
                                       spaceAfter=2)
    story.append(Paragraph(
        "⚠  DISCLAIMER: This report is generated by an AI research system "
        "for educational and research purposes only. It is NOT a substitute "
        "for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified physician.", disclaimer_style))

    doc.build(story)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    """Render the patient input form."""
    return render_template("index.html",
                           features=FEATURE_NAMES,
                           feature_labels=FEATURE_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    """Receive form data, run pipeline, return results + report ID."""
    try:
        patient_values = [float(request.form.get(f, 0)) for f in FEATURE_NAMES]
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    result    = run_pipeline(patient_values)
    report_id = str(uuid.uuid4())[:8]

    # Generate PDF in background
    generate_pdf_report(result, report_id)
    result["report_id"] = report_id

    return render_template("result.html", result=result,
                           feature_labels=FEATURE_LABELS)


@app.route("/report/<report_id>")
def download_report(report_id):
    """Stream the PDF report for download."""
    # Sanitise report_id to prevent path traversal
    safe_id = "".join(c for c in report_id if c.isalnum() or c == "-")
    path = os.path.join(app.config["REPORT_FOLDER"], f"{safe_id}.pdf")
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True,
                     download_name=f"heart_disease_report_{safe_id}.pdf",
                     mimetype="application/pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════
def startup():
    """Load data and train models once when Flask starts."""
    print("⏳ Loading dataset and training models — please wait...")
    df = load_and_prepare(DATA_PATH)
    MODELS.update(build_and_train_models(df))
    print("✓ All models ready. Flask is live.")


if __name__ == "__main__":
    startup()
    app.run(debug=False, host="0.0.0.0", port=5000)
