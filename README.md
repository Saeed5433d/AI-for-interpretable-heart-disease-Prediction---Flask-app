# AI-for-interpretable-heart-disease-Prediction---Flask-app
It is a flask app for a project on the topic "AI for Interpretable Heart Disease Prediction". It takes 13 featurres as input from the user and provides prediction.
```
DEVELOPMENT  (run on Kaggle)
─────────────────────────────
Dataset
   ↓
Cleaning
   ↓
RFE  ──┐
Boruta ─┤── Selected Features
All  ───┘
   ↓
Train RF + SVM + ANN (per feature set)
   ↓
Compare → Pick Best
   ↓
Save: rf_model.pkl, svm_model.pkl,
      ann_model.keras, scaler.pkl, metadata.pkl
 
 
DEPLOYMENT  (run locally in VS Code)
─────────────────────────────────────
User fills form
   ↓
Flask (app.py)
   ↓
Load saved_models/ (instant — no retraining)
   ↓
Soft Voting Ensemble → Prediction
   ↓
SHAP → Bar + Waterfall
   ↓
LIME → Local Explanation
   ↓
PDF Clinical Report (download)
```
 
## File Structure
```
flask_app/
├── app.py                        ← Flask deployment app
├── requirements.txt
├── README.md
├── Book1.csv                     ← Put your dataset here
├── reports/                      ← Auto-created, stores PDFs
├── saved_models/                 ← Created by pipeline script
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── ann_model.keras
│   ├── scaler.pkl
│   └── metadata.pkl              ← Tells Flask which features to use
└── templates/
    ├── index.html
    └── result.html