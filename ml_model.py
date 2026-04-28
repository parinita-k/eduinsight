"""
╔══════════════════════════════════════════════════════════════╗
║  ml_model.py — XGBoost + SHAP Model                        ║
║                                                              ║
║  Run this file ONCE to train and save models:               ║
║      python ml_model.py                                      ║
║                                                              ║
║  It will create:                                             ║
║      reg_model.pkl   — regression model (predict score)      ║
║      clf_model.pkl   — classifier (predict pass/fail)        ║
║      exp_reg.pkl     — SHAP explainer for regression         ║
║      exp_clf.pkl     — SHAP explainer for classifier         ║
║      metrics.json    — accuracy, F1, AUC-ROC, MAE, R²        ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import pickle, json, os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, r2_score
)
import xgboost as xgb
import shap

# ── Config ───────────────────────────────────────────────────────────────────
FEATURES = [
    "attendance",       # attendance percentage
    "assignments",      # assignment completion percentage
    "study_hours",      # weekly study hours
    "participation",    # class participation percentage
    "midterm_score",    # mid-term exam score
]
TARGET_REG = "final_score"   # What the regression model predicts
TARGET_CLF = "passed"        # What the classifier predicts (0 or 1)

DATA_PATH    = "students.csv"
MODEL_PREFIX = ""   # models saved in same folder; change to "models/" if needed

# ── Data loading ─────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load student CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset '{path}' not found.\n"
            "Run: python generate_data.py"
        )
    return pd.read_csv(path)

# ── Training ─────────────────────────────────────────────────────────────────
def train_models(df: pd.DataFrame):
    """
    Train two XGBoost models:
      1. Regressor  → predicts final_score (0-100)
      2. Classifier → predicts pass/fail (0 or 1)
    Then create TreeSHAP explainers for both.
    """
    X     = df[FEATURES].values
    y_reg = df[TARGET_REG].values
    y_clf = df[TARGET_CLF].values

    # 80/20 train-test split, stratified on pass/fail
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    print("🔧 Training XGBoost Regressor (predict final score)…")
    reg = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    reg.fit(
        X_tr, yr_tr,
        eval_set=[(X_te, yr_te)],
        verbose=False,
    )

    print("🔧 Training XGBoost Classifier (predict pass/fail)…")
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(
        X_tr, yc_tr,
        eval_set=[(X_te, yc_te)],
        verbose=False,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    yr_pred = reg.predict(X_te)
    yc_pred = clf.predict(X_te)
    yc_prob = clf.predict_proba(X_te)[:, 1]

    metrics = {
        "regression": {
            "mae": round(float(mean_absolute_error(yr_te, yr_pred)), 3),
            "r2":  round(float(r2_score(yr_te, yr_pred)), 3),
        },
        "classification": {
            "accuracy": round(float(accuracy_score(yc_te, yc_pred)), 3),
            "f1":       round(float(f1_score(yc_te, yc_pred)), 3),
            "auc_roc":  round(float(roc_auc_score(yc_te, yc_prob)), 3),
        },
        "feature_importance": {
            f: round(float(v), 4)
            for f, v in zip(FEATURES, reg.feature_importances_)
        },
    }

    print("\n📊 Model Performance:")
    print(f"   Regression  — MAE: {metrics['regression']['mae']}, R²: {metrics['regression']['r2']}")
    print(f"   Classifier  — Accuracy: {metrics['classification']['accuracy']}, "
          f"F1: {metrics['classification']['f1']}, AUC: {metrics['classification']['auc_roc']}")

    # ── SHAP Explainers ───────────────────────────────────────────────────────
    print("\n🔬 Creating TreeSHAP explainers…")
    exp_reg = shap.TreeExplainer(reg)
    exp_clf = shap.TreeExplainer(clf)

    # ── Save everything ───────────────────────────────────────────────────────
    with open(f"{MODEL_PREFIX}reg_model.pkl", "wb") as f:
        pickle.dump(reg, f)
    with open(f"{MODEL_PREFIX}clf_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(f"{MODEL_PREFIX}exp_reg.pkl", "wb") as f:
        pickle.dump(exp_reg, f)
    with open(f"{MODEL_PREFIX}exp_clf.pkl", "wb") as f:
        pickle.dump(exp_clf, f)
    with open(f"{MODEL_PREFIX}metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Models saved: reg_model.pkl, clf_model.pkl, exp_reg.pkl, exp_clf.pkl")
    print(json.dumps(metrics, indent=2))

    return reg, clf, exp_reg, exp_clf, metrics

# ── Loading saved models ──────────────────────────────────────────────────────
def load_models(prefix: str = MODEL_PREFIX):
    """Load pre-trained models from disk."""
    files = ["reg_model.pkl", "clf_model.pkl", "exp_reg.pkl", "exp_clf.pkl", "metrics.json"]
    missing = [f for f in files if not os.path.exists(prefix + f)]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}\n"
            "Run: python ml_model.py"
        )

    with open(f"{prefix}reg_model.pkl", "rb") as f:
        reg = pickle.load(f)
    with open(f"{prefix}clf_model.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(f"{prefix}exp_reg.pkl",   "rb") as f:
        exp_reg = pickle.load(f)
    with open(f"{prefix}exp_clf.pkl",   "rb") as f:
        exp_clf = pickle.load(f)
    with open(f"{prefix}metrics.json")  as f:
        metrics = json.load(f)

    return reg, clf, exp_reg, exp_clf, metrics

# ── Helper functions ─────────────────────────────────────────────────────────
def score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"

def score_to_risk(score: float, attendance: float) -> str:
    """Determine risk level from score and attendance."""
    if score < 50 or attendance < 60: return "high"
    if score < 65 or attendance < 75: return "medium"
    return "low"

def predict_student(features: dict, reg, clf, exp_reg, exp_clf) -> dict:
    """
    Run full prediction pipeline for one student.

    Args:
        features: dict with keys matching FEATURES list
        reg, clf: trained XGBoost models
        exp_reg, exp_clf: TreeSHAP explainers

    Returns:
        dict with predicted_score, predicted_grade, pass_probability,
             confidence, risk_level, shap_score, shap_pass,
             expected_value_score, expected_value_pass
    """
    X = np.array([[features[f] for f in FEATURES]])

    # Predictions
    pred_score = float(np.clip(reg.predict(X)[0], 0, 100))
    pass_prob  = float(clf.predict_proba(X)[0][1])

    # SHAP values (exact TreeSHAP — no approximation)
    sv_reg = exp_reg.shap_values(X)[0]
    sv_clf = exp_clf.shap_values(X)[0]

    shap_reg = {f: round(float(v), 3) for f, v in zip(FEATURES, sv_reg)}
    shap_clf = {f: round(float(v), 3) for f, v in zip(FEATURES, sv_clf)}

    # Confidence: 0% at decision boundary (50%), 100% at extreme ends
    confidence = round(abs(pass_prob - 0.5) * 2 * 100, 1)

    return {
        "predicted_score":      round(pred_score, 1),
        "predicted_grade":      score_to_grade(pred_score),
        "pass_probability":     round(pass_prob * 100, 1),
        "confidence":           confidence,
        "risk_level":           score_to_risk(pred_score, features.get("attendance", 75)),
        "shap_score":           shap_reg,   # SHAP for regression (score prediction)
        "shap_pass":            shap_clf,   # SHAP for classification (pass/fail)
        "expected_value_score": round(float(exp_reg.expected_value), 2),
        "expected_value_pass":  round(float(exp_clf.expected_value), 2),
    }

# ── Run training if executed directly ────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  EduInsight XAI — Model Training")
    print("=" * 60)

    df = load_data()
    print(f"✅ Loaded {len(df)} students from '{DATA_PATH}'")
    print(f"   Features: {FEATURES}")
    print(f"   Target:   {TARGET_REG} (regression), {TARGET_CLF} (classification)")
    print()

    reg, clf, exp_reg, exp_clf, metrics = train_models(df)

    # Quick sanity check
    print("\n🧪 Sanity check — predicting first student:")
    row = df.iloc[0]
    feats = {f: row[f] for f in FEATURES}
    result = predict_student(feats, reg, clf, exp_reg, exp_clf)
    print(f"   Student:         {row['name']}")
    print(f"   Actual score:    {row['final_score']}")
    print(f"   Predicted score: {result['predicted_score']}")
    print(f"   Grade:           {result['predicted_grade']}")
    print(f"   Pass prob:       {result['pass_probability']}%")
    print(f"   Risk level:      {result['risk_level']}")
    print(f"   SHAP values:     {result['shap_score']}")
    print()
    print("✅ All done! Now run: uvicorn main:app --port 8000 --reload")