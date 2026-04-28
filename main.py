"""
╔══════════════════════════════════════════════════════════════╗
║         EduInsight XAI — FastAPI Backend                    ║
║         Student Performance Prediction System               ║
║                                                              ║
║  HOW TO RUN:                                                 ║
║    1. pip install -r requirements.txt                        ║
║    2. python generate_data.py        (creates students.csv)  ║
║    3. python ml_model.py             (trains & saves models) ║
║    4. uvicorn main:app --port 8000 --reload                  ║
║    5. Open http://localhost:8000/docs  (Swagger UI)          ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, json, io
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

# ── Import model helpers ─────────────────────────────────────────────────────
from ml_model import (
    load_data, train_models, load_models,
    predict_student, score_to_grade, score_to_risk, FEATURES
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EduInsight XAI API",
    description="""
## Explainable AI for Student Performance Prediction

This API uses **XGBoost + SHAP** to predict student grades and explain WHY
each prediction was made — making the AI transparent and trustworthy.

### Key Features
- **Predict** final scores for individual or batch students
- **Explain** predictions using SHAP values (per-feature contribution)
- **Detect** at-risk students automatically
- **Analyse** batch-level trends and correlations
- **Retrain** the model on new data

### XAI (Explainability)
Every prediction comes with SHAP values that show exactly how much
each feature (attendance, assignments, etc.) contributed to the score.
""",
    version="1.0.0",
)

# Allow frontend to call this API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state — loaded once at startup ────────────────────────────────────
reg = clf = exp_reg = exp_clf = metrics = df = student_predictions = None

@app.on_event("startup")
async def startup():
    global reg, clf, exp_reg, exp_clf, metrics, df, student_predictions
    try:
        reg, clf, exp_reg, exp_clf, metrics = load_models()
        df = load_data()
        student_predictions = _predict_all(df)
        print(f"✅ API ready — {len(df)} students loaded, accuracy={metrics['classification']['accuracy']}")
    except FileNotFoundError:
        print("⚠️  Models not found. Run: python ml_model.py")
        print("⚠️  Data not found.   Run: python generate_data.py")

def _predict_all(dataframe: pd.DataFrame) -> dict:
    """Pre-compute predictions for every student in the dataset."""
    preds = {}
    for _, row in dataframe.iterrows():
        feats = {f: float(row[f]) for f in FEATURES}
        preds[row["roll"]] = predict_student(feats, reg, clf, exp_reg, exp_clf)
    return preds

# ── Request / Response schemas ────────────────────────────────────────────────
class StudentFeatures(BaseModel):
    attendance:    float = Field(..., ge=0, le=100,  example=72.5,
                                 description="Attendance percentage (0-100)")
    assignments:   float = Field(..., ge=0, le=100,  example=68.0,
                                 description="Assignment completion percentage (0-100)")
    study_hours:   float = Field(..., ge=0, le=20,   example=4.5,
                                 description="Weekly study hours (0-20)")
    participation: float = Field(..., ge=0, le=100,  example=60.0,
                                 description="Class participation percentage (0-100)")
    midterm_score: float = Field(..., ge=0, le=100,  example=55.0,
                                 description="Mid-term exam score (0-100)")
    name: Optional[str] = Field("Unknown", example="Aryan Kumar")
    roll: Optional[str] = Field("UNKNOWN", example="24B-001")

# ── ROUTES ────────────────────────────────────────────────────────────────────

# 1. Health check
@app.get("/", tags=["Health"], summary="API health check")
def root():
    """Returns API status, model accuracy, and basic stats."""
    return {
        "status": "online",
        "service": "EduInsight XAI API",
        "version": "1.0.0",
        "students_loaded": len(df) if df is not None else 0,
        "model_accuracy": metrics["classification"]["accuracy"] if metrics else None,
        "model_auc":      metrics["classification"]["auc_roc"]  if metrics else None,
        "f1_score":       metrics["classification"]["f1"]        if metrics else None,
        "docs_url": "http://localhost:8000/docs",
    }

# 2. All students with predictions
@app.get("/api/students", tags=["Students"], summary="Get all students with predictions")
def get_students(
    risk:   Optional[str] = Query(None, description="Filter: high / medium / low"),
    batch:  Optional[str] = Query(None, description="Filter by batch name"),
    limit:  int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0,  ge=0,         description="Pagination offset"),
):
    """
    Returns all students merged with their AI predictions.
    Use `risk=high` to get only at-risk students.
    """
    if df is None:
        raise HTTPException(503, "Models not loaded. Run ml_model.py first.")

    result = []
    for _, row in df.iterrows():
        pred = student_predictions.get(row["roll"], {})
        merged = {**row.to_dict(), **pred}
        if risk and pred.get("risk_level") != risk:
            continue
        if batch and row.get("batch") != batch:
            continue
        result.append(merged)

    return {
        "total": len(result),
        "showing": f"{offset}–{offset+limit}",
        "students": result[offset: offset + limit],
    }

# 3. Single student detail
@app.get("/api/students/{roll}", tags=["Students"], summary="Get one student — full detail + SHAP")
def get_student(roll: str):
    """
    Returns complete student profile including:
    - All features and actual scores
    - AI-predicted final score and grade
    - SHAP values (why the prediction)
    - Alerts and intervention flags
    - Subject-wise score breakdown
    """
    if df is None:
        raise HTTPException(503, "Models not loaded.")

    row = df[df["roll"] == roll]
    if row.empty:
        raise HTTPException(404, f"Student '{roll}' not found. Check /api/students for valid rolls.")
    row = row.iloc[0]
    pred = student_predictions.get(roll, {})

    # Top SHAP drivers
    shap_s = pred.get("shap_score", {})
    top_pos = sorted(shap_s.items(), key=lambda x:  x[1], reverse=True)[:2]
    top_neg = sorted(shap_s.items(), key=lambda x:  x[1])[:2]

    # Alerts
    alerts = []
    if row["attendance"] < 60:
        alerts.append({"type":"attendance","severity":"critical",
                       "message":f"Attendance {row['attendance']}% is critically below 60% threshold"})
    if row["assignments"] < 60:
        alerts.append({"type":"assignments","severity":"warning",
                       "message":f"Assignment completion {row['assignments']}% needs improvement"})
    if row["midterm_score"] < 40:
        alerts.append({"type":"midterm","severity":"critical",
                       "message":f"Mid-term score {row['midterm_score']} is dangerously low"})
    if row["study_hours"] < 2:
        alerts.append({"type":"study","severity":"warning",
                       "message":f"Only {row['study_hours']} study hours/week — minimum recommended is 4"})

    return {
        "student": row.to_dict(),
        "prediction": pred,
        "top_positive_factors": [{"feature":f,"shap":round(v,3)} for f,v in top_pos],
        "top_negative_factors": [{"feature":f,"shap":round(v,3)} for f,v in top_neg],
        "intervention_needed": pred.get("risk_level") == "high",
        "alerts": alerts,
        "subject_scores": {
            "mathematics":   round(float(row["midterm_score"] * 0.90 + np.random.normal(0,3)), 1),
            "science":       round(float(row["assignments"]   * 0.85 + np.random.normal(0,4)), 1),
            "english":       round(float(row["participation"] * 0.95 + np.random.normal(0,4)), 1),
            "history":       round(float(row["study_hours"]   * 4.50 + np.random.normal(0,5)), 1),
            "physics":       round(float(row["midterm_score"] * 0.80 + np.random.normal(0,6)), 1),
            "computer_sc":   round(float(row["assignments"]   * 0.90 + np.random.normal(0,3)), 1),
        },
    }

# 4. Predict — single student
@app.post("/api/predict", tags=["Prediction"], summary="Predict grade from raw features")
def predict(student: StudentFeatures):
    """
    **Main prediction endpoint.**

    Send student features → get back:
    - Predicted final score (0-100)
    - Predicted grade (F / D / C / B / A / A+)
    - Pass probability (%)
    - Risk level (low / medium / high)
    - SHAP values for every feature
    - Plain-English AI explanation

    ### SHAP interpretation
    - Positive SHAP = feature HELPED the grade
    - Negative SHAP = feature HURT the grade
    - |SHAP| = how much impact (bigger = more important)
    """
    if reg is None:
        raise HTTPException(503, "Models not loaded. Run ml_model.py first.")

    feats = {f: getattr(student, f) for f in FEATURES}
    result = predict_student(feats, reg, clf, exp_reg, exp_clf)

    # Build plain-English explanation
    shap_s = result["shap_score"]
    sorted_shap = sorted(shap_s.items(), key=lambda x: abs(x[1]), reverse=True)
    top_feat, top_val = sorted_shap[0]
    direction = "positively" if top_val > 0 else "negatively"
    second_feat, second_val = sorted_shap[1]

    explanation = (
        f"Predicted score: {result['predicted_score']:.1f}/100 → Grade {result['predicted_grade']}. "
        f"The strongest influence is {top_feat.replace('_',' ')} which shifts the score "
        f"{direction} by {abs(top_val):.1f} points (SHAP). "
        f"{second_feat.replace('_',' ').title()} is the second biggest factor ({second_val:+.1f} pts). "
        f"Pass probability: {result['pass_probability']:.1f}%. "
        f"Risk level: {result['risk_level'].upper()}."
    )

    return {
        "roll": student.roll,
        "name": student.name,
        "input_features": feats,
        "predicted_score":    result["predicted_score"],
        "predicted_grade":    result["predicted_grade"],
        "pass_probability":   result["pass_probability"],
        "confidence":         result["confidence"],
        "risk_level":         result["risk_level"],
        "shap_score":         result["shap_score"],
        "shap_pass":          result["shap_pass"],
        "expected_value_score": result["expected_value_score"],
        "ai_explanation":     explanation,
    }

# 5. Batch prediction from CSV upload
@app.post("/api/predict/batch", tags=["Prediction"], summary="Batch predict from uploaded CSV")
async def predict_batch(file: UploadFile = File(...)):
    """
    Upload a CSV file with columns:
    `name, roll, attendance, assignments, study_hours, participation, midterm_score`

    Returns predictions for every row.
    """
    if reg is None:
        raise HTTPException(503, "Models not loaded.")

    content = await file.read()
    try:
        batch_df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    missing = [f for f in FEATURES if f not in batch_df.columns]
    if missing:
        raise HTTPException(400, f"CSV is missing columns: {missing}")

    results = []
    for _, row in batch_df.iterrows():
        feats = {f: float(row[f]) for f in FEATURES}
        pred  = predict_student(feats, reg, clf, exp_reg, exp_clf)
        results.append({
            "roll": str(row.get("roll", "N/A")),
            "name": str(row.get("name", "Unknown")),
            **pred,
        })

    passed    = sum(1 for r in results if r["pass_probability"] >= 50)
    high_risk = sum(1 for r in results if r["risk_level"] == "high")

    return {
        "filename":        file.filename,
        "total":           len(results),
        "predicted_pass":  passed,
        "predicted_fail":  len(results) - passed,
        "high_risk_count": high_risk,
        "average_score":   round(float(np.mean([r["predicted_score"] for r in results])), 2),
        "predictions":     results,
    }

# 6. Global XAI — feature importance
@app.get("/api/xai/global", tags=["XAI"], summary="Global feature importance (XGBoost + SHAP)")
def global_feature_importance():
    """
    Returns:
    - XGBoost feature importance (gain-based %)
    - Mean |SHAP| across all students
    - Human-readable feature labels
    """
    if metrics is None:
        raise HTTPException(503, "Models not loaded.")

    fi = metrics["feature_importance"]
    total = sum(fi.values())
    fi_pct = {k: round(v / total * 100, 2) for k, v in fi.items()}

    # Mean absolute SHAP across all students
    X_all   = df[FEATURES].values
    sv_all  = exp_reg.shap_values(X_all)
    mean_shap = {
        f: round(float(np.mean(np.abs(sv_all[:, i]))), 3)
        for i, f in enumerate(FEATURES)
    }

    labels = {
        "attendance":    "Attendance Rate",
        "assignments":   "Assignment Completion",
        "study_hours":   "Weekly Study Hours",
        "participation": "Class Participation",
        "midterm_score": "Mid-term Score",
    }

    return {
        "feature_importance_pct": fi_pct,
        "mean_abs_shap":          mean_shap,
        "feature_labels":         labels,
        "top_feature":            max(fi_pct, key=fi_pct.get),
        "model_type":             "XGBoost (Gradient Boosted Trees)",
        "explainer_type":         "TreeSHAP (exact, not approximate)",
    }

# 7. Per-student SHAP
@app.get("/api/xai/shap/{roll}", tags=["XAI"], summary="SHAP breakdown for one student")
def student_shap(roll: str):
    """
    Returns a fully annotated SHAP breakdown for a single student.

    Verifiable: expected_value + sum(shap_values) ≈ predicted_score
    """
    if df is None:
        raise HTTPException(503, "Models not loaded.")

    row = df[df["roll"] == roll]
    if row.empty:
        raise HTTPException(404, f"Student {roll} not found.")

    pred   = student_predictions.get(roll, {})
    shap_s = pred.get("shap_score", {})

    breakdown = []
    for feat in FEATURES:
        val = shap_s.get(feat, 0)
        raw = float(row.iloc[0][feat])
        breakdown.append({
            "feature":     feat,
            "label":       feat.replace("_", " ").title(),
            "raw_value":   raw,
            "shap_value":  val,
            "direction":   "positive" if val > 0 else "negative",
            "magnitude":   "high" if abs(val) > 5 else "medium" if abs(val) > 2 else "low",
            "interpretation": (
                f"{feat.replace('_',' ').title()} = {raw:.1f} "
                f"{'boosted' if val > 0 else 'reduced'} the predicted score by {abs(val):.2f} pts"
            ),
        })

    breakdown.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "roll":              roll,
        "name":              row.iloc[0]["name"],
        "predicted_score":   pred.get("predicted_score"),
        "expected_value":    pred.get("expected_value_score"),
        "shap_sum":          round(sum(b["shap_value"] for b in breakdown), 3),
        "shap_breakdown":    breakdown,
        "math_check":        f"{pred.get('expected_value_score',0):.2f} + {round(sum(b['shap_value'] for b in breakdown),2)} = {pred.get('predicted_score',0):.1f}",
    }

# 8. Analytics summary
@app.get("/api/analytics/summary", tags=["Analytics"], summary="Batch-level analytics")
def analytics_summary():
    """Returns aggregate statistics for the entire student batch."""
    if df is None:
        raise HTTPException(503, "Models not loaded.")

    preds_list = list(student_predictions.values())
    scores = [p["predicted_score"] for p in preds_list]

    grade_dist = {}
    risk_dist  = {"high": 0, "medium": 0, "low": 0}
    for p in preds_list:
        grade_dist[p["predicted_grade"]] = grade_dist.get(p["predicted_grade"], 0) + 1
        risk_dist[p["risk_level"]] += 1

    return {
        "total_students":     len(df),
        "average_score":      round(float(np.mean(scores)), 2),
        "median_score":       round(float(np.median(scores)), 2),
        "std_score":          round(float(np.std(scores)), 2),
        "min_score":          round(float(np.min(scores)), 2),
        "max_score":          round(float(np.max(scores)), 2),
        "predicted_pass_pct": round(sum(1 for p in preds_list if p["pass_probability"] >= 50) / len(preds_list) * 100, 1),
        "grade_distribution": grade_dist,
        "risk_distribution":  risk_dist,
        "avg_attendance":     round(float(df["attendance"].mean()), 2),
        "avg_assignments":    round(float(df["assignments"].mean()), 2),
        "avg_midterm":        round(float(df["midterm_score"].mean()), 2),
        "avg_study_hours":    round(float(df["study_hours"].mean()), 2),
        "correlations": {
            "attendance_vs_score":  round(float(df["attendance"].corr(df["final_score"])), 3),
            "assignments_vs_score": round(float(df["assignments"].corr(df["final_score"])), 3),
            "study_hrs_vs_score":   round(float(df["study_hours"].corr(df["final_score"])), 3),
            "midterm_vs_score":     round(float(df["midterm_score"].corr(df["final_score"])), 3),
            "participation_vs_score":round(float(df["participation"].corr(df["final_score"])), 3),
        },
    }

# 9. At-risk students
@app.get("/api/analytics/risk", tags=["Analytics"], summary="At-risk students list")
def risk_analysis(top: int = Query(20, ge=1, le=200)):
    """Returns students flagged as high/medium risk, sorted by risk score."""
    if df is None:
        raise HTTPException(503, "Models not loaded.")

    at_risk = []
    for _, row in df.iterrows():
        pred = student_predictions.get(row["roll"], {})
        if pred.get("risk_level") in ("high", "medium"):
            risk_score = round(
                (100 - pred["predicted_score"]) * 0.50 +
                (100 - pred["pass_probability"]) * 0.30 +
                (100 - row["attendance"])        * 0.20,
                1
            )
            shap_s = pred.get("shap_score", {})
            top_concern = min(shap_s, key=lambda k: shap_s[k], default="unknown")
            at_risk.append({
                "roll":             row["roll"],
                "name":             row["name"],
                "risk_level":       pred["risk_level"],
                "risk_score":       risk_score,
                "predicted_score":  pred["predicted_score"],
                "predicted_grade":  pred["predicted_grade"],
                "pass_probability": pred["pass_probability"],
                "attendance":       float(row["attendance"]),
                "assignments":      float(row["assignments"]),
                "top_concern":      top_concern.replace("_", " ").title(),
            })

    at_risk.sort(key=lambda x: x["risk_score"], reverse=True)

    return {
        "total_at_risk": len(at_risk),
        "high_risk":     sum(1 for s in at_risk if s["risk_level"] == "high"),
        "medium_risk":   sum(1 for s in at_risk if s["risk_level"] == "medium"),
        "students":      at_risk[:top],
    }

# 10. Model metrics
@app.get("/api/metrics", tags=["Model"], summary="Model performance metrics")
def model_metrics():
    """Returns XGBoost model accuracy, F1, AUC-ROC, MAE, R²."""
    if metrics is None:
        raise HTTPException(503, "Models not loaded.")
    return {
        **metrics,
        "model_info": {
            "algorithm":       "XGBoost (eXtreme Gradient Boosting)",
            "explainability":  "TreeSHAP (exact Shapley values)",
            "n_estimators":    200,
            "features_used":   FEATURES,
            "training_split":  "80% train / 20% test",
            "cross_validation":"Not applied (use cv_score endpoint for CV)",
        }
    }

# 11. Retrain
@app.post("/api/retrain", tags=["Model"], summary="Retrain model on new/updated data")
async def retrain(file: UploadFile = File(None)):
    """
    Retrain the XGBoost model.
    - Without file: retrains on existing students.csv
    - With CSV file: retrains on uploaded data
    """
    global reg, clf, exp_reg, exp_clf, metrics, df, student_predictions

    if file:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    else:
        df = load_data()

    reg, clf, exp_reg, exp_clf, new_metrics = train_models(df)
    metrics = new_metrics
    student_predictions = _predict_all(df)

    return {
        "status":          "✅ Model retrained successfully",
        "samples_used":    len(df),
        "new_metrics":     new_metrics,
    }

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)