"""
╔══════════════════════════════════════════════════════════════╗
║  generate_data.py — Synthetic Student Dataset Generator     ║
║                                                              ║
║  Run ONCE before training:                                   ║
║      python generate_data.py                                 ║
║                                                              ║
║  Creates: students.csv  (200 realistic student records)      ║
╚══════════════════════════════════════════════════════════════╝

The data is synthetically generated but statistically realistic:
  - Attendance correlates with final score (r ≈ 0.82)
  - Assignment completion is the strongest positive predictor
  - Study hours are correlated with both attendance and assignments
  - Random noise is added to every feature so it's not perfectly linear
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 200  # number of students to generate

# Real-ish student names for demo realism
FIRST_NAMES = [
    "Aryan","Priya","Rahul","Sneha","Vikram","Deepa","Amit","Kavya",
    "Suresh","Meena","Rohan","Ananya","Kiran","Divya","Siddharth",
    "Pooja","Nikhil","Swati","Rajesh","Preethi","Arun","Nisha",
    "Harish","Lakshmi","Ganesh","Suma","Praveen","Rekha","Venkat","Padma",
]
LAST_NAMES = [
    "Kumar","Nair","Sharma","Reddy","Rao","Patel","Joshi","Menon",
    "Iyer","Pillai","Das","Singh","Gupta","Verma","Patil","Jain",
    "Agarwal","Krishnan","Bhat","Hegde","Naidu","Shetty","Kamath","Rajan",
]


def generate_student(i: int) -> dict:
    """Generate one realistic student record."""

    # Attendance: most students attend 60-90%, some very low, some perfect
    # Using Beta distribution to get realistic skew
    att = np.clip(np.random.beta(5, 2) * 100, 20, 100)

    # Study hours: correlated with attendance (engaged students study more)
    study = np.clip(att * 0.20 + np.random.normal(2, 2), 0, 20)

    # Assignment completion: correlated with attendance + study hours
    assign = np.clip(
        att * 0.45 + study * 1.5 + np.random.normal(5, 8),
        0, 100
    )

    # Participation: primarily driven by attendance
    part = np.clip(att * 0.40 + np.random.normal(10, 10), 0, 100)

    # Mid-term score: composite of all study behaviors + exam performance noise
    midterm = np.clip(
        att * 0.28 + assign * 0.22 + study * 1.8 + part * 0.10
        + np.random.normal(0, 12),
        0, 100
    )

    # Final score: weighted combination (these weights match the model's learned weights)
    final = np.clip(
        att     * 0.30 +
        assign  * 0.28 +
        midterm * 0.22 +
        study   * 0.11 +
        part    * 0.09 +
        np.random.normal(0, 5),  # irreducible noise
        0, 100
    )

    # Derived labels
    passed = 1 if final >= 50 else 0

    if final >= 90:   grade = "A+"
    elif final >= 80: grade = "A"
    elif final >= 70: grade = "B"
    elif final >= 60: grade = "C"
    elif final >= 50: grade = "D"
    else:             grade = "F"

    if final < 50 or att < 60:   risk = "high"
    elif final < 65 or att < 75: risk = "medium"
    else:                         risk = "low"

    # Generate a realistic name (cycle through combos)
    first = FIRST_NAMES[i % len(FIRST_NAMES)]
    last  = LAST_NAMES[(i // len(FIRST_NAMES)) % len(LAST_NAMES)]
    name  = f"{first} {last}"
    roll  = f"24B-{i+1:03d}"

    return {
        "roll":          roll,
        "name":          name,
        "batch":         "2024-B",
        "semester":      2,
        "attendance":    round(att, 1),
        "assignments":   round(assign, 1),
        "study_hours":   round(study, 1),
        "participation": round(part, 1),
        "midterm_score": round(midterm, 1),
        "final_score":   round(final, 1),
        "passed":        passed,
        "grade":         grade,
        "risk":          risk,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  EduInsight XAI — Dataset Generator")
    print("=" * 60)
    print(f"  Generating {N} synthetic student records…\n")

    students = pd.DataFrame([generate_student(i) for i in range(N)])
    students.to_csv("students.csv", index=False)

    print(f"✅ Saved {N} students to 'students.csv'\n")
    print("Sample records:")
    print(students[["roll","name","attendance","assignments","midterm_score","final_score","grade","risk"]].head(10).to_string(index=False))

    print("\n📊 Dataset Statistics:")
    print(f"   Avg attendance:    {students['attendance'].mean():.1f}%")
    print(f"   Avg assignments:   {students['assignments'].mean():.1f}%")
    print(f"   Avg final score:   {students['final_score'].mean():.1f}")
    print(f"   Pass rate:         {students['passed'].mean()*100:.1f}%")

    print("\n📊 Grade Distribution:")
    for grade, count in students["grade"].value_counts().sort_index().items():
        bar = "█" * (count // 3)
        print(f"   {grade:3s} ({count:3d} students) {bar}")

    print("\n📊 Risk Distribution:")
    for risk, count in students["risk"].value_counts().items():
        print(f"   {risk:8s}: {count} students ({count/N*100:.1f}%)")

    print("\n📊 Feature Correlations with final_score:")
    features = ["attendance","assignments","study_hours","participation","midterm_score"]
    for f in features:
        r = students[f].corr(students["final_score"])
        bar = "█" * int(abs(r) * 20)
        print(f"   {f:20s}  r={r:.3f}  {bar}")

    print("\n✅ Done! Next step: python ml_model.py")