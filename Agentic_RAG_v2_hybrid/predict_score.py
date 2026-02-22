"""
SwissblAIz Leaderboard Score Predictor

Formal scoring model:
  S = 0.60 * F1 + 0.15 * T + 0.25 * D

Where F1 is difficulty-weighted:
  F1 = 0.20 * F1_easy + 0.30 * F1_medium + 0.50 * F1_hard

Usage:
  python predict_score.py
"""

def predict_score(
    f1_easy: float,
    f1_medium: float,
    f1_hard: float,
    time_score: float = 1.0,
    on_device_ratio: float = 1.0,
) -> dict:
    """Predict leaderboard score from component F1s."""
    
    # Difficulty-weighted F1
    f1_weighted = 0.20 * f1_easy + 0.30 * f1_medium + 0.50 * f1_hard
    
    # Total score
    total = 0.60 * f1_weighted + 0.15 * time_score + 0.25 * on_device_ratio
    
    return {
        "f1_easy": f1_easy,
        "f1_medium": f1_medium,
        "f1_hard": f1_hard,
        "f1_weighted": f1_weighted,
        "time_score": time_score,
        "on_device_ratio": on_device_ratio,
        "total_score": total,
        "total_pct": total * 100,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  SwissblAIz Score Predictor")
    print("=" * 60)
    
    # Submission 1 actual: 40.8%, F1=0.2778, T=1.0, D=1.0
    print("\n--- Submission 1 (actual: 40.8%) ---")
    s1 = predict_score(f1_easy=0.40, f1_medium=0.30, f1_hard=0.15, time_score=1.0, on_device_ratio=1.0)
    print(f"  F1 weighted: {s1['f1_weighted']:.4f}")
    print(f"  Predicted:   {s1['total_pct']:.1f}%")
    
    # Submission 2 estimate: Hard→cloud (F1~0.85), Easy/Medium local with better prompt
    # On-device ratio: ~50% (Easy/Medium local, Hard cloud)
    print("\n--- Submission 2 (predicted) ---")
    # Conservative: prompt fix helps Easy/Medium by +0.15
    s2_conservative = predict_score(
        f1_easy=0.55,      # Better prompt + normalization
        f1_medium=0.45,    # Better prompt + normalization  
        f1_hard=0.85,      # Cloud Gemini
        time_score=0.95,   # Cloud adds latency for Hard
        on_device_ratio=0.50,  # ~50% cases are Easy/Medium (local)
    )
    print(f"  Conservative: {s2_conservative['total_pct']:.1f}% (F1={s2_conservative['f1_weighted']:.4f})")
    
    # Optimistic: prompt fix + normalization significantly helps
    s2_optimistic = predict_score(
        f1_easy=0.70,
        f1_medium=0.60,
        f1_hard=0.90,
        time_score=0.95,
        on_device_ratio=0.50,
    )
    print(f"  Optimistic:   {s2_optimistic['total_pct']:.1f}% (F1={s2_optimistic['f1_weighted']:.4f})")
    
    # What we need to beat: Submission 1's 40.8%
    # Breakeven: what Hard cloud F1 makes mixed strategy = all-local?
    print("\n--- Breakeven Analysis ---")
    print("  For Hard→cloud to beat all-local:")
    print("  F1_H(cloud) - F1_H(local) > 0.4167")
    print(f"  If F1_H(local) = 0.15 → need F1_H(cloud) > {0.15 + 0.4167:.3f}")
    print(f"  If F1_H(local) = 0.05 → need F1_H(cloud) > {0.05 + 0.4167:.3f}")
    
    # What's the top of the board?
    print("\n--- Target Analysis ---")
    targets = [50, 60, 70, 80, 89]
    for t in targets:
        # What weighted F1 is needed?
        # total = 0.60 * F1 + 0.15 * T + 0.25 * D
        # At 50% on-device: total = 0.60*F1 + 0.15*0.95 + 0.25*0.50
        # F1 = (total - 0.2675) / 0.60
        needed_f1 = (t/100 - 0.2675) / 0.60
        print(f"  {t}% total → needs weighted F1 = {needed_f1:.3f}")
    
    print()
