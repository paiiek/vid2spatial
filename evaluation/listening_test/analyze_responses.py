#!/usr/bin/env python3
"""
Statistical analysis of Vid2Spatial perceptual evaluation responses.

Designed for small-N expert evaluation (5-10 participants):
  - Median / IQR (robust central tendency)
  - Wilcoxon signed-rank test (paired non-parametric)
  - Cliff's delta effect size (ordinal, assumption-free)
  - Win-rate (% clips where proposed > baseline)
  - Per-dimension and per-motion-category breakdowns

Reads response JSONs from responses/ directory.

Usage:
    python analyze_responses.py [--responses-dir responses/]
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def load_responses(responses_dir: Path):
    """Load all response JSON files."""
    results = []
    for f in sorted(responses_dir.glob("*.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def extract_ratings(responses):
    """Extract ratings into structured format.

    Returns: {scene_id: {condition_id: {question_id: [scores_across_participants]}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for resp in responses:
        for trial in resp.get("trials", []):
            scene = trial["scene"]
            cond = trial["condition"]
            for q_id, score in trial.get("ratings", {}).items():
                data[scene][cond][q_id].append(score)

    return data


def compute_descriptive(data, conditions, questions):
    """Compute per-condition per-question descriptive statistics.

    Returns: {condition: {question: {median, iqr, mean, n}}}
    """
    stats = {}
    for cond in conditions:
        stats[cond] = {}
        for q in questions:
            vals = []
            for scene_data in data.values():
                vals.extend(scene_data.get(cond, {}).get(q, []))
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            q25, q50, q75 = np.percentile(arr, [25, 50, 75])
            stats[cond][q] = {
                "median": round(float(q50), 2),
                "iqr": round(float(q75 - q25), 2),
                "q25": round(float(q25), 2),
                "q75": round(float(q75), 2),
                "mean": round(float(np.mean(arr)), 2),
                "std": round(float(np.std(arr, ddof=1)), 2) if len(arr) > 1 else 0.0,
                "n": len(arr),
            }
    return stats


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size (ordinal, assumption-free).

    delta = (# x>y - # x<y) / (n_x * n_y)
    Interpretation: |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large
    """
    x, y = np.asarray(x), np.asarray(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    more = 0
    less = 0
    for xi in x:
        more += np.sum(xi > y)
        less += np.sum(xi < y)
    delta = (more - less) / (n_x * n_y)
    return round(float(delta), 4)


def cliffs_delta_category(d):
    """Interpret Cliff's delta magnitude."""
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


def wilcoxon_test(a, b):
    """Wilcoxon signed-rank test for paired samples."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None

    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    # Remove zero differences
    nonzero = diff != 0
    if np.sum(nonzero) < 5:
        return None

    try:
        stat, pval = wilcoxon(a[nonzero], b[nonzero])
        return {"statistic": round(float(stat), 4), "p_value": round(float(pval), 6)}
    except ValueError:
        return None


def compute_pairwise(data, cond_a, cond_b, questions):
    """Compute pairwise statistics between two conditions.

    For each question: paired Wilcoxon, Cliff's delta, win-rate.
    Pairing: per-scene mean rating per participant (or all ratings per scene).
    """
    results = {}

    for q in questions:
        # Collect paired (per-scene) data
        a_vals, b_vals = [], []
        wins_a, wins_b, ties = 0, 0, 0

        for scene_id, scene_data in data.items():
            ra = scene_data.get(cond_a, {}).get(q, [])
            rb = scene_data.get(cond_b, {}).get(q, [])
            if not ra or not rb:
                continue

            mean_a = np.mean(ra)
            mean_b = np.mean(rb)
            a_vals.append(mean_a)
            b_vals.append(mean_b)

            if mean_a > mean_b:
                wins_a += 1
            elif mean_b > mean_a:
                wins_b += 1
            else:
                ties += 1

        if len(a_vals) < 3:
            continue

        a_arr = np.array(a_vals)
        b_arr = np.array(b_vals)
        total = wins_a + wins_b + ties

        # Cliff's delta (positive = cond_a better)
        delta = cliffs_delta(a_arr, b_arr)

        # Wilcoxon
        wilcox = wilcoxon_test(a_arr, b_arr)

        results[q] = {
            "cliffs_delta": delta,
            "effect_size": cliffs_delta_category(delta),
            "wilcoxon": wilcox,
            "win_rate": {
                f"{cond_a}_wins": wins_a,
                f"{cond_b}_wins": wins_b,
                "ties": ties,
                f"{cond_a}_pct": round(100 * wins_a / total, 1) if total else 0,
            },
            "n_scenes": total,
            "mean_diff": round(float(np.mean(a_arr - b_arr)), 3),
        }

    return results


def compute_by_motion(data, scenes_meta, conditions, questions):
    """Breakdown statistics by motion category (fast/moderate/slow)."""
    motion_map = {s["id"]: s["motion"] for s in scenes_meta}
    results = {}

    for motion in ["fast", "moderate", "slow"]:
        scene_ids = [sid for sid, m in motion_map.items() if m == motion]
        if not scene_ids:
            continue

        results[motion] = {}
        for cond in conditions:
            results[motion][cond] = {}
            for q in questions:
                vals = []
                for sid in scene_ids:
                    vals.extend(data.get(sid, {}).get(cond, {}).get(q, []))
                if vals:
                    arr = np.array(vals, dtype=float)
                    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
                    results[motion][cond][q] = {
                        "median": round(float(q50), 2),
                        "iqr": round(float(q75 - q25), 2),
                        "n": len(arr),
                    }

    return results


def generate_report(analysis, output_dir: Path):
    """Generate markdown report and JSON."""
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "analysis_results.json"
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)

    md_path = output_dir / "LISTENING_TEST_REPORT.md"
    lines = [
        "# Vid2Spatial Perceptual Evaluation Results\n",
        f"Participants: **{analysis['n_participants']}** | "
        f"Clips: **{analysis['n_clips']}** | "
        f"Scale: 7-point MOS\n",
    ]

    # Descriptive stats
    desc = analysis.get("descriptive", {})
    questions = analysis.get("questions", [])
    conditions = analysis.get("conditions", [])

    lines.append("\n## Per-Dimension Results (Median [IQR])\n")
    lines.append("| Dimension | " + " | ".join(conditions) + " |")
    lines.append("|-----------|" + "|".join(["---"] * len(conditions)) + "|")
    for q in questions:
        row = f"| **{q}** |"
        for cond in conditions:
            s = desc.get(cond, {}).get(q, {})
            if s:
                row += f" {s['median']} [{s['q25']}–{s['q75']}] |"
            else:
                row += " — |"
        lines.append(row)

    # Pairwise comparison
    pairwise = analysis.get("pairwise", {})
    if pairwise:
        c1, c2 = conditions[0], conditions[1]
        lines.append(f"\n## Proposed vs Baseline Comparison\n")
        lines.append(
            "| Dimension | Cliff's δ | Effect | "
            f"Win-rate ({c1}) | Wilcoxon p | Mean diff |"
        )
        lines.append("|-----------|-----------|--------|------------|-----------|-----------|")
        for q in questions:
            pw = pairwise.get(q, {})
            if not pw:
                continue
            wilcox_p = pw.get("wilcoxon", {})
            p_str = f"{wilcox_p['p_value']:.4f}" if wilcox_p else "—"
            wr = pw.get("win_rate", {})
            lines.append(
                f"| **{q}** | {pw['cliffs_delta']} | {pw['effect_size']} | "
                f"{wr.get(f'{c1}_pct', 0)}% | {p_str} | {pw['mean_diff']:+.3f} |"
            )

    # Motion category breakdown
    by_motion = analysis.get("by_motion", {})
    if by_motion:
        lines.append("\n## Breakdown by Motion Category\n")
        for motion in ["fast", "moderate", "slow"]:
            if motion not in by_motion:
                continue
            lines.append(f"\n### {motion.capitalize()} motion\n")
            lines.append("| Dimension | " + " | ".join(conditions) + " |")
            lines.append("|-----------|" + "|".join(["---"] * len(conditions)) + "|")
            for q in questions:
                row = f"| {q} |"
                for cond in conditions:
                    s = by_motion[motion].get(cond, {}).get(q, {})
                    if s:
                        row += f" {s['median']} (IQR {s['iqr']}) |"
                    else:
                        row += " — |"
                lines.append(row)

    lines.append("\n---\n")
    lines.append(
        "*Analysis: Median/IQR, Cliff's delta effect size, "
        "Wilcoxon signed-rank test, win-rate.*\n"
    )
    lines.append("Generated by `analyze_responses.py`\n")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[report] {md_path}")
    print(f"[json]   {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-dir", default="responses")
    args = parser.parse_args()

    responses_dir = Path(args.responses_dir)
    if not responses_dir.exists() or not list(responses_dir.glob("*.json")):
        print(f"No responses found in {responses_dir}/")
        print("Run the listening test first, then analyze.")
        sys.exit(1)

    responses = load_responses(responses_dir)
    print(f"Loaded {len(responses)} response files")

    config_path = Path(__file__).parent / "stimuli" / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    questions = [q["id"] for q in config["questions"]]
    conditions = [c["id"] for c in config["conditions"] if c["id"] != "mono"]

    data = extract_ratings(responses)
    print(f"Extracted ratings for {len(data)} scenes")

    # Descriptive stats
    desc = compute_descriptive(data, conditions, questions)
    for cond in conditions:
        print(f"\n  {cond}:")
        for q in questions:
            s = desc.get(cond, {}).get(q, {})
            if s:
                print(f"    {q}: median={s['median']} IQR={s['iqr']} (n={s['n']})")

    # Pairwise (proposed vs baseline)
    pairwise = {}
    if len(conditions) >= 2:
        pairwise = compute_pairwise(data, conditions[0], conditions[1], questions)
        if pairwise:
            print(f"\n  Proposed vs Baseline:")
            for q, pw in pairwise.items():
                wr = pw["win_rate"]
                print(
                    f"    {q}: δ={pw['cliffs_delta']} ({pw['effect_size']}), "
                    f"win={wr.get(f'{conditions[0]}_pct', 0)}%"
                )

    # Motion category breakdown
    by_motion = compute_by_motion(data, config["scenes"], conditions, questions)

    # Assemble
    analysis = {
        "n_participants": len(responses),
        "n_clips": len(config["scenes"]),
        "participants": [r["participant_id"] for r in responses],
        "questions": questions,
        "conditions": conditions,
        "descriptive": desc,
        "pairwise": pairwise,
        "by_motion": by_motion,
    }

    generate_report(analysis, Path(__file__).parent / "results")


if __name__ == "__main__":
    main()
