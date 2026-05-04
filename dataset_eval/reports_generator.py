"""
reports_generator.py — Detailed analysis reports for VLM evaluation.
Creates: master_report.csv, analysis charts, analysis_report.txt
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def _style():
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"

def _extract_pair(name):
    return int(os.path.splitext(name)[0][:-1])

# ═══════════════════════════════════════════════════════════════════════════
# 1. MASTER REPORT CSV
# ═══════════════════════════════════════════════════════════════════════════
def generate_master_report_csv(all_cls, all_gen, items, reports_dir):
    """One row per image, columns = each model's results."""
    if not all_cls:
        return None
    item_lookup = {i["number"]: i for i in items}
    df_cls = pd.concat(all_cls, ignore_index=True)
    df_cls["pair_number"] = df_cls["image_name"].apply(_extract_pair)

    # Build pivot: for each (pair_number, image_type, model) -> correct
    rows = []
    for (pn, it), grp in df_cls.groupby(["pair_number", "image_type"]):
        meta = item_lookup.get(pn, {})
        row = {
            "pair_number": pn,
            "image_name": grp["image_name"].iloc[0],
            "image_type": it,
            "grey_object": meta.get("grey_object", ""),
            "color_object": meta.get("color_object", ""),
            "quality": meta.get("quality", ""),
        }
        models_correct = 0
        n_models = 0
        for _, r in grp.iterrows():
            m = r["model"].replace(" ", "_").replace("-", "_").replace(".", "")
            row[f"{m}_correct"] = r["correct"]
            row[f"{m}_pred"] = r.get("pred_object", "")
            if "vlm_answer" in r and pd.notna(r.get("vlm_answer")):
                row[f"{m}_answer"] = r["vlm_answer"]
            if r["correct"]:
                models_correct += 1
            n_models += 1
        row["n_models_correct"] = models_correct
        row["n_models_total"] = n_models
        row["pct_correct"] = models_correct / n_models if n_models else 0
        if row["pct_correct"] >= 0.8:
            row["difficulty"] = "Easy"
        elif row["pct_correct"] >= 0.4:
            row["difficulty"] = "Medium"
        else:
            row["difficulty"] = "Hard"
        rows.append(row)

    # Add generation similarity columns if available
    if all_gen:
        df_gen = pd.concat(all_gen, ignore_index=True)
        df_gen["pair_number"] = df_gen["image_name"].apply(_extract_pair)
        gen_lookup = {}
        for _, r in df_gen.iterrows():
            key = (r["pair_number"], r["image_type"])
            m = r["model"].replace(" ", "_").replace("-", "_").replace(".", "")
            if key not in gen_lookup:
                gen_lookup[key] = {}
            gen_lookup[key][f"{m}_gen_guess"] = r.get("vlm_guess", "")
            gen_lookup[key][f"{m}_gen_sim"] = r.get("similarity_score", "")
        for row in rows:
            key = (row["pair_number"], row["image_type"])
            if key in gen_lookup:
                row.update(gen_lookup[key])

    df = pd.DataFrame(rows).sort_values(["pair_number", "image_type"])
    path = os.path.join(reports_dir, "master_report.csv")
    df.to_csv(path, index=False)
    print(f"  [OK] master_report.csv: {len(df)} rows, {len(df.columns)} columns")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. DETAILED ANALYSIS CHARTS
# ═══════════════════════════════════════════════════════════════════════════
def generate_analysis_charts(all_cls, all_gen, items, reports_dir):
    _style()
    if not all_cls:
        return
    df_cls = pd.concat(all_cls, ignore_index=True)
    df_cls["pair_number"] = df_cls["image_name"].apply(_extract_pair)
    models = sorted(df_cls["model"].unique())
    palette = sns.color_palette("husl", len(models))
    model_colors = dict(zip(models, palette))

    # ── 1. Difficulty distribution ────────────────────────────────────────
    pair_acc = df_cls.groupby(["pair_number", "image_type"])["correct"].mean().reset_index()
    pair_acc["difficulty"] = pd.cut(pair_acc["correct"], bins=[-0.01, 0.4, 0.8, 1.01],
                                     labels=["Hard", "Medium", "Easy"])
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = pair_acc["difficulty"].value_counts().reindex(["Easy", "Medium", "Hard"])
    bars = ax.bar(counts.index, counts.values, color=["#2ecc71", "#f39c12", "#e74c3c"], edgecolor="white")
    for b in bars:
        ax.annotate(f"{int(b.get_height())}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontweight="bold")
    ax.set_title("Image Difficulty Distribution Across All Models", fontweight="bold", fontsize=13)
    ax.set_ylabel("Number of Images")
    plt.savefig(os.path.join(reports_dir, "difficulty_distribution.png"))
    plt.close()

    # ── 2. Model ranking heatmap ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(models)*1.5), 6))
    heat_data = df_cls.groupby(["model", "image_type"])["correct"].mean().unstack()
    if "quality" in df_cls.columns:
        for q in ["L", "M", "H"]:
            sub = df_cls[df_cls["quality"] == q]
            if len(sub):
                heat_data[f"quality_{q}"] = sub.groupby("model")["correct"].mean()
    heat_data["overall"] = df_cls.groupby("model")["correct"].mean()
    heat_data = heat_data.reindex(models)
    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
                vmin=0, vmax=1, linewidths=0.5)
    ax.set_title("Model Performance Heatmap (Accuracy)", fontweight="bold", fontsize=13)
    plt.savefig(os.path.join(reports_dir, "model_ranking_heatmap.png"))
    plt.close()

    # ── 3. Hardest images ─────────────────────────────────────────────────
    img_acc = df_cls.groupby("image_name")["correct"].mean().sort_values()
    hardest = img_acc.head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(hardest)), hardest.values, color="#e74c3c", edgecolor="white")
    ax.set_yticks(range(len(hardest)))
    ax.set_yticklabels(hardest.index, fontsize=8)
    for i, v in enumerate(hardest.values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_title("20 Hardest Images (Lowest Avg Accuracy)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Accuracy (across all models)")
    ax.set_xlim(0, 1.1)
    plt.savefig(os.path.join(reports_dir, "hardest_images_analysis.png"))
    plt.close()

    # ── 4. Easiest images ─────────────────────────────────────────────────
    easiest = img_acc.tail(20).sort_values()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(easiest)), easiest.values, color="#2ecc71", edgecolor="white")
    ax.set_yticks(range(len(easiest)))
    ax.set_yticklabels(easiest.index, fontsize=8)
    for i, v in enumerate(easiest.values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_title("20 Easiest Images (Highest Avg Accuracy)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Accuracy (across all models)")
    ax.set_xlim(0, 1.1)
    plt.savefig(os.path.join(reports_dir, "easiest_images_analysis.png"))
    plt.close()

    # ── 5. Model agreement matrix ─────────────────────────────────────────
    if len(models) >= 2:
        # Build per-model correctness dict
        model_correct = {}
        for m in models:
            sub = df_cls[df_cls["model"] == m].set_index("image_name")["correct"]
            model_correct[m] = sub.astype(int)

        n = len(models)
        agree_data = np.ones((n, n))
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if m1 in model_correct and m2 in model_correct:
                    common = model_correct[m1].index.intersection(model_correct[m2].index)
                    if len(common) > 0:
                        agree_data[i, j] = (model_correct[m1].loc[common].values == model_correct[m2].loc[common].values).mean()

        agree = pd.DataFrame(agree_data, index=models, columns=models)
        fig, ax = plt.subplots(figsize=(max(8, len(models)*1.2), max(6, len(models))))
        sns.heatmap(agree, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    vmin=0.5, vmax=1, linewidths=0.5)
        ax.set_title("Pairwise Model Agreement Rate", fontweight="bold", fontsize=13)
        plt.savefig(os.path.join(reports_dir, "model_agreement_matrix.png"))
        plt.close()

    # ── 6. Grey vs color bias scatter ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    grey_acc = df_cls[df_cls["image_type"]=="grey"].groupby("model")["correct"].mean()
    color_acc = df_cls[df_cls["image_type"]=="color"].groupby("model")["correct"].mean()
    for m in models:
        if m in grey_acc.index and m in color_acc.index:
            ax.scatter(grey_acc[m], color_acc[m], s=120, label=m,
                       color=model_colors.get(m, "#999"), zorder=3)
            ax.annotate(m, (grey_acc[m], color_acc[m]), fontsize=8,
                        xytext=(5, 5), textcoords="offset points")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5, label="No bias")
    ax.set_xlabel("Grey Accuracy", fontsize=12)
    ax.set_ylabel("Color Accuracy", fontsize=12)
    ax.set_title("Grey vs Color Accuracy Bias\n(Above line = color bias, Below = grey bias)",
                 fontweight="bold", fontsize=13)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    plt.savefig(os.path.join(reports_dir, "grey_vs_color_bias_scatter.png"))
    plt.close()

    # ── 7. Per-model error breakdown ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    err_by_type = df_cls.groupby(["model", "image_type"])["correct"].apply(lambda x: 1 - x.mean()).unstack()
    err_by_type = err_by_type.reindex(models)
    err_by_type.plot(kind="bar", ax=axes[0], color={"grey": "#6B7B8D", "color": "#E8734A"})
    axes[0].set_title("Error Rate by Image Type", fontweight="bold")
    axes[0].set_ylabel("Error Rate")
    axes[0].tick_params(axis="x", rotation=45)
    if "quality" in df_cls.columns:
        err_by_q = df_cls.groupby(["model", "quality"])["correct"].apply(lambda x: 1 - x.mean()).unstack()
        err_by_q = err_by_q.reindex(models)
        q_cols = [c for c in ["L", "M", "H"] if c in err_by_q.columns]
        err_by_q[q_cols].plot(kind="bar", ax=axes[1], color={"L": "#F1C40F", "M": "#E67E22", "H": "#E74C3C"})
    axes[1].set_title("Error Rate by Quality Tier", fontweight="bold")
    axes[1].set_ylabel("Error Rate")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "per_model_error_analysis.png"))
    plt.close()

    # ── 8. Cumulative accuracy curve ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        sub = df_cls[df_cls["model"] == m]
        per_img = sub.groupby("image_name")["correct"].mean().sort_values().values
        ax.plot(np.linspace(0, 100, len(per_img)), per_img,
                label=m, color=model_colors.get(m, "#999"), linewidth=1.5)
    ax.set_xlabel("Image Percentile (sorted by accuracy)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cumulative Accuracy Curve Per Model", fontweight="bold", fontsize=13)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    plt.savefig(os.path.join(reports_dir, "cumulative_accuracy_curve.png"))
    plt.close()

    # ── 9. Radar chart ────────────────────────────────────────────────────
    try:
        categories = ["Grey Acc", "Color Acc", "Overall Acc", "Consistency", "Quality-H Acc"]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        overall = df_cls.groupby("model")["correct"].mean()
        consistency = 1 - df_cls.groupby(["model", "image_name"])["correct"].std().groupby("model").mean().fillna(0)
        h_acc = df_cls[df_cls["quality"]=="H"].groupby("model")["correct"].mean() if "quality" in df_cls.columns else overall
        for m in models:
            vals = [
                grey_acc.get(m, 0),
                color_acc.get(m, 0),
                overall.get(m, 0),
                consistency.get(m, 0),
                h_acc.get(m, 0),
            ] + [grey_acc.get(m, 0)]
            ax.plot(angles, vals, linewidth=1.5, label=m, color=model_colors.get(m, "#999"))
            ax.fill(angles, vals, alpha=0.05, color=model_colors.get(m, "#999"))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_title("Model Performance Radar", fontweight="bold", fontsize=13, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
        plt.savefig(os.path.join(reports_dir, "model_performance_radar.png"))
        plt.close()
    except Exception as e:
        print(f"  Warning: radar chart failed: {e}")

    print(f"  [OK] Generated analysis charts in {reports_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. TEXT ANALYSIS REPORT
# ═══════════════════════════════════════════════════════════════════════════
def generate_text_report(all_cls, all_gen, items, reports_dir):
    if not all_cls:
        return
    df_cls = pd.concat(all_cls, ignore_index=True)
    df_cls["pair_number"] = df_cls["image_name"].apply(_extract_pair)
    models = sorted(df_cls["model"].unique())

    lines = []
    w = lines.append
    w("=" * 80)
    w("DETAILED VLM EVALUATION ANALYSIS REPORT")
    w("=" * 80)
    w("")

    # ── 1. Executive summary ──────────────────────────────────────────────
    w("1. EXECUTIVE SUMMARY")
    w("-" * 40)
    overall = df_cls.groupby("model")["correct"].mean().sort_values(ascending=False)
    best_m = overall.index[0]
    worst_m = overall.index[-1]
    w(f"  Models evaluated: {len(models)}")
    w(f"  Dataset: {len(items)} image pairs, {len(df_cls)} total predictions")
    w(f"  Best performer:  {best_m} ({overall[best_m]:.3f} accuracy)")
    w(f"  Worst performer: {worst_m} ({overall[worst_m]:.3f} accuracy)")
    grey_all = df_cls[df_cls["image_type"]=="grey"]["correct"].mean()
    color_all = df_cls[df_cls["image_type"]=="color"]["correct"].mean()
    w(f"  Avg grey accuracy:  {grey_all:.3f}")
    w(f"  Avg color accuracy: {color_all:.3f}")
    bias = "greyscale" if grey_all > color_all else "color"
    w(f"  Overall cue bias: Models favor {bias} cues (delta: {abs(grey_all-color_all):.3f})")
    w("")

    # ── 2. Per-model performance ──────────────────────────────────────────
    w("2. PER-MODEL PERFORMANCE")
    w("-" * 40)
    grey_acc = df_cls[df_cls["image_type"]=="grey"].groupby("model")["correct"].mean()
    color_acc = df_cls[df_cls["image_type"]=="color"].groupby("model")["correct"].mean()
    w(f"  {'Model':<20} {'Overall':>8} {'Grey':>8} {'Color':>8} {'Delta':>8} {'Bias':>10}")
    w(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for m in overall.index:
        g = grey_acc.get(m, 0)
        c = color_acc.get(m, 0)
        d = g - c
        b = "Grey" if d > 0.02 else ("Color" if d < -0.02 else "Balanced")
        w(f"  {m:<20} {overall[m]:>8.3f} {g:>8.3f} {c:>8.3f} {d:>+8.3f} {b:>10}")
    w("")

    # generation summary
    if all_gen:
        df_gen = pd.concat(all_gen, ignore_index=True)
        w("  Generation (Semantic Similarity):")
        gen_overall = df_gen.groupby("model")["similarity_score"].mean().sort_values(ascending=False)
        for m in gen_overall.index:
            w(f"    {m:<20} avg_sim: {gen_overall[m]:.3f}")
        w("")

    # ── 3. Cue bias analysis ─────────────────────────────────────────────
    w("3. CUE BIAS ANALYSIS")
    w("-" * 40)
    for m in models:
        g = grey_acc.get(m, 0)
        c = color_acc.get(m, 0)
        if g > c + 0.05:
            w(f"  {m}: Strong GREYSCALE bias (+{g-c:.3f}). Model relies heavily on luminance cues.")
        elif c > g + 0.05:
            w(f"  {m}: Strong COLOR bias (+{c-g:.3f}). Model relies heavily on chromatic cues.")
        else:
            w(f"  {m}: Balanced ({g:.3f} grey vs {c:.3f} color).")
    w("")

    # ── 4. Difficulty analysis ────────────────────────────────────────────
    w("4. DIFFICULTY ANALYSIS")
    w("-" * 40)
    img_acc = df_cls.groupby("image_name")["correct"].mean()
    hard = img_acc[img_acc < 0.4]
    easy = img_acc[img_acc >= 0.8]
    medium = img_acc[(img_acc >= 0.4) & (img_acc < 0.8)]
    w(f"  Easy images (>80% models correct):   {len(easy)}")
    w(f"  Medium images (40-80%):              {len(medium)}")
    w(f"  Hard images (<40%):                  {len(hard)}")
    w("")
    if len(hard) > 0:
        w("  Hardest images (all models struggle):")
        item_lookup = {i["number"]: i for i in items}
        for name in hard.sort_values().head(10).index:
            pn = _extract_pair(name)
            meta = item_lookup.get(pn, {})
            w(f"    {name}: acc={img_acc[name]:.2f}, grey={meta.get('grey_object','?')}, "
              f"color={meta.get('color_object','?')}, quality={meta.get('quality','?')}")
    w("")

    # ── 5. Architecture comparison ────────────────────────────────────────
    w("5. ARCHITECTURE COMPARISON")
    w("-" * 40)
    arch_map = {
        "CLIP": "Contrastive", "SigLIP": "Contrastive", "ALIGN": "Contrastive",
        "BLIP-2": "Generative (Q-Former)",
        "LLaVA-1.5": "Instruction-Tuned LLM", "LLaVA-1.6": "Instruction-Tuned LLM",
        "GPT-4o-mini": "Proprietary API", "GPT-5.5": "Proprietary API",
        "SmolVLM": "Compact Multimodal (Idefics3)",
        "Qwen2-VL": "Instruction-Tuned (Dynamic Res)",
        "Moondream2": "Compact Generative VLM",
    }
    arch_acc = {}
    for m in models:
        arch = arch_map.get(m, "Unknown")
        arch_acc.setdefault(arch, []).append(overall.get(m, 0))
    for arch, accs in sorted(arch_acc.items(), key=lambda x: -np.mean(x[1])):
        w(f"  {arch}: avg accuracy = {np.mean(accs):.3f} (n={len(accs)} models)")
    w("")

    # ── 6. Quality tier analysis ──────────────────────────────────────────
    w("6. QUALITY TIER ANALYSIS")
    w("-" * 40)
    if "quality" in df_cls.columns:
        for q in ["L", "M", "H"]:
            sub = df_cls[df_cls["quality"] == q]
            if len(sub):
                qa = sub["correct"].mean()
                w(f"  Quality {q}: {qa:.3f} accuracy ({len(sub)} predictions)")
        w("  (L=Low quality illusion=easy to see through, H=High quality=hard to see through)")
    w("")

    # ── 7. Agreement analysis ─────────────────────────────────────────────
    w("7. MODEL AGREEMENT ANALYSIS")
    w("-" * 40)
    if len(models) >= 2:
        preds = df_cls.pivot_table(index="image_name", columns="model", values="correct", aggfunc="first")
        unanim = (preds.sum(axis=1) == len(models)) | (preds.sum(axis=1) == 0)
        w(f"  Unanimous agreement: {unanim.mean()*100:.1f}% of images")
        full_agree = preds.sum(axis=1) == len(models)
        full_disagree = preds.sum(axis=1) == 0
        w(f"    All correct: {full_agree.sum()} images")
        w(f"    All wrong:   {full_disagree.sum()} images")
    w("")

    # ── 8. Key findings ──────────────────────────────────────────────────
    w("8. KEY FINDINGS & RECOMMENDATIONS")
    w("-" * 40)
    w(f"  1. {best_m} is the best-performing model overall ({overall[best_m]:.3f}).")
    w(f"  2. Models generally show {'greyscale' if grey_all > color_all else 'color'} cue bias.")
    w(f"  3. {len(hard)} images are universally difficult (<40% accuracy).")
    w(f"  4. {len(easy)} images are universally easy (>80% accuracy).")
    if "quality" in df_cls.columns:
        h_acc_val = df_cls[df_cls["quality"]=="H"]["correct"].mean()
        l_acc_val = df_cls[df_cls["quality"]=="L"]["correct"].mean()
        w(f"  5. High-quality illusions are {'harder' if h_acc_val < l_acc_val else 'easier'} "
          f"(H={h_acc_val:.3f} vs L={l_acc_val:.3f}).")
    w("")
    w("=" * 80)
    w("END OF REPORT")
    w("=" * 80)

    path = os.path.join(reports_dir, "analysis_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OK] analysis_report.txt written ({len(lines)} lines")


def generate_all_reports(all_cls, all_gen, items, reports_dir):
    """Entry point: generate CSV, charts, and text report."""
    os.makedirs(reports_dir, exist_ok=True)
    print(f"\n{'='*60}\n  Generating Detailed Reports\n{'='*60}")
    generate_master_report_csv(all_cls, all_gen, items, reports_dir)
    generate_analysis_charts(all_cls, all_gen, items, reports_dir)
    generate_text_report(all_cls, all_gen, items, reports_dir)
    print(f"  [OK] All reports saved to: {reports_dir}")
