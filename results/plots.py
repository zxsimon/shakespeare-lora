from dataclasses import dataclass, field
import json, os, sys, code
import math
import pandas as pd
from matplotlib import pyplot as plt

logs_dir = "logs"
results_dir = "results"
project_name = "shakespeare-lora"

@dataclass
class Results:
    config: dict = field(default_factory=dict)
    train_iter: list = field(default_factory=list)
    test_iter: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    test_loss: list = field(default_factory=list)
    test_llmjudge: list = field(default_factory=list)
    test_mmlu: list = field(default_factory=list)
    
def parse_log(dataset, lora_target_modules, lora_r, lora_alpha):
    results = Results()
    log_file_name = f"{project_name}_{dataset}-{lora_target_modules}-{lora_r}-{lora_alpha}.jsonl"
    filepath = os.path.join(logs_dir, log_file_name)

    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)
            log = data["log"]
            if data["type"] == "config":
                results.config = log
            elif data["type"] == "step":
                results.train_iter.append(log['iter'])
                results.train_loss.append(log['train_loss'])
            elif data["type"] == "eval":
                results.test_iter.append(log['iter'])
                results.test_loss.append(log['test_loss'])
                results.test_llmjudge.append(log['llmjudge_score'])
                results.test_mmlu.append(log['mmlu_score'])

    results.config["lora_target_modules"] = lora_target_modules
    results.config["lora_r"] = lora_r
    results.config["lora_alpha"] = lora_alpha
    
    return results

def plot_single_run(results, name = "training_curves", save_path = None, smooth_window = 101):

    has_llmjudge = len(results.test_llmjudge) > 0

    plt.style.use("default")
    fig, ax_loss = plt.subplots(figsize=(10, 6))

    handles, labels = [], []
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_ylim(0.3, 1.5)
    ax_loss.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_score = ax_loss.twinx()
    ax_score.set_ylabel("MMLU/LLM-as-a-Judge Score")
    ax_score.set_ylim(0, 100)

    # Train Loss (raw)
    ax_loss.plot(
        results.train_iter,
        results.train_loss,
        color="#1f77b4",
        linewidth=0.8,
        alpha=0.20,
    )
    # Train Loss (smoothed)
    tl_sm = pd.Series(results.train_loss).rolling(
        window=max(3, smooth_window), min_periods=1, center=True
    ).mean().tolist()
    h1 = ax_loss.plot(
        results.train_iter,
        tl_sm,
        label="Train Loss (smoothed)",
        color="#1f77b4",
        linewidth=2.0,
    )[0]
    handles.append(h1)
    labels.append("Train Loss (smoothed)")

    # Test Loss
    step = max(1, len(results.test_iter) // 20)
    h2 = ax_loss.plot(
        results.test_iter,
        results.test_loss,
        label="Test Loss",
        color="#ff7f0e",
        linewidth=2.6,
        linestyle="--",
        marker="o",
        markersize=4,
        markevery=step,
        markerfacecolor="white",
        markeredgewidth=0.8,
    )[0]
    handles.append(h2)
    labels.append("Test Loss")

    # MMLU Score
    mmlu_norm = [x * 100.0 for x in results.test_mmlu]
    step = max(1, len(mmlu_norm) // 20)
    h3 = ax_score.plot(
        results.test_iter[: len(mmlu_norm)],
        mmlu_norm,
        label="MMLU",
        color="#2ca02c",
        linewidth=2.0,
        linestyle="-.",
        marker="s",
        markersize=4,
        markevery=step,
        alpha=0.9,
    )[0]
    handles.append(h3)
    labels.append("MMLU")
    
    # LLM-as-a-Judge Scores
    if has_llmjudge:
        judge_dicts = [d for d in results.test_llmjudge if isinstance(d, dict)]
        judge_keys = sorted({k for d in judge_dicts for k in d.keys()})
        key_styles = {
            "intensity": ("#d62728", "^"),
            "helpfulness": ("#9467bd", "v"),
            "clarity": ("#8c564b", "<"),
            "appropriateness": ("#e377c2", ">"),
        }
        step = max(1, len(results.test_iter) // 20)
        for k in judge_keys:
            vals = [d.get(k) if isinstance(d, dict) else None for d in results.test_llmjudge]
            vals = [v * 10.0 if isinstance(v, (int, float)) else None for v in vals]
            col, mark = key_styles.get(k, (None, None))
            h = ax_score.plot(
                results.test_iter[: len(vals)],
                vals,
                label=f"Judge {k}",
                color=col,
                linewidth=1.8,
                linestyle=":",
                marker=mark or "o",
                markersize=4,
                markevery=step,
                alpha=0.9,
            )[0]
            handles.append(h)
            labels.append(f"LLM-as-a-Judge: {k}")

    # Title and Legend
    cfg = results.config or {}
    title_parts = [
        cfg.get("model", "model=unknown"),
        cfg.get("dataset", "dataset=unknown"),
    ]
    lora_bits = []
    lora_bits.append(f"r={cfg['lora_r']}")
    lora_bits.append(f"alpha={cfg['lora_alpha']}")
    lora_bits.append(f"targets={cfg['lora_target_modules']}")
    title_parts.append("LoRA(" + ", ".join(lora_bits) + ")")
    ax_loss.set_title(" | ".join(title_parts))
    ax_loss.legend(handles, labels, loc="lower left", frameon=False, ncol=2)
    fig.tight_layout()

    # Save Plot
    if save_path is None:
        os.makedirs(os.path.join(results_dir, "single_run_plots"), exist_ok=True)
        save_path = os.path.join(results_dir, "single_run_plots", f"{name}.png")
    try:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to: {save_path}")
    except Exception:
        pass



def parse_all_logs(dataset = "alpaca", selected_alpha = 32):
    # Assume a constant alpha, for now
    
    results = {}
    log_files = os.listdir(logs_dir)
    for log_file in log_files:
        if log_file.startswith(f"{project_name}_{dataset}"):
            params = log_file.split(".")[0].split("_")[-1]
            _, lora_target_modules, lora_r, lora_alpha = params.split("-")
            if int(lora_alpha) != selected_alpha:
                continue
            result = parse_log(dataset, lora_target_modules, int(lora_r), int(lora_alpha))

            if lora_target_modules not in results:
                results[lora_target_modules] = {}
            results[lora_target_modules][lora_r] = result
    
    return results


def _plot_steps_by_rank_figure(all_results, series_extractor, ylabel, save_name, xlabel: str = "Iteration", xscale: str | None = None, ylim: tuple | None = None, xticks: list | None = None):
    # For each target module, plot iteration on x-axis and multiple rank lines
    plt.style.use("default")
    modules = ["attn", "mlp", "all"]
    present = [m for m in modules if m in all_results]
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 4), sharey=False)
    if len(present) == 1:
        axes = [axes]
    for ax, mod in zip(axes, present):
        rank_to_res = all_results[mod]
        ranks = sorted(int(r) for r in rank_to_res.keys())
        any_plotted = False
        for r in ranks:
            key = str(r) if str(r) in rank_to_res else r
            xs, ys = series_extractor(rank_to_res[key])
            if not xs or not ys:
                continue
            ax.plot(xs, ys, linewidth=2.0, label=f"LoRA Rank {r}")
            any_plotted = True
        ax.set_title(f"{mod.upper()} Layers")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
        if any_plotted:
            ax.legend(frameon=False)
        if xscale:
            ax.set_xscale(xscale)
        if ylim:
            ax.set_ylim(*ylim)
        if xticks:
            ax.set_xticks(xticks)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    os.makedirs(os.path.join(results_dir, "rank_plots"), exist_ok=True)
    out_path = os.path.join(results_dir, "rank_plots", f"{save_name}.png")
    try:
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to: {out_path}")
    except Exception:
        pass


def plot_test_loss_vs_rank(all_results):
    def _series(res):
        xs, ys = res.test_iter, res.test_loss
        # drop iteration 0 to allow log x-scale and ignore initial spike
        if xs and xs[0] == 0:
            xs = xs[1:]
            ys = ys[1:]
        return (xs, ys)
    _plot_steps_by_rank_figure(
        all_results,
        series_extractor=_series,
        ylabel="Test Loss",
        save_name="test_loss_by_rank_over_steps",
        xlabel="Step",
        xscale="log",
        ylim=(0.80, 1.22),
        xticks=[150, 300, 600, 1200, 1796],
    )


def plot_mmlu_vs_rank(all_results):
    def _series(res):
        if not res.test_mmlu:
            return ([], [])
        ys = [x * 100.0 for x in res.test_mmlu]
        xs = res.test_iter[: len(ys)]
        return (xs, ys)
    _plot_steps_by_rank_figure(
        all_results,
        series_extractor=_series,
        ylabel="MMLU (%)",
        save_name="mmlu_by_rank_over_steps",
        xlabel="Step",
        ylim=(1, 100),
    )


def plot_judge_overall_vs_rank(all_results):
    # Two lines per rank: avg(intensity, clarity) and avg(helpfulness, appropriateness)
    plt.style.use("default")
    modules = ["attn", "mlp", "all"]
    present = [m for m in modules if m in all_results]
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 4), sharey=False)
    if len(present) == 1:
        axes = [axes]
    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = color_cycle.by_key()["color"] if color_cycle else ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for ax, mod in zip(axes, present):
        rank_to_res = all_results[mod]
        ranks = sorted(int(r) for r in rank_to_res.keys())
        any_plotted = False
        for idx, r in enumerate(ranks):
            key = str(r) if str(r) in rank_to_res else r
            res = rank_to_res[key]
            if not res.test_llmjudge:
                continue
            xs_raw = res.test_iter[: len(res.test_llmjudge)]
            xs_a, ys_a, xs_b, ys_b = [], [], [], []
            for x, d in zip(xs_raw, res.test_llmjudge):
                if not isinstance(d, dict):
                    continue
                iv, cv = d.get("intensity"), d.get("clarity")
                hv, av = d.get("helpfulness"), d.get("appropriateness")
                if isinstance(iv, (int, float)) and isinstance(cv, (int, float)):
                    xs_a.append(x); ys_a.append((iv + cv) / 2.0)
                if isinstance(hv, (int, float)) and isinstance(av, (int, float)):
                    xs_b.append(x); ys_b.append((hv + av) / 2.0)
            color = colors[idx % len(colors)]
            if xs_a and ys_a:
                ax.plot(xs_a, ys_a, color=color, linestyle="-", linewidth=2.0, label=f"LoRA Rank {r}, Style Score")
                any_plotted = True
            if xs_b and ys_b:
                ax.plot(xs_b, ys_b, color=color, linestyle="--", linewidth=2.0, label=f"LoRA Rank {r}, Content Score")
                any_plotted = True
        ax.set_title(f"{mod.upper()} Layers")
        ax.set_xlabel("Step")
        ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_ylim(5.5, 8.5)
        # defer legend to a shared, figure-level legend
    axes[0].set_ylabel("LLM-as-Judge Overall (1-10)")
    # Shared legend outside at the bottom across all subplots
    handles_all, labels_all = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles_all.extend(h)
        labels_all.extend(l)
    # Deduplicate while preserving order
    seen = set()
    handles_dedup, labels_dedup = [], []
    for h, l in zip(handles_all, labels_all):
        if l and l not in seen:
            seen.add(l)
            handles_dedup.append(h)
            labels_dedup.append(l)
    if labels_dedup:
        ncol = max(1, math.ceil(len(labels_dedup) / 2))  # target ~2 rows
        fig.legend(
            handles_dedup,
            labels_dedup,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
            ncol=ncol,
        )
        fig.subplots_adjust(bottom=0.20)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    os.makedirs(os.path.join(results_dir, "rank_plots"), exist_ok=True)
    out_path = os.path.join(results_dir, "rank_plots", "judge_pairs_by_rank_over_steps.png")
    try:
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to: {out_path}")
    except Exception:
        pass


def plot_train_loss_vs_rank(all_results, smooth_window: int = 101, ylim: tuple | None = (0.2, 1.8)):
    plt.style.use("default")
    modules = ["attn", "mlp", "all"]
    present = [m for m in modules if m in all_results]
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 4), sharey=False)
    if len(present) == 1:
        axes = [axes]
    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = color_cycle.by_key()["color"] if color_cycle else ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for ax, mod in zip(axes, present):
        rank_to_res = all_results[mod]
        ranks = sorted(int(r) for r in rank_to_res.keys())
        for idx, r in enumerate(ranks):
            key = str(r) if str(r) in rank_to_res else r
            res = rank_to_res[key]
            xs, ys = res.train_iter, res.train_loss
            if not xs or not ys:
                continue
            # keep only from iteration >= 10 (avoid early artifacts and log(0))
            filtered = [(x, y) for x, y in zip(xs, ys) if x is not None and x >= 10]
            if not filtered:
                continue
            xs, ys = zip(*filtered)
            xs, ys = list(xs), list(ys)
            color = colors[idx % len(colors)]
            # raw (faint)
            ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.25)
            # smoothed (bold)
            ys_sm = pd.Series(ys).rolling(window=max(3, smooth_window), min_periods=1, center=True).mean().tolist()
            ax.plot(xs, ys_sm, color=color, linewidth=2.0, label=f"LoRA Rank {r}")
        ax.set_title(f"{mod.upper()} Layers")
        ax.set_xlabel("Step (log scale)")
        ax.set_xscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Train Loss")
    fig.tight_layout()
    os.makedirs(os.path.join(results_dir, "rank_plots"), exist_ok=True)
    out_path = os.path.join(results_dir, "rank_plots", "train_loss_by_rank_over_steps.png")
    try:
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to: {out_path}")
    except Exception:
        pass

def plot_radar_final(all_results):
    plt.style.use("default")
    modules = ["attn", "mlp", "all"]
    present = [m for m in modules if m in all_results]
    if not present:
        print("No results to plot.")
        return
    
    categories = ["Intensity", "Clarity", "Helpfulness", "Appropriateness", "MMLU"]
    K = len(categories)
    angles = [2 * math.pi * i / K for i in range(K)]
    
    fig, axes = plt.subplots(
        1, len(present), figsize=(6 * len(present), 5), subplot_kw=dict(polar=True)
    )
    if len(present) == 1:
        axes = [axes]
    
    # Compare only r=4 and r=64 (colors from default Matplotlib cycle for consistency)
    ranks_to_compare = [4, 64]
    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    _cycle_colors = color_cycle.by_key()["color"] if color_cycle else ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    rank_colors = {r: _cycle_colors[i % len(_cycle_colors)] for i, r in enumerate(sorted(ranks_to_compare))}
    
    for ax, mod in zip(axes, present):
        rank_to_res = all_results[mod]
        
        for r in ranks_to_compare:
            key = str(r) if str(r) in rank_to_res else r
            if key not in rank_to_res:
                continue
            
            res = rank_to_res[key]
            color = rank_colors[r]
            
            # Get first evaluation (skip iter=0 if it exists, use first real eval)
            first_j = None
            first_mmlu = None
            for i, d in enumerate(res.test_llmjudge):
                if isinstance(d, dict):
                    first_j = d
                    first_mmlu = res.test_mmlu[i] if i < len(res.test_mmlu) else None
                    break
            
            # Get last evaluation
            last_j = next((d for d in reversed(res.test_llmjudge) if isinstance(d, dict)), None)
            last_mmlu = res.test_mmlu[-1] if res.test_mmlu else None
            
            if not (first_j and first_mmlu is not None and last_j and last_mmlu is not None):
                continue
            
            # First eval values
            vals_first = [
                first_j.get("intensity"),
                first_j.get("clarity"),
                first_j.get("helpfulness"),
                first_j.get("appropriateness"),
                first_mmlu * 10.0,
            ]
            
            # Last eval values
            vals_last = [
                last_j.get("intensity"),
                last_j.get("clarity"),
                last_j.get("helpfulness"),
                last_j.get("appropriateness"),
                last_mmlu * 10.0,
            ]
            
            if any(v is None for v in vals_first + vals_last):
                continue
            
            # Close the loops
            vals_first_c = vals_first + [vals_first[0]]
            vals_last_c = vals_last + [vals_last[0]]
            ang_c = angles + [angles[0]]
            
            # Plot first eval (dashed, lighter, thinner)
            ax.plot(ang_c, vals_first_c, 
                   linewidth=1.5, 
                   linestyle='--', 
                   alpha=0.5,
                   color=color, 
                    label=f"LoRA Rank {r} Initial")
            ax.fill(ang_c, vals_first_c, alpha=0.05, color=color)
            
            # Plot last eval (solid, bolder)
            ax.plot(ang_c, vals_last_c, 
                   linewidth=2.5, 
                   linestyle='-',
                   alpha=0.9,
                   color=color, 
                    label=f"LoRA Rank {r} Final")
            ax.fill(ang_c, vals_last_c, alpha=0.15, color=color)
        
        ax.set_title(f"{mod.upper()} Layers", fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_rlabel_position(0)
        ax.set_ylim(1, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.grid(True, which="both", linestyle=":", linewidth=1.0, alpha=0.6)
        if "polar" in ax.spines:
            ax.spines["polar"].set_color("#888")
            ax.spines["polar"].set_linewidth(1.2)
    
    # Shared legend at bottom
    handles_all, labels_all = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles_all.extend(h)
        labels_all.extend(l)
    
    # Deduplicate legend entries
    seen = set()
    handles_dedup, labels_dedup = [], []
    for h, l in zip(handles_all, labels_all):
        if l and l not in seen:
            seen.add(l)
            handles_dedup.append(h)
            labels_dedup.append(l)
    
    if labels_dedup:
        ncol = max(1, math.ceil(len(labels_dedup) / 2))  # ~2 rows
        fig.legend(
            handles_dedup,
            labels_dedup,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
            ncol=ncol,
        )
        fig.subplots_adjust(bottom=0.20)
    
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    
    os.makedirs(os.path.join(results_dir, "rank_plots"), exist_ok=True)
    out_path = os.path.join(results_dir, "rank_plots", "radar_final.png")
    try:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {out_path}")
    except Exception:
        pass
    


all_results = parse_all_logs()
plot_test_loss_vs_rank(all_results)
plot_mmlu_vs_rank(all_results)
plot_judge_overall_vs_rank(all_results)
plot_radar_final(all_results)
plot_train_loss_vs_rank(all_results)


dataset = "alpaca"
lora_target_modules = "mlp"
lora_r = 16
lora_alpha = 32

results = parse_log(dataset, lora_target_modules, lora_r, lora_alpha)
plot_single_run(results, name = f"{dataset}-{lora_target_modules}-{lora_r}-{lora_alpha}")


plt.show()