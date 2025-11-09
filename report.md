# Low-Rank Adaptation for Shakespeare Style Transfer
## An Experiment on Parameter Efficient Fine-Tuning and LLM-as-a-Judge Evaluation

---

## Experimental Setup

**Base Model**: Qwen/Qwen3-8B [^2]
**Dataset**: Alpaca instruction dataset [^1] with Shakespearean-style responses synthetically generated using Qwen-30B-A3B-2507 (4bit MLX) [^2].
**Training**: 3 epochs, batch size 16, learning rate 2e-4 (~4.4M tokens per run)

**Ablation Variables**:
- **LoRA Rank**: 4, 8, 16, 32, 64
- **Target Layers**: Attention-only, MLP-only, All layers

**Evaluation Metrics**:
- Training & test loss. 128 test examples per evaluation.
- LLM-as-Judge scores across four dimensions: Shakespearean intensity, appropriateness, helpfulness, clarity (10-point scale). 32 examples per evaluation.
- MMLU [^3] for general capabilities and knowledge preservation throughout style transfer fine-tuning. 128 examples per evaluation.

---

## Results
We summarize the effect of LoRA rank across the three target-module settings (ATTN-only, MLP-only, ALL). Each figure shows the evaluation trajectory over training steps, with one line per rank (r∈{4,8,16,32,64}).

### Test loss vs. rank
![Test loss by rank](results/rank_plots/test_loss_by_rank_over_steps.png)

- Loss decreases steadily with training and stabilizes near the final evaluations across all settings, with signs of overfitting for MLP-only and ALL towards the end.
- Rank has a modest effect: mid/high ranks (r=16–64) typically achieve the lowest loss, while r=4 trails slightly.
- MLP-only appears to outperform attention-only layer, with ALL showing a negligible improvement from MLP-only.

### MMLU vs. rank (scaled to 1–10; higher is better)
![MMLU by rank](results/rank_plots/mmlu_by_rank_over_steps.png)

- General capability measured by MMLU remains in a narrow band across ranks and modules.
- MMLU scores are noisy (±0.05-0.10) across and within runs. Admittedly, a higher sample size for MMLU could have been more informative. In the least, we can surmise that fine-tuning a model on Shakespearean data does not significantly degrade general capabilities insofar as MMLU is a good proxy for general capabilities.
- If anything, MLP-only seems to retain MMLU slightly better, but again, differences are small and noisy.

### LLM‑as‑Judge (Style vs. Content) by rank
![Judge pairs by rank](results/rank_plots/judge_pairs_by_rank_over_steps.png)

- Two curves per rank:
  - “Style” = average score of Intensity and Clarity.
  - “Content” = average score of Helpfulness and Appropriateness.
- Style scores improves significantly after the first evaluation interval (2,048 examples), while content scores stay within a band.
- Similar to MMLU, the LLM-as-Judge scores are noisy (±0.05-0.10) across and within runs. Though this time, the variance is more problematic as it is difficult to draw evaluative conclusions from these trends. It is unclear whether 1) the Shakespearean style transfer improved past the first evaluation interval, 2) there is significant performance difference across ranks and LoRA target modules.

### Comparing initial vs final evaluations
![Radar (final eval)](results/rank_plots/radar_final.png)

- Compares initial (step=0) vs. final evaluation (step=1796) for representative ranks to visualize trade‑offs among Style metrics and MMLU (all on 1–10 scale), for ranks r=4 and r=64.
- Across all three target modules, final evaluations show consistent gains in intensity and appropriateness with a slight decrease in helpfulness. MMLU scores do not degrade significantly.
- Going from r=4 to r=64 does not significantly improve any of the metrics.

---

## Discussion

### 1. The LoRA model achieves Shakespearean style transfer while retaining general capabilities, though its responses become less verbose and helpful.

Chatting with the mlp-only r=16 final checkpoint as an example, we observe that the model is quite convincingly Shakespearean in style. The model also preserves decent levels of general capabilities such as coding, and is capable of responding to general queries like trip-planning, though 


### 2. Layer Selection Matters More Than Rank: Rank = 16 is good enough for practical efficiency. MLP-only performs better than attention-only, and all-layers does not provide significant incremental improvements MLP-only.

Across all three target modules, we observe that varying LoRA rank does not significantly impact MMLU and LLM-as-Judge scores. For training loss, rank does have a slightly more notable effect, especially when we move from r=8 to r=16 for attention-only, and r=4 to r=8 for MLP-only/all-layers. Still, the magnitude of the effect is modest as we increase rank, which is consistent with existing literature.

![Training Curves](results/rank_plots/train_loss_by_rank_over_steps.png)

Comparing the effect of applying LoRA to different layers, we see that MLP-only configuration achieves better train and test loss than the attention-only configuration, whereas the all-layers configuration does not provide significant incremental improvements over MLP-only. This is consistent with the findings of [^4]. Though we do note here that, at least for the Qwen3-8B architecture, MLP-only LoRA introduces almost 2x the number of trainable parameters as the attention-only configuration. For example, r=16 MLP-only LoRA introduces 28M trainable parameters (0.34% of the total parameters), while r=16 attention-only LoRA introduces 15M trainable parameters (0.19%).

### 3. Challenges in implementing LLM-as-a-Judge

#### Noisy Scores

#### LLM-as-a-Judge Hallucinations

---

## Notes for Further Work

### How might we better compare the performance of different LoRA configurations leveraging LLM-as-a-judge?
- Pairwise comparison using model checkpoints, with an elo-like implementation
    - Observation that LLM judge tends to avoid giving extreme ends of the scale, and cluster around the mean

---
## Citations

[^1]: Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., & Hashimoto, T. B. (2023). Stanford Alpaca: An Instruction-following LLaMA model. GitHub repository. https://github.com/tatsu-lab/stanford_alpaca

[^2] Qwen Team, “Qwen3 Technical Report,” arXiv preprint, 2025. [Online]. Available: [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388).

[^3] Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). **Measuring Massive Multitask Language Understanding**. *Proceedings of the International Conference on Learning Representations (ICLR)*.

[^4]: Schulman, John and Thinking Machines Lab, "LoRA Without Regret", Thinking Machines Lab: Connectionism, Sep 2025.
