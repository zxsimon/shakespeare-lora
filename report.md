# Low-Rank Adaptation for Shakespeare Style Transfer
An Experiment on Parameter-Efficient Fine-Tuning and LLM-as-a-Judge Evaluation

## Experimental Setup

**Base Model**: Qwen/Qwen3-8B [^1]  
**Dataset**: Stanford Alpaca [^2], first 10,000 examples. Assistant responses rewritten in Shakespearean style using Qwen‑30B‑A3B‑2507 (4‑bit MLX).  
**Training**: 3 epochs, batch size 16, LR 2e-4 (~4.4M tokens/run). ~1 hour per run on a single NVIDIA H200 SXM

**Ablation Variables**:
- **LoRA Rank**: 4, 8, 16, 32, 64
- **Target Layers**: Attention-only, MLP-only, All layers

**Evaluation Metrics**:
- Training & test loss. 128 test examples per evaluation.
- LLM‑as‑a‑Judge (Qwen‑30B‑A3B‑2507 scoring outputs) across four dimensions: Shakespearean intensity, appropriateness, helpfulness, clarity (10‑point scale). Conversations are generated with Smoltalk [^3]. 32 examples per evaluation.
- MMLU [^4] to track general‑capability retention during fine‑tuning. 128 examples per evaluation.

---

## Results
We summarize LoRA rank effects across attention‑only, MLP‑only, and all‑layers. Each plot shows evaluation over ~1,800 training steps (~3 epochs) with one line per rank (r∈{4,8,16,32,64}).

### Test loss vs. rank
![Test loss by rank](results/rank_plots/test_loss_by_rank_over_steps.png)

- Loss decreases steadily and stabilizes. Overfitting appears late in training for MLP‑only and all‑layers.
- Rank has a modest effect: r=16–64 slightly outperform r=4.
- MLP‑only slightly beats attention‑only; all‑layers adds negligible gains.

### MMLU vs. rank (scaled to 1–10; higher is better)
![MMLU by rank](results/rank_plots/mmlu_by_rank_over_steps.png)

- General capability (MMLU) remains in a narrow band across ranks and modules.
- Scores are noisy (±0.05–0.10) within/between runs; larger eval sets would help. Importantly, we observe no significant degradation from style fine‑tuning.
- If anything, MLP‑only retains MMLU slightly better, but differences are small and noisy.

### LLM‑as‑Judge (Style vs. Content) by rank
![Judge pairs by rank](results/rank_plots/judge_pairs_by_rank_over_steps.png)

- Two curves per rank:
  - “Style” = average score of Intensity and Clarity.
  - “Content” = average score of Helpfulness and Appropriateness.
- Style scores improve noticeably after the first evaluation (~2,048 examples); content scores remain relatively stable.
- Given the variance (±0.05–0.10), we cannot draw strong conclusions across ranks or modules beyond the early improvement in style.

### Comparing initial vs final evaluations
![Radar (final eval)](results/rank_plots/radar_final.png)

- Initial (step=0) vs. final (step=1796) for r=4 and r=64 highlights trade‑offs among style metrics and MMLU (all on 1–10).
- Across modules, style (intensity/appropriateness) improves; helpfulness dips slightly; MMLU remains stable.
- Increasing rank from 4 to 64 yields minimal gains across metrics.

---

## Findings and Discussion

### 1. LoRA fine-tuning achieves Shakespearean style transfer while largely retaining capabilities for short conversations; responses become briefer and sometimes less helpful.

Using the MLP‑only r=16 checkpoint as an example, the model is convincingly Shakespearean. Observations include:

**Capability preservation**: Solves basic math, provides simple correct code with explanations, and answers general queries.

**Style-helpfulness trade-off**: Strong style sometimes reduces utility.
- Occasionally mimics play-script formatting (short lines); mitigated by prompting "no line breaks."
- Cannot switch back to modern English when prompted.

**Response brevity**: Outputs shorter than base model; likely attributable to training data, which averages ~160 tokens/example (std ≈ 70).

### 2. Layer selection matters more than rank; rank 16 is a strong default
MLP‑only outperforms attention‑only on train/test loss; all‑layers offers limited incremental gains over MLP‑only.

Across modules, varying rank does not materially change MMLU or LLM‑as‑a‑Judge scores. Rank affects training loss more (notably r=8→16 for attention‑only, r=4→8 for MLP‑only/all‑layers), but gains taper with larger ranks. Since these differences don't translate to test loss or downstream improvements, higher ranks may primarily enable overfitting rather than better generalization.

![Training Curves](results/rank_plots/train_loss_by_rank_over_steps.png)

MLP‑only achieves better train/test loss than attention‑only; all‑layers adds little beyond MLP‑only. This observation echoes the findings of [^5]. Importantly, on Qwen3‑8B, MLP‑only LoRA uses ~28M trainable params at r=16 (0.34%) vs 15M for attention‑only (0.19%), so gains should be weighed against added capacity.

### 3. LLM-as-a-Judge is useful but difficult to implement reliably

#### LLM-as-a-Judge Hallucinations
Looking at some of the LLM-as-a-Judge logs, we observe that the judge model hallucinates when providing reasoning for its scores. In the example below, the judge claims the response uses "archaic pronouns (thou, thee)" and "inverted syntax" - none of which appear in the actual response. The judge may be anchoring on its rubric rather than the response, 
essentially "reasoning backward" from a score to justify it. This pattern persisted across judge models (including GPT-OSS-20B). Perhaps this could be mitigated by using better/larger models or better prompt engineering.

Example LLM-as-a-Judge log:
```json
{"intensity": 6, "helpfulness": 9, "clarity": 8, "appropriateness": 7, "reasoning": "The response applies moderate Shakespearean style with consistent use of archaic pronouns (thou, thee), inverted syntax, and period-appropriate phrasing, but lacks advanced rhetorical devices and sophisticated metaphors. It remains highly helpful and accurate, clearly outlining the task-completion process. Clarity is well-maintained for modern readers, though the style occasionally interrupts flow. The intensity is appropriate for a procedural request but slightly heavier than ideal for a neutral instruction-following task, where a lighter touch would better suit the context.", "prompt": "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.", "response": "Certainly! Please provide me with the task you'd like me to complete. I'll work through it step by step and explain my reasoning as I go."}
```

#### Noisy Scores
As noted earlier, the LLM-as-Judge scores are noisy across and within runs. This curtails the effectiveness of this evaluation metric, beyond the (not so insightful) conclusion that LoRA fine-tuning does lead to some degree of style transfer. Granted, there are many sources of variance embedded within these scores, from the diversity of dialogues generated by the Smoltalk dataset to potential inconsistencies between LLM-as-Judge evaluations. Again, increasing evaluation sample size and quantifying standard errors could help in denoising this metric. Even so, the LLM judge seems to systematically avoid giving extreme ends of the scale, and cluster around the mean. As with the earlier point on hallucinations, this could be a more fundamental issue that warrants better prompt engineering and/or evaluation design. Perhaps we could consider alternative evaluation methods like reward modeling, or a pairwise comparison using model checkpoints, with an elo-like implementation.

---
## Sample Conversations

**Simple arithmetic** (correctly solved)
<p><img src="results/sample_conversations/math1.png" alt="Simple math problem" width="960"></p>

**Harder problem** (both models correct; Claude Sonnet 4.5 initially failed)
<p><img src="results/sample_conversations/math2.png" alt="Harder math problem" width="960"></p>

**Trip planning** (brief but similar to base)
<p><img src="results/sample_conversations/trip_planner.png" alt="Trip planner conversation" width="960"></p>

**Rust snippet** (correct, concise)
<p><img src="results/sample_conversations/rust.png" alt="Rust coding request" width="960"></p>

---
## Citations

[^1]: Qwen Team, “Qwen3 Technical Report,” arXiv preprint, 2025. [Online]. Available: [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388).

[^2]: Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., & Hashimoto, T. B. (2023). Stanford Alpaca: An Instruction-following LLaMA model. GitHub repository. https://github.com/tatsu-lab/stanford_alpaca

[^3]: L. B. Allal et al., “SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model,” arXiv preprint, 2025. [Online]. Available: [https://arxiv.org/abs/2502.02737](https://arxiv.org/abs/2502.02737).

[^4]: Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). **Measuring Massive Multitask Language Understanding**. *Proceedings of the International Conference on Learning Representations (ICLR)*.

[^5]: Schulman, John and Thinking Machines Lab, "LoRA Without Regret", Thinking Machines Lab: Connectionism, Sep 2025.
