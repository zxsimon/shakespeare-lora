Thoughts

- Baseline evaluation: mostly a variance reduction tool to judge model incremental performance during training. Nice to have, but does not seem too necessary, especially when we can just increase sample size
- How to best implement LLM as a judge? Can consider a pairwise comparison using model checkpoints, with an elo-like implementation
    - Observation that LLM judge tends to avoid giving extreme ends of the scale, and cluster around the mean
- Issue with using PEFT and torch.compile reduce-overhead