# How LLMs Work at the Architecture Level

## Summary
Large language models (LLMs) are built on transformer architectures: stacked layers composed of multi‑head self‑attention and position‑wise feed‑forward networks that operate on entire token sequences in parallel rather than recurrently [2]. Because transformers are permutation‑invariant, positional encodings are required to represent token order; several encoding schemes exist with different inductive biases [3]. Model family design (decoder‑only, encoder‑only, encoder‑decoder), pretraining objectives (causal/autoregressive, masked, prefix), and scaling/training choices (compute allocation, RLHF) strongly influence capabilities and downstream use [5][6]. Practical inference and training at large scale rely on caching, parallelism, and attention optimizations to manage memory, latency, and device limits [8][9].

## Key Findings

### Transformer core: parallel attention + feed‑forward
Transformers are organized as a stack of layers where each layer contains multi‑head self‑attention and a positionwise feed‑forward network. This design processes whole token sequences in parallel (in contrast to RNNs), enabling efficient use of parallel hardware for both training and the initial encoding of context [2][1]. Parallel processing removes sequential recurrence but requires explicit encoding of token order through positional encodings [2][3].

### Positional encodings and permutation invariance
Because the self‑attention mechanism is permutation‑invariant, positional information must be injected. Common approaches include fixed sinusoidal encodings, learnable encodings, relative position encodings, and rotary/RoPE‑style encodings. These schemes trade off expressiveness and inductive bias; the choice affects how the model represents order and generalizes over sequence positions [3].

### Architecture families and objectives
There are three common architectural patterns with distinct trade‑offs:
- Decoder‑only (autoregressive): optimized for left‑to‑right generation and used by models like GPT and LLaMA.
- Encoder‑only: optimized for understanding and classification tasks (e.g., BERT).
- Encoder‑decoder (seq2seq): uses separate stacks for input encoding and output generation (e.g., T5, BART), often excelling at sequence‑to‑sequence tasks [5].

Pretraining objectives align with these patterns. Masked‑LM, causal/autoregressive LM, and prefix LM are common; modern large generative LLMs typically use a causal/autoregressive objective (predicting the next token), which shapes model design and fine‑tuning strategies [5].

### Scaling, training dynamics, and RLHF
Empirical scaling laws permit extrapolation of compute to performance for pretraining runs, guiding resource allocation. Incorporating reinforcement learning from human feedback (RLHF) changes scaling behavior and complicates compute distribution because of additional phases like data collection, rollouts, and policy updates beyond pure pretraining [6].

### Inference phases and caching
Inference proceeds in two phases: a parallel “prefill” that encodes the input context, and an autoregressive “decode” that generates tokens one by one. The decode phase is memory‑bound and benefits from key–value (KV) caching to avoid recomputing past attention; KV caching, however, increases memory usage with longer sequences and larger batches [8].

### Distributed execution and attention optimizations
Scaling training and inference requires multiple parallelism strategies: tensor parallelism (splitting operations inside layers), pipeline parallelism (splitting layers across devices), and sequence/activation partitioning to reduce per‑device memory requirements [8]. Additional inference optimizations — batching, careful KV cache management, and attention kernels such as Flash or Paged attention that fuse kernels or chunk work to reduce memory I/O — are commonly applied to lower latency and memory overhead during generation [9].

## Conclusion
At the architecture level, LLMs center on the transformer stack (multi‑head self‑attention + feed‑forward) with positional encodings to recover sequence order. Design choices — architecture family, pretraining objective, and scaling/training strategy — determine strengths for generation or understanding. Efficient large‑scale training and low‑latency inference depend on parallelism, KV caching, and attention/kernel optimizations to balance compute, memory, and latency constraints [2][3][5][6][8][9].

## References
1. https://www.truefoundry.com/blog/transformer-architecture
2. https://engineersuniverse.com/studios/ai/aie-transformer-architecture-deep-dive
3. https://arxiv.org/html/2502.12370v1
4. https://www.linkedin.com/posts/brijpandeyji_large-language-models-llms-may-look-similar-activity-7376114094786895872-vaJl
5. https://wandb.ai/site/articles/training-llms
6. https://cameronrwolfe.substack.com/p/rl-scaling-laws
7. https://www.youtube.com/watch?v=jk2FsJxZFo8
8. https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization
9. https://pub.towardsai.net/llm-inference-optimization-730b2d718a44