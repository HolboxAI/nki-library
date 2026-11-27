# COVEX: QK-norm in the transformer_tkg decode megakernel

**Why:**
- Upstream `experimental/transformer/transformer_tkg.py` hard-codes `rmsnorm_QK_pre_rope_enabled=False` with no Q/K gammas.
- Qwen3 models (QK-norm) therefore can't use the all-layer decode megakernel.
- `attention_block_tkg` already supports `rmsnorm_QK_pre_rope_*`.

**Change:**
- New optional args: `W_q_norm_gammas` and `W_k_norm_gammas` (per layer, `[1, d_head]` @ HBM) and `qk_norm_eps`.
- When both are given, every layer's `attention_block_tkg` call, in both the SBUF-residual and HBM paths, runs pre-RoPE RMSNorm on Q and K.
- Without them, behaviour is identical to upstream.

**Open question, before wiring into covex-neuron:** `transformer_tkg` calls `attention_block_tkg` with `K_cache_transposed=True` and `active_blocks_table=None`, i.e. a flat, transposed KV cache per layer. vllm-neuron decode uses a paged block cache. Using the megakernel from vllm-neuron needs either:
- a paged-cache path (`active_blocks_table`), or
- a per-sequence flat cache layout for that model.

So this branch is the kernel-side prerequisite. The serving integration and its A/B come after the layout decision.

**Correctness check, first:** a standalone NKI run on one Trainium chip comparing `transformer_tkg` with QK-norm against a torch reference of 2 Qwen3-8B layers (identical weights), within BF16 tolerance.
