# COVEX: configurable FA tile (`COVEX_NKILIB_FA_TILE_SIZE`)

**Why:**
- Upstream `attention_tkg` tiles decode attention in 8K chunks, and only when `s_prior > 8K`.
- nkilib's dynamic FA early exit (`max_context_len`, used by covex-neuron's `COVEX_DECODE_DYNAMIC_FA`) skips whole tiles. So at `max_model_len` ≤ 8K it saves nothing, and at 32K it still reads one full 8K tile per request.

**Change:**
- `_FA_TILE_SIZE` in `core/attention/attention_tkg_utils.py` comes from `COVEX_NKILIB_FA_TILE_SIZE`: default 8192 (unchanged), power of two ≥ 512.
- It's read at import, so it's a compile-time constant.
- **Constraint (existing assert):** the tile must be a multiple of `block_len × 128`. For example, a 2048 tile needs a resized `block_len` ≤ 16. Incompatible configs fail at compile time, not silently.
- GPT-OSS giga kernels keep their own copies and are unchanged.

**A/B, on top of covex-neuron `feat/llama-dynamic-fa`:**
- **Model:** Llama-3.1-8B TP8, `max_model_len` 8192, 1024/256 at concurrency 16 and 32
- **A:** `COVEX_DECODE_DYNAMIC_FA=1`, default tile
- **B:** the same, plus `COVEX_NKILIB_FA_TILE_SIZE=2048`
- **Metric:** TPOT p50, plus GSM8K
- **Expected (estimate):** decode KV read ~4x lower at short contexts. The win depends on attention's share of the step, which the profile shows.
