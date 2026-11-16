# LM_Dec_Long_HybridGRU_LocalAttn

HybridGRU_LocalAttn - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Runnable**
- Failure class: Implementation Unverified: overfit check not passed

## Provenance

- Source repository: (new) lucidrains/local-attention
- Upstream / license: lucidrains/local-attention (MIT)

## Differences from the source ([Unified] protocol)

- GRU layer + 6 blocked sliding-window (256) rotary attention layers from local-attention (real bounded memory)

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 20.74 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy local-attention==1.11.2
python run.py check hyb_gru_local
python run.py lm02 hyb_gru_local      # enwik8 is downloaded to data/ on first use
```
