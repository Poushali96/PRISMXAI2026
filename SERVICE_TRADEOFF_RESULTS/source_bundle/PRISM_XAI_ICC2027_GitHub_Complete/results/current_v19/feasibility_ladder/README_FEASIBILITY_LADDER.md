
# PRISM-XAI Feasibility Ladder

This diagnostic stops scheduler tuning and tests where explanation signal is lost.

Stages:
A. Explanation oracle: no clipping, no DP noise, no compression.
B. Clipped oracle: clipping only.
C. Private full packet: clipping plus Gaussian DP noise, no compression.
D. Private compressed packet: clipping plus Gaussian DP noise plus Top-k/quantization.
E. MAP-XAI scheduler: existing internal fallback scheduler.
F. PRISM-XAI scheduler: existing PRISM scheduler.

Main interpretation:
- If B collapses relative to A, clipping is too restrictive.
- If C collapses relative to B, cumulative household-level DP is the bottleneck.
- If D collapses relative to C, compression is the bottleneck.
- If D is strong but F collapses, scheduler tuning is justified.
- If all private stages are near random Top-K, the privacy horizon/release rate is too strict.

Settings:
- datasets: ['opsd', 'uci']
- models: ['tree']
- seeds: [11, 22, 33]
- ladder epsilon: 4.0
- privacy sweep epsilon: [2, 4, 8, 16, 32]
- privacy sweep max releases: [4, 8, 16, 32, 64]
- steps: 512
- calibration steps: 96
