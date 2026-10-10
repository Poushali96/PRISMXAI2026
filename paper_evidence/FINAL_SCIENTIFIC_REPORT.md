# PRISM-XAI ICC 2027 reproducibility and publication report

Generated UTC: 2026-10-10T03:03:48.199703+00:00

## Scientific status

- Run A is the unchanged original-v1.9 M8 reproducibility result.
- Run B is a separate M4 sensitivity analysis and must not be pooled with A.
- Run C is a new experimental public-score algorithm with its own saved hash and patch.
- None of these executions alone proves full-transcript differential privacy.
- A comparison is called matched-service/matched-privacy only when it passes the declared overlap tolerances.

## Validation

- **A_ORIGINAL_V19_M8_WITH_ABLATIONS:** 720 rows; validation=True; duplicate keys=0; epsilon overruns=0; script SHA-256 `1b903dcf971010ef35d70b65c97ec532cf9e7010b4f12f57c57c989c0c51c598`.
- **B_ORIGINAL_V19_M4_WITH_ABLATIONS:** 720 rows; validation=True; duplicate keys=0; epsilon overruns=0; script SHA-256 `1b903dcf971010ef35d70b65c97ec532cf9e7010b4f12f57c57c989c0c51c598`.
- **C_EXPERIMENTAL_PUBLIC_SCORE_M8:** 720 rows; validation=True; duplicate keys=0; epsilon overruns=0; script SHA-256 `dc16f19721c7a2a48db99589ca23fe548bf657bcd07d30e9d5df8dcb86b0b629`.

## Paired PRISM-XAI versus repaired MAP-XAI

- **A_ORIGINAL_V19_M8_WITH_ABLATIONS:** traffic Δ=-1.185 ± 0.411 kbit (-0.50%); coverage Δ=0.021 pp; critical-recall Δ=-0.039 pp.
- **B_ORIGINAL_V19_M4_WITH_ABLATIONS:** traffic Δ=-0.022 ± 0.028 kbit (-0.00%); coverage Δ=0.002 pp; critical-recall Δ=-0.136 pp.
- **C_EXPERIMENTAL_PUBLIC_SCORE_M8:** traffic Δ=-36.398 ± 3.321 kbit (-15.72%); coverage Δ=-0.010 pp; critical-recall Δ=-0.083 pp.

## Privacy claim boundary

The accountant check shows that recorded private releases do not exceed each declared epsilon budget. It does not establish privacy for participation, endpoint identity, release timing, packet mode, public predictions, calibration, candidate selection, or other observable transcript fields. A theorem must explicitly define adjacency, public information, observable outputs, and composition for every data-dependent channel.

## Required manuscript language

Use: ‘The implemented zCDP accountant bounds the explicitly privatized explanation releases under the stated clipping and adjacency assumptions.’ Do not use: ‘The complete communication transcript is differentially private’ unless a separate proof covers timing, selection, modes, and all other observations.

## Submission gate

1. Retain A as the original-v19 result and describe its PRISM-over-MAP effect at its measured magnitude.
2. Treat C as a new method, cite its exact hash and patch, and regenerate every dependent table and figure.
3. Report non-overlap for fixed-DP baselines instead of calling common nominal epsilon a matched comparison.
4. Prespecify the primary hypotheses or apply the included BH correction before highlighting significance.
5. Add an operational explanation-usefulness study; Top-K-over-random alone is not sufficient.

## Experimental variant provenance

```json
{
  "created_utc": "2026-10-10T01:59:01.908453+00:00",
  "base_file": "source_bundle/PRISM_XAI_ICC2027_GitHub_Complete/code/prism_xai_icc2027_v1_9_selective_release.py",
  "base_sha256": "1b903dcf971010ef35d70b65c97ec532cf9e7010b4f12f57c57c989c0c51c598",
  "variant_file": "source_bundle/PRISM_XAI_ICC2027_GitHub_Complete/code/prism_xai_icc2027_v2_0_public_score_experimental.py",
  "variant_sha256": "dc16f19721c7a2a48db99589ca23fe548bf657bcd07d30e9d5df8dcb86b0b629",
  "patch_file": "provenance/public_score_variant.patch",
  "patch_sha256": "f69939c14f3332c5b1738397c2ef9fc57f26cf52e3b8f5c625490e8ecd137707",
  "privacy_status": "experimental; not a full-transcript DP certificate"
}
```
