# Plate coverage sweep — 18 September 2026

Same 20 road frames and 40 annotated plates as the tiled-pass measurement in
`PERSONAL_PROFILE_SCORECARD_2026-09-18.md` (Deepak `vid-2` training side; `vid-1`
stays locked). Covered = at least half the plate box under the burnt-in mask.
Every configuration uses the personal threshold profile and the full provider
set; only the listed options change.

| Configuration | Plates covered | Rate | Mean photo erased |
|---|---:|---:|---:|
| Single pass (no tiles) | 24 / 40 | 0.600 | 9.2% |
| Tiles 768 — **shipped default** | 33 / 40 | 0.825 | 10.9% |
| Tiles 512 | 35 / 40 | 0.875 | 13.7% |
| Tiles 768 + second plate model | 35 / 40 | 0.875 | 16.7% |
| Tiles 512 + second plate model — `-MaxPlateCoverage` | **38 / 40** | **0.950** | 21.8% |

The second plate model is the CCPD-to-India fine-tune, fused alongside v4.

## Decision

`-MaxPlateCoverage` clears the 85% whole-region bar for plates, but it doubles
the share of each road photo that gets erased, and a separate diagnostic found
plate detectors already putting ~6% of stray boxes on photos with no plates in
them. That false-alarm cost was not measured for this configuration. It ships
as an opt-in launcher switch, not the default, until it is.

## Caveats

40 plates from consecutive frames of one video are correlated: the effective
sample is smaller than 40, and 35 versus 38 is three plates. This is enough to
rank the configurations, not to quote a precise recall.

Source: `reports/plate_coverage_sweep_2026-09-18.json`.
