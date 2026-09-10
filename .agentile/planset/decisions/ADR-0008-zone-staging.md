# ADR-0008 — Stage zones: validate the pass on six, test capability on three first

**Status:** accepted · **Date:** 2026-06-18 · **Remediation:** #5

## Decision
L0 validates the forward pass on all six zones (done — the architecture is
exercised end to end). But the H-01 *capability* ablation at L1 runs first on a
reduced **3-zone {HP, PF, CX}** config — the zones with real data — before
widening to SM/CB.

## Why
SM (multimodal) is "text-heavy thin slice" at v1 and CB (cerebellar/timing) has
the weakest data story. Paying the partitioning capability tax across five
learned zones while two have almost no signal would confound H-01. Isolating the
three data-rich zones gives a cleaner read on whether partitioning costs
capability, then SM/CB are added once their data earns it.

## Note
This does not change the architecture (still six zones); it sequences the
*evidence*. The sidecar already supports a subset config via zone declarations.
