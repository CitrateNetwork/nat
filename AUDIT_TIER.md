---
created: 2026-06-18T00:00:00Z
branch: main
author: nat-bootstrap
status: active
---

# Audit Tier — `nat`

**Classification**: **Tier 1 — full audit** before first stable (`v1.0.0`) release tag.

## Rationale

NAT sits on two audit-relevant surfaces at once:

1. **Reward-bearing.** NAT emits the data-quality score, the metered-compute
   receipt, and the provenance hash that `citrate-compute-pool` settles into
   participant rewards (see `docs/SETTLEMENT_SEAM.md`). Any code path that
   influences a reward computation is Tier-1 by the same logic that makes
   compute-pool Tier-1.
2. **Verification-bearing.** The provenance trace is committed on-chain and
   replayed by third parties to verify an inference. A flaw in the trace hash,
   the deterministic merge, or the MCP action gate is a correctness/security
   defect with on-chain consequences.

## What this means concretely

- **Full code audit** by an external security firm covering: the deterministic
  merge (Q16.16) path, the provenance hash and replay verifier, the MCP harness
  state machine (`NoUngatedSideEffect`, `NoExecOnFailedCodec`), the sidecar
  loader, the settlement seam, the dependency tree (`cargo audit` + `cargo deny`),
  and CI/CD supply chain.
- **No stable release tag** without a written audit attestation referencing the
  exact commit SHA.
- **Prerelease tags** (`v0.x.y-rc.N`) may ship without audit but MUST carry
  `prerelease: true` and a clear warning in the release notes.
- **Re-audit cadence**: every major version AND any change that materially
  expands attack surface (new settlement field, new signing primitive, a learned
  component entering the merge/gate path, a new on-chain commitment format).

## Promotion status

Tier-1 **classification** is declared from day one (this file). Tier-1
**promotion** — the operator sign-off that the repo has cleared its first full
sprint pass (Gate 1 + Gate 2 green) and is ready to carry the classification's
obligations — lands when:

- Gate 1 artifacts are complete (`PLANSET/`, `.agentile/planset/gates.yaml`,
  the three TLA+ modules, the Gherkin feature set), and
- Gate 2 is green (the L0 forward pass passes the `features/gate2_*.feature`
  acceptance set; `cargo test` is green).

Until promotion, this is a Tier-1 repo *under construction*; the obligations bind
at the first stable release, per **D6** of the federation-split decisions.

## Decision authority

Per **D6**, every repo audits before its first stable release. Tier changes
require: (a) a commit to this file explaining the change, AND (b) sign-off from
the operator in CODEOWNERS (when present) or the federation lead.
