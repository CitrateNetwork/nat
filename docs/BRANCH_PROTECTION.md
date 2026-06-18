# Branch protection — status and the post-upgrade TODO

## Current state (2026-06-18)

`main` has **no server-side protection**. The CitrateNetwork org's current plan
does not allow protected branches or rulesets on **private** repos (both the
legacy `branches/*/protection` API and the `rulesets` API return
`403 — "Upgrade to GitHub Pro or make this repository public."`).

Decision (owner): stay private, no server protection for now. The repo is
patent-sensitive and pre-counsel-sign-off, so making it public to unlock free
protection is not acceptable yet.

What stands in for protection in the meantime:

- **`.github/CODEOWNERS`** — owner review is requested on the whole repo and on
  the Tier-1 hot paths (provenance, mcp, fixed-point, settlement seam, audit
  tier). This is advisory without a ruleset, but it routes review correctly the
  moment protection is enabled.
- **`scripts/hooks/pre-push`** — a local guard that blocks a direct push to
  `main` (override: `NAT_ALLOW_MAIN_PUSH=1`). Install with
  `scripts/install-hooks.sh`. Guardrail only; not enforcement.

## TODO — apply when the org upgrades (GitHub Team or repo goes public)

Run this to enable the real protection (PR + code-owner review + CI gate +
no force-push/deletion, with org admins able to bypass):

```sh
gh api -X POST /repos/CitrateNetwork/nat/rulesets \
  -H "Accept: application/vnd.github+json" \
  --input - <<'JSON'
{
  "name": "protect-main",
  "target": "branch",
  "enforcement": "active",
  "conditions": { "ref_name": { "include": ["~DEFAULT_BRANCH"], "exclude": [] } },
  "bypass_actors": [
    { "actor_id": 1, "actor_type": "OrganizationAdmin", "bypass_mode": "always" }
  ],
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    { "type": "required_linear_history" },
    { "type": "pull_request", "parameters": {
        "required_approving_review_count": 1,
        "require_code_owner_review": true,
        "dismiss_stale_reviews_on_push": true,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true
    }},
    { "type": "required_status_checks", "parameters": {
        "strict_required_status_checks_policy": true,
        "required_status_checks": [ { "context": "build-test" } ]
    }}
  ]
}
JSON
```

After applying, the local `pre-push` hook is redundant but harmless — keep it for
contributors who push from clones without admin bypass.
