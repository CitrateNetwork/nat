---
created: 2026-06-22T00:00:00Z
branch: docs/data-s1-hermes-plansets
author: Larry Klosowski (@SaulBuilds) + Claude Opus 4.8 (1M context)
status: active
sprint: HERMES-S1
---

# HERMES-S1 — Hermes, the data-research agent

Stand up a configured Hermes instance whose standing job is to **research, collect,
and refine permissively-licensed data into the formats NAT trains on**, working
largely unattended (Operator autonomy) under hard guardrails, and **operating like
the rest of the ecosystem's agents** — it creates repos, tracks work in sprints,
and journals — via a set of Agentile skills we author and feed it.

Grounding: Hermes already has an **operating model** in the Citrate repo
(`Citrate/.agentile/audits/2026-04/.../03_AGENT_LOGSEQ_HERMES_OPERATING_MODEL.md`):
policy profiles, capability grants, append-only trails, an approval queue, and a
Logseq journal projection, surfaced in the GUI's Operations / Agent Center. What is
missing is a **configured instance pointed at the DATA-S1 job**. Agent infra lives
in `citrate-agent-runtime` (agent / agent-cron / agent-code / capsules) and
`citrate-node-agent` (heartbeat, lifecycle, supervision, cron).

## Mission

Given the daily intent (`.agentile/research-loop/INTENT.md`), Hermes:
1. **finds** sources for the target domains (DATA-S1) — permissive only;
2. **fetches** + **converts** them to the `RawDoc` JSONL contract;
3. **runs** the `nat-data` pipeline → shards + manifest (license/PII/dedup/quality
   gates do the screening, fail-closed);
4. **reports** coverage / aggregate quality / zone balance / quarantine reasons in a
   daily standup (the Logseq projection);
5. **builds** the small tools it needs (scrapers, converters, license detectors) in
   a sandbox, tracked as its own repos/sprints;
6. **escalates** only on policy boundaries (new source domain, non-allow-listed
   license, large fetch, spend).

## Autonomy & guardrails (Operator, from the operating model)

- **Policy profile**: `Operator` — runs routine fetch/refine/pipeline unattended
  within scoped grants; escalates on policy violations only.
- **Capability grants** (scoped, expiring, revocable): network-**fetch** + write to a
  **data dir** + write to a **tools/ sandbox**. NO chain keys, NO secrets, NO shell
  outside policy, NO write outside scopes (the formal boundary).
- **Approval queue** interrupts for: a new source *domain*, any non-allow-listed
  license, a fetch above a size/spend ceiling, or any policy escalation.
- **Append-only trail** of every action; **provenance immutable**; the
  `ALLOWED_LICENSES` quarantine makes bad data fail-closed — this is what makes
  Operator autonomy safe.
- **Kill switch** visible at all times; per-session revoke.

## Skill/tool set

Capabilities Hermes draws on (existing where possible, built where not):
- web search + fetch; dataset APIs (HuggingFace, Project Gutenberg, Common Crawl);
- the `nat-data` gates as a library (license, PII, dedup, quality) + the pipeline
  runner (a CLI, WP-H3);
- format conversion → `RawDoc` JSONL (WP-H2);
- a quality scorer (heuristic now; model-based at DATA-S1 WP-D5);
- `agent-code` to scaffold its own throwaway tools in the sandbox.

## Agentile skills to author (owner ask: "operate like our other agents")

A skill pack so Hermes plans/tracks/ships like the ecosystem (mirrors the user's
`/sprint`, `/journal`, `/case-study` skills, specialized for data work):

| Skill | Purpose |
|-------|---------|
| `data-source-intake` | Vet a candidate source: license, robots/ToS, volume, domain fit → an intake record + an approval-queue entry if new. |
| `corpus-run` | Convert → run the `nat-data` pipeline → persist shards → write the manifest summary. |
| `research-standup` | Write the daily standup entry (new tokens, quality, zone balance, quarantine, tools built) to the Logseq journal / research loop. |
| `repo-scaffold` | Create a new repo for a tool it builds, with Agentile frame (AGENT_ENTRY, gates, sprint dir) — so its tools are first-class, tracked repos. |
| `sprint` / `journal` | The standard Agentile lifecycle skills, so its own work is sprint-tracked and journaled like every other agent. |

## Work packages

| WP | Subject | Acceptance |
|----|---------|-----------|
| WP-H1 | **Hermes config**: Operator profile + capability grant (data dir + fetch + sandbox) + approval-queue wiring | a configured, revocable session visible in Agent Center |
| WP-H2 | **`RawDoc` JSONL contract** + converters | round-trips a fetched doc → `RawDoc` → pipeline, provenance intact |
| WP-H3 | **Pipeline-runner CLI** (`nat-data`) | `cargo run`-able: JSONL in → shards + manifest out, on disk |
| WP-H4 | **Agentile skill pack** (above) | each skill runs; Hermes creates a repo + sprint + standup unaided |
| WP-H5 | **Source connectors** (Gutenberg, HF, permissive code) | each fetches + screens; new domains hit the approval queue |
| WP-H6 | **Discord wiring + daily cron** | Hermes runs the daily cycle on a schedule; reports to Discord + the research loop |
| WP-H7 | **Operations surfacing** | seen → configured → monitored → paused → audited from the GUI (the operating model's "finished feature" bar) |

## The boundary (what needs the owner)

Claude **cannot** deploy/configure the Discord bot or hosting (needs the server,
bot token, secrets). Claude **can** build everything local: WP-H2/H3/H4 + the
config templates (H1) + connectors (H5). The owner (or a `!`-prefixed login in
session) wires Discord/cron (H6) and grants the capabilities. Until then Hermes has
a job, a sandbox, and guardrails ready.

## Exit criteria

- [ ] Hermes runs a daily cycle unattended within scoped grants, escalating only on
      policy boundaries.
- [ ] It grows the DATA-S1 corpus across the four zones, all provenance/license
      auditable, all bad data quarantined.
- [ ] It creates + tracks its own tool repos/sprints/journals like other agents.
- [ ] Fully `seen → configured → monitored → paused → audited` from the GUI.

## Honest posture

Operator autonomy over network fetch + data refinement carries real risk
(licensing exposure, runaway fetches, quality drift). The mitigations are
structural — fail-closed license gate, immutable provenance, approval queue for new
domains, scoped/expiring grants, append-only trail. If those can't be enforced,
drop Hermes to `Guided Builder` until they can.
