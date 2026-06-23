---
created: 2026-06-22T00:00:00Z
branch: main
author: Larry Klosowski (@SaulBuilds) + Claude Opus 4.8 (1M context)
status: engaged
note: A dedicated agent was started on this track on 2026-06-22 and is actively working it,
  in parallel with the Hermes operator-agent build on citrate-agent-runtime. Update to
  `completed` when the open-work list below is discharged.
---

# Handoff — NAT research-and-data track

This is the onboarding brief handed to the agent now running the NAT + corpus track in
parallel with the Hermes build. It is version-controlled so the agent (or its successor
after a context clear) can be re-pointed at it.

---

You are taking over the NAT (Citrate Neuroarchitectural Transformer) research-and-data
track for Citrate Network, working alongside the owner (Larry Klosowski / "saul" —
founder + sole maintainer) and a research team that is writing the arXiv paper. Another
agent is building the Hermes operator agent in parallel; your lane is NAT + the training
corpus. Read this whole brief before acting, then confirm your understanding and propose
your first concrete step.

═══════════════════════════════════════════════════════════════════════
WHERE YOU ARE — repos & the federation
═══════════════════════════════════════════════════════════════════════
• Local working root: /home/saul/Projects/Citrate-Labs/  (this box IS the DGX Spark,
  GB10, aarch64 — candle-cuda builds need CUDA 12.8 side-by-side + CUDA_COMPUTE_CAP=120;
  see nat/scripts/dgx-gpu.sh and nat/.cargo/config.toml for the build flags).
• Your repo: ./nat  (GitHub: CitrateNetwork/nat). A Rust workspace; crate boundaries are
  load-bearing — nat-types, nat-provenance, nat-mcp, nat-sidecar, nat-core, nat-candle
  (the Candle L1 stack), nat-ablation, nat-data (the corpus pipeline), nat-train,
  nat-eval, nat-federated (Gate-4 scaffold).
• The GitHub org is CitrateNetwork; sibling repos live under Citrate-Labs/ locally
  (citrate-agent-runtime, citrate-chain, gateway, etc.). You normally only need ./nat.
• BRANCH/PUSH MODEL for nat: the research team pulls main, so nat uses a push-to-main
  flow — commit to main and push with `NAT_ALLOW_MAIN_PUSH=1 git push origin main`.
  ALWAYS `git pull --rebase origin main` first (the paper team pushes often); keep their
  paper/ work untouched. Higher-audit repos (e.g. citrate-agent-runtime) use PRs instead
  — but you live in nat.
• Secrets/keys: never commit them. The corpus itself (./nat/corpus/) is GITIGNORED — data
  is built locally, never committed.

═══════════════════════════════════════════════════════════════════════
HOW WE WORK — agentile methodology
═══════════════════════════════════════════════════════════════════════
Everything is planned and tracked under .agentile/ and PLANSET/:
• .agentile/sprints/active/ → completed/ — each sprint is one markdown file with Rule-12
  frontmatter (created / branch / author / status / sprint), a WP (work-package) table
  with explicit Acceptance criteria, and a Status column. When a sprint's WPs are done
  you write a REPORT.md (or a close-out section) and MOVE the file to completed/.
• .agentile/planset/gates.yaml — the rung gates (L0..L3) with exit_criteria and `met:`
  flags. RULE: a criterion stays `met: false` until it is *actually, verifiably* true.
  When you scaffold something not yet fully proven, add a `scaffold:` note but DO NOT
  flip met to true. Honesty over green.
• .agentile/planset/hypotheses.md — H-01..H-05 with status (open/supported/refuted).
  H-01 ("zone partitioning does not cost capability per parameter") is THE load-bearing
  bet; it currently HOLDS 5/5 seeds on real text at small scale.
• PLANSET/09_JOURNAL.md — the build journal. After any non-trivial session, append an
  entry with the *durable lesson* and an honest "what's true now / what's still a bet"
  section. This is a first-class deliverable, not an afterthought.
• Discipline: RED-TEST-FIRST (write the failing test, then make it green). HONEST POSTURE
  is the whole culture — if a result is marginal or a claim is bigger than the evidence,
  say so plainly. There is a `claim-grader` agent for checking that commit/PR/journal
  language matches what was actually done; use it on anything load-bearing.
• Commit convention: conventional-commit subject; end the body with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. PR bodies (when
  used) end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

═══════════════════════════════════════════════════════════════════════
THE citrate-memories MCP — the federation knowledge graph
═══════════════════════════════════════════════════════════════════════
The federation runs a knowledge-graph MCP server, "citrate-memories", that ingests
changes across the org and lets you recall/assert facts and the edges between them. Its
tools appear in your session as `mcp__citrate-memories__*` but are DEFERRED — their
schemas aren't loaded until you ask for them:
  1. Run ToolSearch with `select:mcp__citrate-memories__memory_recall,
     mcp__citrate-memories__memory_search,mcp__citrate-memories__memory_assert,
     mcp__citrate-memories__memory_neighbors,mcp__citrate-memories__memory_propose_edge,
     mcp__citrate-memories__memory_confirm_edge` to load them.
  2. START every work session by `memory_search` / `memory_recall` on your topic
     (e.g. "NAT H-01", "corpus pipeline", "research loop intent") to pull what the
     federation already knows — don't re-derive context that's recorded.
  3. As you establish durable facts (a result, a decision, a new corpus source and its
     license), `memory_assert` them, and link related nodes with `memory_propose_edge`
     / `memory_confirm_edge`. Treat recalled memories as *background context* (they
     reflect what was true when written) — verify a cited file/flag still exists before
     relying on it.
If the MCP tools don't appear at all, the server isn't connected in this session — tell
the owner; it's configured at the Claude Code / session level, not something you install.

═══════════════════════════════════════════════════════════════════════
WHAT'S DONE (do not redo — verify, then build on)
═══════════════════════════════════════════════════════════════════════
• Data pipeline WP-D1..D11 all DONE: byte tokenizer, corpus persist/loader, BPE
  (nat-data::bpe, 1.99 bytes/tok @ vocab 1024), code-aware normalize, from-text +
  LaTeX-strip connector (Boole 15114 + Tractatus 5740 ingested), mini-batch SGD,
  per-position autoregressive LM (nat-candle::autoreg, 3.42 bits/byte @ 53K params),
  and the scale ladder S→M→L (5-zone L = best, 3.953 bits/byte).
• H-01 DECISIVE: `nat-ablation::run_real_corpus_ablation` — NAT held-out loss 2.88–2.91
  < dense 2.97–2.99 at equal params, 5/5 seeds, on the 1.12M-token corpus.
• GGUF export (nat-candle::gguf) round-trips; TLC specs green; Gate-4 federated core
  scaffolded in nat-federated (signed verify-before-compose gather + ChainCommit/
  Settlement seams + H-05b tolerance harness).
• Corpus build recipe: scripts/fetch-values-spine.sh (writes to gitignored corpus/);
  CLI is `nat-corpus` (run / emit-seed / from-gutenberg / from-code / train-bpe /
  from-text). Permissive/public-domain licenses ONLY — the pipeline is fail-closed on
  ALLOWED_LICENSES, and provenance is immutable. Never weaken that gate.

═══════════════════════════════════════════════════════════════════════
WHAT'S LEFT — your open work (prioritized; confirm with the owner before big runs)
═══════════════════════════════════════════════════════════════════════
1. H-02 HELD-OUT on real data — H-01 is done; H-02 (context-aware routing produces
   measurably different zone mixes) is still in-sample. Build the real held-out H-02 read
   in nat-eval (see h02_heldout.rs) and update hypotheses.md honestly. [highest-value,
   no infra needed]
2. SPRINT CLOSE-OUTS — .agentile/sprints/active/ has four sprints whose WPs are largely
   delivered (DATA-S1, S2-trainable-zone-pass, NAT-S3-federated, HERMES-S1) but
   completed/ is empty. Write REPORT.md close-outs and move the done ones. Agentile
   hygiene that's overdue. (If the owner is intentionally keeping any of these active,
   skip that one.)
3. RESEARCH LOOP — .agentile/research-loop/INTENT.md is the append-only "what to learn
   next / what data to gather"; READING_LIST.md is the source queue. Grow the corpus
   against the latest intent (more public-domain primaries in logic/language/CS/math —
   the goal is "a good coder with great logic, creative + expressive, that follows the
   rules of the room"). Append a daily standup. Keep H-01 in view at every corpus rung —
   if it ever stops holding, SAY SO; that's the discipline.
4. MODEL-QUALITY GATE — wire the data-quality scorer (nat-data::quality::NgramModel)
   into the corpus acceptance path as a gate, not just a score.
5. g3-gguf (gates.yaml, met:false) — NAT's GGUF round-trips but doesn't execute in stock
   llama.cpp/Ollama (the zone graph isn't a recognized arch). The runtime mapping is a
   separate effort if the owner wants it.
6. INFRA-GATED (need owner-provisioned resources — DON'T start without him):
   • Gate-4 real run (nat-federated WP-F3..F6): multi-node gather + on-chain commit vs
     citrate-chain + settlement vs citrate-compute-pool + the production operator signer.
   • L2 scale run (g5-l2): committed compute.

═══════════════════════════════════════════════════════════════════════
THE ENDGAME — integrating Hermes into the pipeline
═══════════════════════════════════════════════════════════════════════
A parallel track is building "Hermes", a general-purpose operator agent on
citrate-agent-runtime whose skills are "capsules". Its RESEARCH subagent (capsules:
source-vet, corpus-fetch, corpus-normalize — see that repo's
.agentile/planset/hermes/01_CAPSULE_CATALOG.md, domain A) is designed to EXECUTE this
exact research loop: read INTENT.md, fetch+vet+normalize sources through nat-corpus, and
report standups. So: keep nat-corpus's CLI contract and the INTENT.md/READING_LIST.md
format stable and scriptable — that's the integration seam. When Hermes is ready, it
plugs in behind nat-corpus; the cleaner that boundary stays, the cheaper the integration.

═══════════════════════════════════════════════════════════════════════
FIRST ACTIONS
═══════════════════════════════════════════════════════════════════════
1. `git pull --rebase origin main` in ./nat; skim .agentile/planset/gates.yaml +
   hypotheses.md + PLANSET/09_JOURNAL.md (tail) + the four active sprints.
2. Load the citrate-memories MCP tools (above) and memory_search "NAT" / "corpus".
3. Run the test suite (`cargo test` workspace-wide; GPU paths via scripts/dgx-gpu.sh) to
   confirm a green baseline before changing anything.
4. Propose your first step to the owner — recommended is item #1 (H-02 held-out), then
   close out the delivered sprints (#2). Don't kick off any infra-gated run (#6) without
   explicit go-ahead.

Operating constraints, always: permissive/public-domain licenses only into the corpus;
provenance immutable; corpus/ gitignored; honest posture over green checkmarks; rebase
before push; never touch the paper team's paper/ work.
