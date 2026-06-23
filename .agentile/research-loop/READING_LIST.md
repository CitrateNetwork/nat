---
created: 2026-06-22T00:00:00Z
branch: docs/nat-values-spine-reading-list
author: Larry Klosowski (@SaulBuilds) + Claude Opus 4.8 (1M context)
status: active
---

# Corpus Reading List — toward a good coder with great logic, creative & expressive, who follows the rules of the room

A curated, deliberately opinionated target list for the corpus (DATA-S1 / Hermes
WP-H5). The thesis tying it together: **a rule has no meaning without a community
and a form of life** (Wittgenstein) — which is also why a maker follows *the rules
of the room they are in*, why provenance must answer to a *public* standard, and
why good code reads like the codebase around it. Logic (Boole → Frege → Russell →
Belnap), computation (Turing → Church → Shannon), craft (SICP, the Rust Book,
permissive code), and expression (Strunk, Whitman, Montaigne) are the four pillars
on that one foundation.

**License key** — the pipeline is fail-closed on license:
`✓` PD / permissive (trainable) · `©→CC0` copyrighted, capture via an authored CC0
explainer (we own the framing) · `⚠` verify license before fetch.

## I. Rules, language, meaning — "the rules of the room"

| Work | Status | Source |
|------|--------|--------|
| Wittgenstein — *Philosophical Investigations* (rule-following §§138–242; private language §§243–315) | `©→CC0` | **done**: seed `wittgenstein-rule-following`, `wittgenstein-private-language`, `form-of-life-and-the-room` |
| Wittgenstein — *Tractatus Logico-Philosophicus* | ✓ done | Gutenberg **5740** (.tex → `from-text --strip latex`) |
| George Boole — *An Investigation of the Laws of Thought* | ✓ done | Gutenberg **15114** (.tex → `from-text --strip latex`) |
| Bertrand Russell — *The Problems of Philosophy* / *Introduction to Mathematical Philosophy* / *The Analysis of Mind* | ✓ | Gutenberg 5827 ✅, 41654 ✅, 2529 |
| Lewis Carroll — *What the Tortoise Said to Achilles* (the rule-following regress) | ✓ | PD (transcribe) |
| Belnap four-valued logic; first-degree entailment | `©→CC0` | **done**: seed `belnap-four`, `fde-entailment` |
| Frege — sense vs reference; *Foundations of Arithmetic* | `©→CC0`/⚠ | translation ©; **done**: seed `sense-reference` |
| J.L. Austin — *How to Do Things with Words* (speech acts); Grice — implicature | `©→CC0` | authored explainers (planned) |

## II. Computation & the Turing lineage — technology

| Work | Status | Source |
|------|--------|--------|
| Alan Turing — *On Computable Numbers…* (1936); *Computing Machinery and Intelligence* (1950) | ✓ (life+70, PD 2025) | clean transcription; **done**: seed `turing-machine`, `turing-test` |
| Ada Lovelace — *Notes on the Analytical Engine* (1843) | ✓ | PD (first program; computation + imagination) |
| Claude Shannon — *A Mathematical Theory of Communication* (1948) | `©→CC0` | Shannon d.2001; authored explainer (information/entropy) |
| Alonzo Church — lambda calculus; the Church–Turing thesis | `©→CC0` | authored explainer (planned) |
| John von Neumann — *First Draft of a Report on the EDVAC* | ⚠ | verify |

## III. The craft & design of code — a good coder

| Work | Status | Source |
|------|--------|--------|
| Abelson & Sussman — *Structure and Interpretation of Computer Programs* (SICP) | ✅ done | **CC-BY-SA-4.0** — `sarabander/sicp` HTML (book files explicitly CC-BY-SA-4.0) → tag-strip → `from-text` (461 passages, 2026-06-22; owner-approved the CC-BY-SA fetch) |
| *The Rust Programming Language* (the Book) | ✅ done | **MIT/Apache** — `rust-lang/book` markdown → `from-text` (550 passages, 2026-06-22) |
| Permissive source repos (MIT/Apache/BSD) | ✅ | **`nat-corpus from-code`** — `rust-lang/log` + **anyhow, itertools, serde (MIT/Apache, 2026-06-22)**; `scripts/fetch-code-craft.sh` |
| The Unix philosophy; "worse is better"; "do one thing well" | `©→CC0` | authored explainer (planned) |
| IETF RFCs (e.g. RFC 1925 *Twelve Networking Truths*) | ⚠ | IETF Trust license — verify |

## IV. Creativity & expression — creative, expressive

| Work | Status | Source |
|------|--------|--------|
| William Strunk — *The Elements of Style* (clarity, economy) | ✓ | Gutenberg **37134** |
| Walt Whitman — *Leaves of Grass* | ✓ | Gutenberg **1322** |
| Montaigne — *Essays*; Emerson — *Essays* | ✓ | Gutenberg 3600; 2945 |
| Lewis Carroll — *Alice's Adventures in Wonderland* | ✓ | Gutenberg **11** (logic at play in language) |
| Aesop — *Fables* (concise reasoning + narrative) | ✓ | Gutenberg 11339 |

## Notes

- **Code structure & NORMALIZE**: the pipeline's NORMALIZE flattens whitespace, so
  code currently trains lexically but loses layout. A code-aware normalization
  (preserve newlines/indentation) is a DATA-S1 follow-up before the CX zone is fully
  served.
- **Text/markdown ingest**: SICP and the Rust Book are CC text in markdown — they
  need a `from-text`/`from-markdown` connector (small WP) since `from-code` only
  takes source extensions.
- **Turing PD**: Turing died 1954, so his works entered PD (life+70) in 2025 — fetch
  a clean transcription, record provenance honestly.
- **Owner-licensed values data**: SPIRIT.md / SOUL.md / the Agentile rules could be
  included as direct values-alignment text (owner decision; they ARE the room's
  rules made explicit).
