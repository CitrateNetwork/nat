# NAT — arXiv source

Single-document LaTeX assembly of the NAT paper (also Gradient Paper XI), built from the
section drafts in `paper/*.md` after the adversarial red-team and remediation.

## Files
- `main.tex` — the complete paper (standard `article` class; only common packages:
  `geometry, amsmath, booktabs, natbib, hyperref, microtype`). No custom `.sty`.
- `references.bib` — bibliography; every entry verified real in the citation red-team. The
  `eprint` arXiv IDs should be re-confirmed against the live listing before final submission.

## Build
```sh
# Any of:
latexmk -pdf main.tex
# or
pdflatex main && bibtex main && pdflatex main && pdflatex main
# or (self-contained, downloads packages on first run):
tectonic main.tex
```
Overleaf: upload `main.tex` + `references.bib`, set compiler to pdfLaTeX, compile.

## Status / before submission
- **Title:** kept ambitious per the owner's decision; the body is remediated to state every
  claim's status (demonstrated / implemented / specified) so the title is honest.
- **Open items:** (1) re-verify the `[eprint]` arXiv IDs; (2) the Citrate Papers II/X are cited
  as technical reports (`@misc`) — replace with public URLs/DOIs when available; (3) the paper
  honestly lists the missing experiments (MoE baseline, component ablations, standard corpus) —
  running them strengthens it but is not required to post a preprint.
- **Provenance:** the prose is the post-red-team text from `paper/00_OVERVIEW.md` and
  `paper/02..09_*.md`; the findings + remediations are logged in
  `paper/research/REDTEAM_FINDINGS.md`; citations in `paper/research/RELATED_WORK_AND_CITATIONS.md`.
