# Visual Design Brief — NAT Training Console

**Document:** RFC-CIT-NAT-0001 / Design Brief
**Status:** Draft v0.1
**For:** Claude Design
**Grounded in:** `02_ARCHITECTURE_SPEC.md`, `03_FORMAL_SPEC_SCAFFOLD.md`
**House style:** ellipses over em dashes, grounded copy, claims anchored to data

---

## 1. The subject, pinned

The product is the **NAT Training Console**: the interface where a person watches
and steers the training of a zone-partitioned model, and where a federation of
people sees their collective model take shape. Its single job: **make the
invisible visible.** NAT's whole reason to exist is that you can see which zones
fired and why. The console is that thesis made interactive.

Two audiences, one product:
- **The individual operator** training on their own machine (Spark, or a node).
  Wants to see their run's health, their zones, their contribution.
- **The federation** training together toward a shared model. Wants to see the
  collective: who is training what, how zones are converging, the shared model
  breathing.

The console is not a generic ML dashboard with loss curves and GPU gauges. Those
exist and they are commodity. The console's job is to render **the zone
architecture itself** as the primary object... the six zones, the topology
between them, the router's per-input modulation, the merge, the provenance trace.

## 2. Design thesis (the hero)

The hero is **the living zone map.** Not a number, not a hero headline... a
real-time rendering of the six zones as a connected topology, lit by activity.
When a prompt runs through the model, you watch signal flow along the declared
edges, watch zones brighten by activation strength, watch the merge prune the dim
ones and compose the bright ones. The provenance trace is not a log buried in a
tab... it is the visual itself, replayable.

This is the one memorable thing. Everything else stays quiet around it.

A person should be able to point at the hero and say "that is the model
thinking, and I can see it." That sentence is the whole product.

## 3. Token system (starting point, designer to refine)

Treat these as a deliberate starting palette derived from the subject... a model
that is part nervous system, part ledger, part instrument. Refine or replace, but
do not default to cream-serif-terracotta, near-black-with-acid-accent, or
broadsheet hairlines. The subject earns something more specific.

**Color.** The subject is living signal over a verifiable substrate. Suggested
direction: a deep, near-neutral substrate (not pure black... a cool graphite or
deep slate that reads as "instrument," not "void"), with **zone-coded hues** that
carry meaning rather than decorate. Each zone owns a hue:

- `SM` Sensorimotor — a cool perceptual cyan
- `CB` Cerebellar — a fast, kinetic amber
- `HP` Hippocampal — a warm memory magenta
- `PF` Prefrontal — a deep cognitive indigo
- `CX` Codec — a precise verification green (green = verified, by convention)
- `MX` MCP Harness — a neutral executive white/silver (it is the non-learned one)

Zone hues are a *system*, not a mood. They recur everywhere a zone is referenced,
so a person learns the color language once and reads it everywhere. Provide 4–6
named substrate/neutral values plus the six zone hues with on-dark and on-light
variants.

**Type.** Two roles minimum, chosen for this brief, not defaults:
- A **display face** with a technical-but-humane character for the zone names and
  section markers... something that reads as "designed instrument," used with
  restraint.
- A **body/data face** with excellent tabular figures, because this interface is
  full of scores, weights, latencies, and hashes that must align in columns.
- A **mono utility face** for hashes, config, and trace fields... the
  provenance record should *look* like something you can verify.

**Layout.** Two-zone composition (the design conversation's instinct, applied to
UI): a **stage** that holds the living zone map (dominant, center/left), and a
**rail** that holds the controllable detail (run controls, zone inspector, trace
scrubber). The stage is for watching; the rail is for steering. On mobile the
stage stacks above a collapsible rail.

**Signature.** The **provenance scrubber.** A timeline you drag to replay a single
inference pass: as you scrub, the zone map re-lights to that moment, the merge
re-prunes, the tool calls re-fire. It makes "replay the trace" a physical
gesture. This is the element the console is remembered by, and it embodies the
core claim (Architecture §7, provenance faithfulness) as an interaction.

## 4. The two views

### 4.1 Individual view
- **Stage:** the operator's own zone map, live during training/inference.
- **Rail:**
  - Run controls (start/pause, rung selector L0–L3, config hash visible).
  - Zone inspector: tap a zone, see its core type (SSM/attention), its current
    activation, confidence, latency, internal state summary.
  - Trace scrubber for the most recent passes.
  - Health: the commodity metrics (loss, throughput, memory) live here, quiet,
    in the rail... present but not the hero.
- **Tone of copy:** plain verbs, what the operator controls. "Start run," not
  "Initialize training job." The config hash is shown because reproducibility is
  the floor (Research Strategy §8), and showing it is a trust gesture.

### 4.2 Federation view
- **Stage:** the **shared model** as one zone map, with contributions flowing in
  from nodes. A person sees their own node's contribution highlighted against the
  collective. Signal from many nodes converging on the shared zones is the
  emotional core of the federation view... the thing that makes collective
  training feel real.
- **Rail:**
  - Node roster: who is training, which zone(s) they own, their status
    (training, submitted, gathered, timed_out... mirroring AsyncGather states).
  - Gather state: the deadline window, what has arrived, what is still pending.
    Render the async gather honestly, including stragglers.
  - On-chain provenance: the committed trace hashes, with a verify action that
    replays and confirms. This is where Citrate shows up... auditability you can
    click.
  - Composition panel: which zones are swappable now (composition rules), so a
    federation can see where it can evolve a single zone.
- **Grandma-proof bar:** the federation view must make a node operator's path
  obvious... pull verified shards, train, submit. The complexity of the merge and
  the topology is *visible but not required reading* for a node operator. Show the
  beauty; do not demand they understand it to participate.

## 5. What each screen must communicate (from the specs)

Tie every visual element to a spec property so the design stays honest:

| Visual element | Spec source | What it must convey |
|----------------|-------------|----------------------|
| Zone map nodes | Architecture §4 | six zones, core type, activation strength |
| Edges between zones | Architecture §5.1 | fixed topology... only declared edges exist |
| Edge thickness/brightness | Architecture §5.2 | learned per-input modulation |
| Dimming + drop animation | Architecture §6 | merge pruning the bottom 70–80% |
| Surviving zone glow | Architecture §6 | re-weighted survivors composing |
| Trace scrubber | Architecture §7 | replayable provenance, faithfulness |
| MX panel | Architecture §8 | the state machine; the action gate as a visible checkpoint |
| CX badge | Architecture §4.5 | verification pass/fail/unverified (green/dim/neutral) |
| Gather window | Formal §A.2 | deadline, arrivals, stragglers |
| On-chain verify | Formal Gate 4 | commit hash, replay, confirm |

The action gate (`ACTION_GATE`, Architecture §8) deserves special visual weight
in the MX panel... it is a safety checkpoint, and the design should make "nothing
acts before this gate" legible. A held action waiting on approval should look
held, not hidden.

## 6. Motion (deliberate, not decorative)

- **Signal flow** along edges during a pass... the one ambient motion that earns
  its place because it *is* the content.
- **Prune drop** when the merge cuts low scorers... a quick, honest fall, not a
  flourish.
- **Scrub re-light** as the provenance scrubber moves... the map re-lighting to a
  past moment is the signature interaction; make it crisp.
- Respect reduced-motion: the map stays readable as a static state with the same
  information, motion off.

Everything else stays still. The page should not twitch.

## 7. Copy principles (house style + frontend-design)

- Ellipses over em dashes. No "not X but Y." Grounded, specific.
- Name things by what the operator controls, never by how the system is built. A
  person "pauses a run," they do not "halt the orchestration loop."
- Empty and failed states are direction, not mood. A timed-out zone says what
  happened and what the operator can do, in the interface's voice.
- The config hash, the trace hash, the verification badge are trust surfaces.
  Show them plainly. Do not dress them up.

## 8. Quality floor

Responsive to mobile (stage over collapsible rail). Visible keyboard focus.
Reduced motion respected. Tabular figures align. Zone color language is
consistent everywhere a zone is named. The provenance scrubber works as the
signature on both desktop and mobile.

## 9. What to build first

A single-pass **demo of the hero**: load one prompt, run it through a (mocked or
real) NAT pass, and render the living zone map with the provenance scrubber. If
that one screen lands... if a person watches a prompt light up the zones, watches
the merge prune, and then scrubs back through it... the console has proven its
thesis and the rest is elaboration.
