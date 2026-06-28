-------------------------- MODULE WeightCommitment --------------------------
\* WS-2 Layer B / WP-B2 — the consensus-grade weight-commitment spec.
\*
\* Companion to `nat-weightspace::commit::canonical_digest`. That function produces a
\* permutation-invariant, Q16-exact, tamper-detecting digest of a model's weights; this
\* spec proves the two properties that make committing such a digest on-chain safe:
\*
\*   (1) CommitmentSound  — a Valid verdict implies the off-chain weights actually match
\*                          the on-chain commitment (you cannot get a Valid for a model
\*                          whose preimage does not hash to the committed digest).
\*   (2) TamperDetection  — if the off-chain weights were mutated after commit so they no
\*                          longer match, verification never returns Valid.
\*
\* `Commitment(w) == w` is the identity here (TLA abstracts the hash); in Rust it is the
\* WL canonical digest. The proof is about the protocol, not the hash's collision
\* resistance, which is a cryptographic assumption.
\*
\* Mirrors the house pattern of citrate-chain's EmbeddedModelCommitment.tla.

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Models,          \* finite set of model ids
    WeightVals       \* finite set of abstract weight contents

Null == "null"

\* Status of a model's on-chain verification.
StatusUnverified == "unverified"
StatusValid      == "valid"
StatusTampered   == "tampered"
Statuses == {StatusUnverified, StatusValid, StatusTampered}

\* The commitment function. Identity in TLA; the WL/SHA-256 digest in nat-weightspace.
Commitment(w) == w

VARIABLES
    published,   \* Models -> WeightVals \cup {Null}   (off-chain weights peers hold)
    commit,      \* Models -> WeightVals \cup {Null}   (on-chain committed digest preimage)
    status       \* Models -> Statuses

vars == <<published, commit, status>>

TypeOK ==
    /\ published \in [Models -> WeightVals \cup {Null}]
    /\ commit    \in [Models -> WeightVals \cup {Null}]
    /\ status    \in [Models -> Statuses]

Init ==
    /\ published = [m \in Models |-> Null]
    /\ commit    = [m \in Models |-> Null]
    /\ status    = [m \in Models |-> StatusUnverified]

\* Publish a model's weights and commit their digest atomically (the honest path).
Publish(m, w) ==
    /\ commit[m] = Null              \* not yet committed
    /\ published' = [published EXCEPT ![m] = w]
    /\ commit'    = [commit    EXCEPT ![m] = Commitment(w)]
    /\ status'    = [status    EXCEPT ![m] = StatusUnverified]

\* Adversary mutates the off-chain weights after commit (the on-chain commit is immutable).
\* Any prior verdict is invalidated and must be recomputed.
Tamper(m, w2) ==
    /\ commit[m] # Null
    /\ w2 # published[m]
    /\ published' = [published EXCEPT ![m] = w2]
    /\ commit'    = commit
    /\ status'    = [status EXCEPT ![m] = StatusUnverified]

\* Verify recomputes the digest of the current off-chain weights and compares to commit.
Verify(m) ==
    /\ commit[m] # Null
    /\ published[m] # Null
    /\ status' = [status EXCEPT ![m] =
                    IF Commitment(published[m]) = commit[m]
                    THEN StatusValid
                    ELSE StatusTampered]
    /\ UNCHANGED <<published, commit>>

Next ==
    \/ \E m \in Models, w \in WeightVals : Publish(m, w)
    \/ \E m \in Models, w \in WeightVals : Tamper(m, w)
    \/ \E m \in Models : Verify(m)

Spec == Init /\ [][Next]_vars

\* ---- Safety properties ----------------------------------------------------

\* A Valid verdict is sound: the off-chain preimage hashes to the on-chain commitment.
CommitmentSound ==
    \A m \in Models :
        status[m] = StatusValid => Commitment(published[m]) = commit[m]

\* Tampered weights are never accepted as Valid.
TamperDetection ==
    \A m \in Models :
        (commit[m] # Null /\ Commitment(published[m]) # commit[m])
            => status[m] # StatusValid

THEOREM Spec => [](TypeOK /\ CommitmentSound /\ TamperDetection)
=============================================================================
