-------------------------- MODULE LoraRegistration --------------------------
\* WS-3 / WP-G6 — registration soundness for generated LoRA adapters.
\*
\* Companion to `nat-lora::commit::lora_commitment` + the `LoRAFactory` landing. A
\* generated adapter must traverse commit -> (eval-gate) -> verify -> register, and the
\* registry must guarantee:
\*
\*   RegistrationSound    — a Registered adapter's off-chain factors hash to its committed
\*                          digest (you cannot register factors that differ from what was
\*                          committed).
\*   OnlyVerifiedRegisters— a Registered adapter passed the eval/promotion gate (the
\*                          McNemar + no-regression ratchet from WP-G4); an unpromoted
\*                          adapter can never reach Registered.
\*   TamperRejected       — if the off-chain factors were mutated after commit so they no
\*                          longer match, the adapter is never Verified or Registered.
\*
\* There is NO existing TLA+ spec for LoRA registration (only LearningCycleLifecycle.tla,
\* a cycle state machine); this proof is net-new. `Commitment(f) == f` is the identity
\* abstraction (Q16 WL-style digest in Rust). Mirrors the EmbeddedModelCommitment idiom.

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Adapters,   \* finite set of adapter ids
    Factors     \* finite set of abstract factor contents (the A/B matrices)

Null == "null"

StUncommitted == "uncommitted"
StCommitted   == "committed"
StVerified    == "verified"
StRegistered  == "registered"
StRejected    == "rejected"
Statuses == {StUncommitted, StCommitted, StVerified, StRegistered, StRejected}

Commitment(f) == f   \* identity in TLA; the Q16 lora_commitment digest in Rust

VARIABLES
    offFactors,  \* Adapters -> Factors \cup {Null}   (off-chain A/B the creator holds)
    commit,      \* Adapters -> Factors \cup {Null}   (on-chain committed preimage)
    promoted,    \* Adapters -> BOOLEAN                (passed the eval/McNemar gate)
    status       \* Adapters -> Statuses

vars == <<offFactors, commit, promoted, status>>

TypeOK ==
    /\ offFactors \in [Adapters -> Factors \cup {Null}]
    /\ commit     \in [Adapters -> Factors \cup {Null}]
    /\ promoted   \in [Adapters -> BOOLEAN]
    /\ status     \in [Adapters -> Statuses]

Init ==
    /\ offFactors = [a \in Adapters |-> Null]
    /\ commit     = [a \in Adapters |-> Null]
    /\ promoted   = [a \in Adapters |-> FALSE]
    /\ status     = [a \in Adapters |-> StUncommitted]

\* Commit a generated adapter's factors and its digest atomically.
Commit(a, f) ==
    /\ status[a] = StUncommitted
    /\ offFactors' = [offFactors EXCEPT ![a] = f]
    /\ commit'     = [commit     EXCEPT ![a] = Commitment(f)]
    /\ status'     = [status     EXCEPT ![a] = StCommitted]
    /\ UNCHANGED promoted

\* The off-chain eval/McNemar gate result (WP-G4) — pass or fail, nondeterministic here.
SetPromoted(a, b) ==
    /\ status[a] = StCommitted
    /\ promoted' = [promoted EXCEPT ![a] = b]
    /\ UNCHANGED <<offFactors, commit, status>>

\* Adversary mutates the off-chain factors after commit (commit is immutable). Only a
\* not-yet-registered adapter can be tampered; this demotes it back to committed.
Tamper(a, f2) ==
    /\ status[a] \in {StCommitted, StVerified}
    /\ commit[a] # Null
    /\ f2 # offFactors[a]
    /\ offFactors' = [offFactors EXCEPT ![a] = f2]
    /\ status'     = [status     EXCEPT ![a] = StCommitted]
    /\ UNCHANGED <<commit, promoted>>

\* Verify recomputes the digest and checks the eval gate; only a digest match AND a
\* promoted adapter becomes Verified, else Rejected.
Verify(a) ==
    /\ status[a] = StCommitted
    /\ promoted[a]
    /\ status' = [status EXCEPT ![a] =
                    IF Commitment(offFactors[a]) = commit[a]
                    THEN StVerified
                    ELSE StRejected]
    /\ UNCHANGED <<offFactors, commit, promoted>>

Register(a) ==
    /\ status[a] = StVerified
    /\ status' = [status EXCEPT ![a] = StRegistered]
    /\ UNCHANGED <<offFactors, commit, promoted>>

\* Terminal self-loop so fully-settled states do not deadlock the model.
Done ==
    /\ \A a \in Adapters : status[a] \in {StRegistered, StRejected}
    /\ UNCHANGED vars

Next ==
    \/ \E a \in Adapters, f \in Factors : Commit(a, f)
    \/ \E a \in Adapters, b \in BOOLEAN : SetPromoted(a, b)
    \/ \E a \in Adapters, f \in Factors : Tamper(a, f)
    \/ \E a \in Adapters : Verify(a)
    \/ \E a \in Adapters : Register(a)
    \/ Done

Spec == Init /\ [][Next]_vars

\* ---- Safety properties ----------------------------------------------------

RegistrationSound ==
    \A a \in Adapters :
        status[a] = StRegistered => Commitment(offFactors[a]) = commit[a]

OnlyVerifiedRegisters ==
    \A a \in Adapters :
        status[a] = StRegistered => promoted[a]

TamperRejected ==
    \A a \in Adapters :
        (commit[a] # Null /\ Commitment(offFactors[a]) # commit[a])
            => status[a] \notin {StVerified, StRegistered}

THEOREM Spec => [](TypeOK /\ RegistrationSound /\ OnlyVerifiedRegisters /\ TamperRejected)
=============================================================================
