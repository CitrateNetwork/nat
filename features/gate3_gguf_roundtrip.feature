# Gate 3 — Trainable and portable (Formal Scaffold §B.4). NOT YET IMPLEMENTED
# (L1 work): the ecosystem onramp. Tracked here so the acceptance bar is fixed
# before the build. See critique remediation #7 (docs/PLANSET): the
# Ollama-loadable export is the flattened-dense form; the sidecar carries the
# zone graph.
Feature: GGUF round-trip and Ollama-class load
  So that the ecosystem onramp holds

  Scenario: NAT exports to valid GGUF
    Given a trained NAT model at rung "L1"
    When I export to GGUF with the sidecar
    Then a standard GGUF loader loads the tensor container without error

  Scenario: Sidecar-unaware runtime runs the model opaquely
    Given an Ollama-class runtime that ignores the sidecar
    And a NAT export of kind "FlattenedDense"
    When it loads the NAT GGUF
    Then it runs inference as an opaque transformer
    And it produces coherent output

  Scenario: NAT-aware runtime runs the full zone pass
    Given a NAT-aware runtime
    When it loads the same GGUF plus sidecar
    Then it runs the six-zone pass
    And it emits the provenance trace

  Scenario: Routing differentiates by prompt class
    Given prompt classes "math", "narrative", "sensory"
    When I run inference on each class
    Then the dominant activated zones differ by class
    And the difference exceeds the configured significance threshold
