"""
config.py

Defines the experiment configurations used to evaluate the RAG system.

These configurations mirror the defense groupings described in the thesis plan:

- baseline
- retrieval-only
- prompt-only
- output-only

Why this file exists:
- keeps experiment settings separate from pipeline logic
- makes comparisons between configurations reproducible
- ensures that defense combinations are easy to inspect and log

Important thesis note:
Some parameters below are already fixed by the source files, while others are
temporary implementation defaults added to make the code runnable early.
Temporary values are marked clearly with TODO comments.
"""

from app.schemas import RunConfig, DefenseConfig


# -----------------------------------------------------------------------------
# Baseline configuration
# -----------------------------------------------------------------------------
# This represents the undefended system.
#
# Aligned with the system model:
# - top-k retrieval
# - no retrieval filtering
# - plain prompt construction
# - no output filtering
# - deterministic generation
#
# Source support:
# - k = 5 is explicitly specified in the system model.
# - temperature = 0 is explicitly specified in the system model.
# - max_new_tokens = 256 matches the "Max tokens: 256" choice in the system model.
# - seed is mentioned in the logging/evaluation section, but its exact value was not fixed.
BASELINE_CONFIG = RunConfig(
    run_name="baseline",
    # TODO: Exact seed value was not fixed in the source files.
    # Temporary choice: 42.
    # Reason: widely used reproducibility seed; acceptable as a placeholder.
    seed=42,

    # Fixed in source files:
    # System model Stage 4 sets retrieval to top-k with k = 5.
    top_k=5,

    # Fixed in source files:
    # System model Stage 6 sets max output length to 256 tokens.
    max_new_tokens=256,

    # Fixed in source files:
    # System model Stage 6 sets temperature to 0 for deterministic decoding.
    temperature=0.0,

    # Baseline intentionally enables no defenses.
    defense=DefenseConfig(),
)


# -----------------------------------------------------------------------------
# Retrieval-only configuration
# -----------------------------------------------------------------------------
# This configuration enables only retrieval-stage defenses.
#
# Aligned with the defense design:
# - MMR
# - security-aware reranking / filtering
#
# Source support:
# - MMR is explicitly included in retrieval-only.
# - mmr_lambda = 0.5 is explicitly proposed in the defense design.
# - suspicion_threshold = 0.7 is explicitly proposed in the defense design.
# - k remains 5 in the defense design and system model.
RETRIEVAL_ONLY_CONFIG = RunConfig(
    run_name="retrieval_only",

    # TODO: Exact seed value was not fixed in the source files.
    # Temporary choice: 42.
    seed=42,

    # Fixed in source files.
    top_k=5,

    # Fixed in source files:
    # Defense design proposes lambda = 0.5 for MMR.
    mmr_lambda=0.5,

    # Fixed in source files:
    # Defense design proposes a suspicion threshold of 0.7.
    suspicion_threshold=0.7,

    # TODO: max_new_tokens is inherited from RunConfig default (256).
    # This is currently consistent with the system model, but if you want full
    # explicitness in every config, you could also set max_new_tokens=256 here.
    #
    # TODO: temperature is inherited from RunConfig default (0.0).
    # Also consistent with the system model.

    defense=DefenseConfig(
        use_mmr=True,
        use_retrieval_filtering=True,
    ),
)


# -----------------------------------------------------------------------------
# Prompt-only configuration
# -----------------------------------------------------------------------------
# This configuration enables only prompt-construction-stage defenses.
#
# Aligned with the defense design:
# - context isolation / delimitation
# - instruction hierarchy enforcement
# - grounding
# - context filtering / trust flag annotation
#
# Source support:
# - these four defenses are explicitly listed in the prompt-only configuration.
PROMPT_ONLY_CONFIG = RunConfig(
    run_name="prompt_only",

    # TODO: Exact seed value was not fixed in the source files.
    # Temporary choice: 42.
    seed=42,

    # Fixed in source files.
    top_k=5,

    # TODO: max_new_tokens is not repeated here, so it falls back to the
    # RunConfig default of 256, which is consistent with the system model.
    #
    # TODO: temperature is not repeated here, so it falls back to 0.0,
    # which is consistent with the system model.

    defense=DefenseConfig(
        use_prompt_isolation=True,
        use_instruction_hierarchy=True,
        use_grounding=True,
        use_prompt_context_flags=True,
    ),
)


# -----------------------------------------------------------------------------
# Output-only configuration
# -----------------------------------------------------------------------------
# This configuration enables only output-stage defenses.
#
# Aligned with the defense design:
# - sensitive information filtering
# - policy compliance check
#
# Source support:
# - both defenses are explicitly listed in the output-only configuration.
OUTPUT_ONLY_CONFIG = RunConfig(
    run_name="output_only",

    # TODO: Exact seed value was not fixed in the source files.
    # Temporary choice: 42.
    seed=42,

    # Fixed in source files.
    top_k=5,

    # TODO: max_new_tokens is inherited from RunConfig default (256),
    # consistent with the system model.
    #
    # TODO: temperature is inherited from RunConfig default (0.0),
    # consistent with the system model.

    defense=DefenseConfig(
        use_sensitive_info_filter=True,
        use_policy_check=True,
    ),
)