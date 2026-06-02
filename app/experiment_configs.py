from __future__ import annotations

"""Canonical experiment configuration registry shared across runtime and scripts."""

from app.schemas import DefenseConfig, RunConfig
from app.seed_config import DEFAULT_SEED


BASELINE_CONFIG = RunConfig(
    run_name="baseline",
    seed=DEFAULT_SEED,
    top_k=5,
    max_new_tokens=256,
    temperature=0.0,
    defense=DefenseConfig(),
)

RETRIEVAL_ONLY_CONFIG = RunConfig(
    run_name="retrieval_only",
    seed=DEFAULT_SEED,
    top_k=5,
    mmr_lambda=0.5,
    suspicion_threshold=0.7,
    defense=DefenseConfig(
        use_mmr=True,
        use_retrieval_filtering=True,
    ),
)

PROMPT_ONLY_CONFIG = RunConfig(
    run_name="prompt_only",
    seed=DEFAULT_SEED,
    top_k=5,
    defense=DefenseConfig(
        use_prompt_isolation=True,
        use_instruction_hierarchy=True,
        use_grounding=True,
        use_prompt_context_flags=True,
    ),
)

OUTPUT_ONLY_CONFIG = RunConfig(
    run_name="output_only",
    seed=DEFAULT_SEED,
    top_k=5,
    defense=DefenseConfig(
        use_output_filtering=True,
    ),
)

MMR_ONLY_CONFIG = RunConfig(
    run_name="mmr_only",
    seed=DEFAULT_SEED,
    top_k=5,
    mmr_lambda=0.5,
    defense=DefenseConfig(
        use_mmr=True,
    ),
)

RETRIEVAL_FILTERING_ONLY_CONFIG = RunConfig(
    run_name="retrieval_filtering_only",
    seed=DEFAULT_SEED,
    top_k=5,
    suspicion_threshold=0.7,
    defense=DefenseConfig(
        use_retrieval_filtering=True,
    ),
)

PROMPT_ISOLATION_ONLY_CONFIG = RunConfig(
    run_name="prompt_isolation_only",
    seed=DEFAULT_SEED,
    top_k=5,
    defense=DefenseConfig(
        use_prompt_isolation=True,
    ),
)

INSTRUCTION_HIERARCHY_ONLY_CONFIG = RunConfig(
    run_name="instruction_hierarchy_only",
    seed=DEFAULT_SEED,
    top_k=5,
    defense=DefenseConfig(
        use_instruction_hierarchy=True,
    ),
)

GROUNDING_ONLY_CONFIG = RunConfig(
    run_name="grounding_only",
    seed=DEFAULT_SEED,
    top_k=5,
    defense=DefenseConfig(
        use_grounding=True,
    ),
)

PROMPT_CONTEXT_FLAGS_ONLY_CONFIG = RunConfig(
    run_name="prompt_context_flags_only",
    seed=DEFAULT_SEED,
    top_k=5,
    defense=DefenseConfig(
        use_prompt_context_flags=True,
    ),
)

GROUPED_CONFIG_NAMES = [
    BASELINE_CONFIG.run_name,
    RETRIEVAL_ONLY_CONFIG.run_name,
    PROMPT_ONLY_CONFIG.run_name,
    OUTPUT_ONLY_CONFIG.run_name,
]

SINGLE_DEFENSE_CONFIG_NAMES = [
    MMR_ONLY_CONFIG.run_name,
    RETRIEVAL_FILTERING_ONLY_CONFIG.run_name,
    PROMPT_ISOLATION_ONLY_CONFIG.run_name,
    INSTRUCTION_HIERARCHY_ONLY_CONFIG.run_name,
    GROUNDING_ONLY_CONFIG.run_name,
    PROMPT_CONTEXT_FLAGS_ONLY_CONFIG.run_name,
]

ALL_CONFIG_NAMES = GROUPED_CONFIG_NAMES + SINGLE_DEFENSE_CONFIG_NAMES
DEFAULT_GROUPED_CONFIG_NAMES = list(GROUPED_CONFIG_NAMES)

ALL_EXPERIMENT_CONFIGS: dict[str, RunConfig] = {
    config.run_name: config
    for config in (
        BASELINE_CONFIG,
        RETRIEVAL_ONLY_CONFIG,
        PROMPT_ONLY_CONFIG,
        OUTPUT_ONLY_CONFIG,
        MMR_ONLY_CONFIG,
        RETRIEVAL_FILTERING_ONLY_CONFIG,
        PROMPT_ISOLATION_ONLY_CONFIG,
        INSTRUCTION_HIERARCHY_ONLY_CONFIG,
        GROUNDING_ONLY_CONFIG,
        PROMPT_CONTEXT_FLAGS_ONLY_CONFIG,
    )
}
