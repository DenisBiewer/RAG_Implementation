from __future__ import annotations

"""
End-to-end query execution for the current RAG pipeline.

This module coordinates the major runtime stages of one experiment run:
- query embedding
- retrieval
- prompt construction
- generation
- output filtering
- structured logging output assembly

The pipeline is designed so that retrieval-, prompt-, and output-stage
defenses can be toggled from `RunConfig.defense` without changing the overall
record structure used for experiment logging.
"""

import random

import numpy as np
import torch

from app.logging_utils import StageTimer, new_run_id
from app.output_filters import apply_output_defenses
from app.prompt_builder import build_prompt_messages
from app.retrieval import retrieve_top_k, retrieve_with_defenses
from app.schemas import RunConfig


def set_seed(seed: int) -> None:
    """
    Seed the local random number generators used by the current pipeline.

    Args:
        seed: Deterministic seed value for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_retrieval(query_embedding, index, config: RunConfig) -> tuple[list, dict]:
    """
    Execute retrieval according to the enabled defense toggles in `RunConfig`.

    Args:
        query_embedding: Dense embedding of the input query.
        index: Loaded `FaissIndex`.
        config: Active experiment configuration.

    Returns:
        A pair `(retrieved_chunks, defense_decisions)`.
    """
    if config.defense.use_mmr or config.defense.use_retrieval_filtering:
        return retrieve_with_defenses(
            index=index,
            query_embedding=query_embedding,
            top_k=config.top_k,
            defense=config.defense,
            mmr_lambda=config.mmr_lambda,
            suspicion_threshold=config.suspicion_threshold,
        )

    return retrieve_top_k(index, query_embedding, top_k=config.top_k), {
        "defenses_enabled": [],
    }


def _merge_defense_decisions(
    retrieval_decisions: dict,
    prompt_decisions: dict,
    output_decisions: dict,
) -> dict:
    """
    Merge retrieval-, prompt-, and output-stage defense logs into one dictionary.

    Args:
        retrieval_decisions: Decisions recorded during retrieval.
        prompt_decisions: Decisions recorded during prompt construction.
        output_decisions: Decisions recorded during output filtering.

    Returns:
        One merged defense decision dictionary suitable for JSONL logging.
    """
    merged = dict(retrieval_decisions)
    enabled = list(merged.get("defenses_enabled", []))

    for defense_name in prompt_decisions.get("prompt_defenses_enabled", []):
        if defense_name not in enabled:
            enabled.append(defense_name)

    for defense_name in output_decisions.get("output_defenses_enabled", []):
        if defense_name not in enabled:
            enabled.append(defense_name)

    merged["defenses_enabled"] = enabled

    if "prompt_context_flags" in prompt_decisions:
        merged["prompt_context_flags"] = prompt_decisions["prompt_context_flags"]

    for key in ("sensitive_info_filter", "policy_check"):
        if key in output_decisions:
            merged[key] = output_decisions[key]

    return merged


def run_query(
    query_record,
    embedder,
    index,
    generator,
    config: RunConfig,
) -> dict:
    """
    Run one RAG query end-to-end under a specific experiment configuration.

    Args:
        query_record: One evaluation query object.
        embedder: Loaded query embedder.
        index: Loaded retrieval index.
        generator: Loaded text generator.
        config: Active experiment configuration.

    Returns:
        A dictionary ready for JSONL logging.

    Notes:
        Retrieval-, prompt-, and output-stage defenses are supported through the
        same stable logging interface.
    """
    set_seed(config.seed)
    timer = StageTimer()

    t = timer.start("query_embedding")
    query_embedding = embedder.encode([query_record.question])
    timer.stop("query_embedding", t)

    t = timer.start("retrieval")
    retrieved, retrieval_decisions = _run_retrieval(query_embedding, index, config)
    timer.stop("retrieval", t)

    t = timer.start("prompt_building")
    messages, prompt_decisions = build_prompt_messages(
        query_record.question,
        retrieved,
        config.defense,
    )
    timer.stop("prompt_building", t)

    t = timer.start("generation")
    output = generator.generate(
        messages,
        max_new_tokens=config.max_new_tokens,
    )
    timer.stop("generation", t)

    final_output, output_decisions = apply_output_defenses(output, config.defense)

    record = {
        "run_id": new_run_id(),
        "query_id": query_record.query_id,
        "query": query_record.question,
        "expected_answer": query_record.answer,
        "supporting_facts": query_record.supporting_facts.model_dump(),
        "seed": config.seed,
        "config_name": config.run_name,
        "retrieved_chunks": [chunk.model_dump() for chunk in retrieved],
        "prompt": messages,
        "model_output": output,
        "final_output": final_output,
        "defense_decisions": _merge_defense_decisions(
            retrieval_decisions,
            prompt_decisions,
            output_decisions,
        ),
        "latencies_ms": timer.timings,
    }

    return record


def run_baseline_query(
    query_record,
    embedder,
    index,
    generator,
    top_k: int = 5,
    seed: int = 42,
) -> dict:
    """
    Backward-compatible wrapper for the plain baseline configuration.

    Existing scripts in the repository already call `run_baseline_query`.
    Keeping this wrapper avoids breaking them while the general `run_query`
    function becomes the preferred entry point for experiment execution.
    """
    config = RunConfig(
        run_name="baseline",
        seed=seed,
        top_k=top_k,
        max_new_tokens=256,
        temperature=0.0,
    )
    return run_query(
        query_record=query_record,
        embedder=embedder,
        index=index,
        generator=generator,
        config=config,
    )
