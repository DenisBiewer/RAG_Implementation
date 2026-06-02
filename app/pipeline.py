from __future__ import annotations

"""
End-to-end query execution for the current RAG pipeline.

This module coordinates the major runtime stages of one experiment run:
- query embedding
- retrieval
- prompt construction
- generation
- output-stage rule-based filtering
- structured logging output assembly

The pipeline is designed so that retrieval-, prompt-, and output-stage
defenses can be toggled from `RunConfig.defense` without changing the overall
record structure used for experiment logging.
"""

from app.logging_utils import StageTimer, new_run_id
from app.output_filters import apply_output_defenses
from app.prompt_builder import build_prompt_messages
from app.retrieval import retrieve_top_k, retrieve_with_defenses
from app.schemas import DefenseConfig, RetrievedChunk, RunConfig
from app.seed_config import DEFAULT_SEED, seed_everything


TOP_LEVEL_DEFENSE_FLAGS = (
    "use_mmr",
    "use_retrieval_filtering",
    "use_prompt_isolation",
    "use_instruction_hierarchy",
    "use_grounding",
    "use_prompt_context_flags",
    "use_output_filtering",
    "use_sensitive_info_filter",
)
DEFENSE_LATENCY_KEYS = (
    "mmr",
    "retrieval_filtering",
    "prompt_isolation",
    "instruction_hierarchy",
    "grounding",
    "prompt_context_flags",
    "sensitive_info_filter",
)
INTERNAL_DECISION_KEYS = {"normalized_decisions", "defense_trace", "defense_latencies_ms"}


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


def _top_level_defenses_enabled(defense: DefenseConfig) -> list[str]:
    """
    Return the enabled top-level defense flags for run-level logging.

    Output-filter flags are included here so run logs capture output-stage
    filtering consistently alongside the other defense families.
    """
    return [
        flag_name
        for flag_name in TOP_LEVEL_DEFENSE_FLAGS
        if getattr(defense, flag_name)
    ]


def _merge_defense_decisions(
    retrieval_decisions: dict,
    prompt_decisions: dict,
    output_decisions: dict,
    defense: DefenseConfig,
) -> dict:
    """
    Merge retrieval-, prompt-, and output-stage defense logs into one dictionary.

    Args:
        retrieval_decisions: Decisions recorded during retrieval.
        prompt_decisions: Decisions recorded during prompt construction.
        output_decisions: Decisions recorded during output-stage rule-based filtering.
        defense: Active defense toggle configuration for the current run.

    Returns:
        One merged defense decision dictionary suitable for JSONL logging.
    """
    merged = {
        key: value
        for key, value in retrieval_decisions.items()
        if key not in INTERNAL_DECISION_KEYS
    }
    merged["defenses_enabled"] = _top_level_defenses_enabled(defense)

    if "prompt_context_flags" in prompt_decisions:
        merged["prompt_context_flags"] = prompt_decisions["prompt_context_flags"]

    for key in ("sensitive_info_filter",):
        if key in output_decisions:
            merged[key] = output_decisions[key]

    return merged


def _trace_entry_from_retrieved_chunk(chunk: RetrievedChunk) -> dict:
    """Serialize one final retrieved chunk for the run-level defense trace."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "rank": chunk.rank,
        "score": round(float(chunk.score), 6),
        "is_malicious": chunk.is_malicious,
        "attack_type": chunk.attack_type,
        "document_type": chunk.document_type,
        "topic_id": chunk.topic_id,
        "entered_final_prompt": True,
    }


def _build_pass_through_trace(retrieved_chunks: list[RetrievedChunk]) -> dict:
    """
    Build a pass-through retrieval trace for baseline and non-retrieval defenses.

    When no retrieval-stage defense is enabled, each stage mirrors the same
    dense top-k ranking so downstream analysis can rely on a stable shape.
    """
    entries = [
        _trace_entry_from_retrieved_chunk(chunk)
        for chunk in retrieved_chunks
    ]
    return {
        "raw_similarity_top_k": list(entries),
        "after_retrieval_filtering": list(entries),
        "after_mmr": list(entries),
    }


def _build_defense_trace(
    retrieval_decisions: dict,
    retrieved_chunks: list[RetrievedChunk],
) -> dict:
    """Assemble the top-level `defense_trace` object for one run."""
    trace = dict(retrieval_decisions.get("defense_trace") or _build_pass_through_trace(retrieved_chunks))
    trace["final_prompt_chunks"] = [
        _trace_entry_from_retrieved_chunk(chunk)
        for chunk in retrieved_chunks
    ]
    return trace


def _collect_normalized_decisions(
    retrieval_decisions: dict,
    prompt_decisions: dict,
    output_decisions: dict,
) -> list[dict]:
    """Collect normalized detector/filter decisions across all pipeline stages."""
    normalized_decisions: list[dict] = []
    for stage_decisions in (retrieval_decisions, prompt_decisions, output_decisions):
        normalized_decisions.extend(stage_decisions.get("normalized_decisions", []))
    return normalized_decisions


def _zeroed_defense_latencies() -> dict[str, float]:
    """Return a stable zero-filled latency map for all integrated defenses."""
    return {name: 0.0 for name in DEFENSE_LATENCY_KEYS}


def _collect_defense_latencies(
    retrieval_decisions: dict,
    prompt_decisions: dict,
    output_decisions: dict,
) -> dict[str, float]:
    """Merge per-stage direct defense timings into one stable top-level map."""
    merged = _zeroed_defense_latencies()
    for stage_decisions in (retrieval_decisions, prompt_decisions, output_decisions):
        for name, value in stage_decisions.get("defense_latencies_ms", {}).items():
            if name in merged:
                merged[name] = round(float(value), 3)
    return merged


def _token_count_from_ids(token_ids) -> int:
    """Return the token count from a tokenizer output sequence."""
    if token_ids is None:
        return 0
    if hasattr(token_ids, "shape"):
        return int(token_ids.shape[-1])
    if isinstance(token_ids, list):
        if token_ids and isinstance(token_ids[0], list):
            return len(token_ids[0])
        return len(token_ids)
    return len(token_ids)


def _count_text_tokens(generator, text: str) -> int:
    """Count tokens for a plain text string using the generator tokenizer."""
    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is None:
        return 0
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded["input_ids"]
    return _token_count_from_ids(input_ids)


def _count_prompt_tokens(generator, messages: list[dict[str, str]]) -> int:
    """Count tokens for the full chat prompt sent to the generator."""
    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return 0
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )
    return _token_count_from_ids(token_ids)


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
        Retrieval-, prompt-, and output-stage filtering are supported through
        the same stable logging interface.
    """
    seed_everything(config.seed)
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
    retrieved_context_tokens = _count_text_tokens(
        generator,
        prompt_decisions.get("_context_text", ""),
    )
    prompt_tokens = _count_prompt_tokens(generator, messages)

    t = timer.start("generation")
    output = generator.generate(
        messages,
        max_new_tokens=config.max_new_tokens,
    )
    timer.stop("generation", t)
    output_tokens = _count_text_tokens(generator, output)

    t = timer.start("output_filtering")
    final_output, output_decisions = apply_output_defenses(output, config.defense)
    timer.stop("output_filtering", t)

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
        "defense_trace": _build_defense_trace(retrieval_decisions, retrieved),
        "defense_decision_log": _collect_normalized_decisions(
            retrieval_decisions,
            prompt_decisions,
            output_decisions,
        ),
        "defense_latencies_ms": _collect_defense_latencies(
            retrieval_decisions,
            prompt_decisions,
            output_decisions,
        ),
        "defense_decisions": _merge_defense_decisions(
            retrieval_decisions,
            prompt_decisions,
            output_decisions,
            config.defense,
        ),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "retrieved_context_tokens": retrieved_context_tokens,
        "latencies_ms": timer.timings,
    }

    return record


def run_baseline_query(
    query_record,
    embedder,
    index,
    generator,
    top_k: int = 5,
    seed: int = DEFAULT_SEED,
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
