from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorProfile:
    """
    Preset for generator model selection in thesis experiments.

    The intended workflow is:
    - use `DEV_GENERATOR_PROFILE` for fast iteration
    - use `MAIN_GENERATOR_PROFILE` for the main attack/defense runs
    - use `VALIDATION_GENERATOR_PROFILE` on a smaller subset to confirm trends
    """

    name: str
    model_name: str
    device: str = "auto"
    quantization: str = "none"
    max_new_tokens: int = 256
    description: str = ""


DEV_GENERATOR_PROFILE = GeneratorProfile(
    name="dev",
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    device="auto",
    quantization="none",
    max_new_tokens=256,
    description="Fast local development profile for debugging the pipeline.",
)


MAIN_GENERATOR_PROFILE = GeneratorProfile(
    name="main",
    model_name="Qwen/Qwen2.5-3B-Instruct",
    device="auto",
    quantization="4bit",
    max_new_tokens=256,
    description="Primary thesis profile for the main RAG attack/defense experiments.",
)


VALIDATION_GENERATOR_PROFILE = GeneratorProfile(
    name="validation",
    model_name="Qwen/Qwen2.5-7B-Instruct",
    device="auto",
    quantization="4bit",
    max_new_tokens=256,
    description="Higher-capacity validation profile for a smaller confirmatory subset.",
)


GENERATOR_PROFILES = {
    DEV_GENERATOR_PROFILE.name: DEV_GENERATOR_PROFILE,
    MAIN_GENERATOR_PROFILE.name: MAIN_GENERATOR_PROFILE,
    VALIDATION_GENERATOR_PROFILE.name: VALIDATION_GENERATOR_PROFILE,
}
