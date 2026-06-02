from __future__ import annotations

import hashlib
import json
import os
import random

import numpy as np
import torch


DEFAULT_SEED = 42
_NUMPY_SEED_MODULUS = 2**32
_DERIVED_SEED_MODULUS = 2**63


def _stable_seed_part(value: object) -> str:
    """Serialize one seed component into a stable string representation."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def resolve_seed(explicit_seed: int | None = None) -> int:
    """
    Resolve the effective project seed.

    Resolution order:
    1. explicit function argument
    2. `RAG_SEED` environment variable
    3. `DEFAULT_SEED`
    """
    if explicit_seed is not None:
        return int(explicit_seed)

    env_seed = os.getenv("RAG_SEED")
    if env_seed is not None:
        try:
            return int(env_seed)
        except ValueError as exc:
            raise ValueError(
                f"Invalid RAG_SEED value {env_seed!r}. Expected an integer."
            ) from exc

    return DEFAULT_SEED


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs for reproducible stochastic behavior.

    `deterministic_torch=False` keeps the current model/runtime behavior unless
    a caller explicitly opts into stricter deterministic Torch flags.
    """
    normalized_seed = int(seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed % _NUMPY_SEED_MODULUS)
    torch.manual_seed(normalized_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable positive integer sub-seed from a base seed and context."""
    payload = "||".join(
        [_stable_seed_part(base_seed), *(_stable_seed_part(part) for part in parts)]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % _DERIVED_SEED_MODULUS
