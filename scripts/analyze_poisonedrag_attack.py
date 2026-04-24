from __future__ import annotations

"""
Placeholder analyzer for the PoisonedRAG attack.

This script is intentionally empty for now because the PoisonedRAG attack
schema, dominance metric inputs, and poisoned-answer success criteria have not
been implemented yet. Once that attack exists, this script should read
logs/attacks/poisoned_rag/<dataset>/ and write compact files under
results/attacks/poisoned_rag/<dataset>/.
"""


def main() -> None:
    """Explain why this analyzer is not implemented yet."""
    raise NotImplementedError(
        "PoisonedRAG attack analysis is not implemented yet. "
        "Implement the poisoned corpus and dominance/success criteria first."
    )


if __name__ == "__main__":
    main()
