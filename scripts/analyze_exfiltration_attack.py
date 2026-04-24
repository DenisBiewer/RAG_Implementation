from __future__ import annotations

"""
Placeholder analyzer for the exfiltration attack.

This script is intentionally empty for now because the exfiltration attack
schema and success criteria have not been implemented yet. Once that attack
exists, this script should read logs/attacks/exfiltration/<dataset>/ and write
compact files under results/attacks/exfiltration/<dataset>/.
"""


def main() -> None:
    """Explain why this analyzer is not implemented yet."""
    raise NotImplementedError(
        "Exfiltration attack analysis is not implemented yet. "
        "Implement the exfiltration attack and leakage criteria first."
    )


if __name__ == "__main__":
    main()
