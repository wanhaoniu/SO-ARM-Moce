#!/usr/bin/env python3
"""Read current soarmMoce state."""

from __future__ import annotations

from soarmmoce_cli_common import run_and_print
from soarmmoce_sdk import SoArmMoceController


def main() -> None:
    run_and_print(lambda: SoArmMoceController().read())


if __name__ == "__main__":
    main()
