"""Frozen task sets for the formal granularity main study.

Pilot code lives under ``experiments/pilot_granularity/`` and is treated as
historical exploration only; do not add formal main-study results there.
"""

from __future__ import annotations

# Primary table: 6 tasks for the main paper.
MAIN_SIX: tuple[str, ...] = (
    "ClockTimerEntry",
    "SimpleCalendarAddOneEvent",
    "ExpenseAddSingle",
    "FilesMoveFile",
    "RecipeAddSingleRecipe",
    "SimpleSmsSendReceivedAddress",
)

# Backup if a main task is temporarily unstable.
BACKUP_TASKS: tuple[str, ...] = ("MarkorMoveNote",)

# Sanity / intent-only calibration only (not in the formal 6).
SANITY_CALIBRATION_TASKS: tuple[str, ...] = (
    "ContactsAddContact",
    "SimpleSmsSend",
)

# Explicitly excluded from formal main results (document rationale in README).
EXCLUDED_FROM_MAIN: tuple[str, ...] = (
    "RetroCreatePlaylist",
    "MarkorCreateNote",
    # Wi‑Fi–related tasks: keep out of primary table until evaluators are stable.
    "SystemWifiTurnOn",
    "SystemWifiTurnOff",
    "TurnOnWifiAndOpenApp",
    "TurnOffWifiAndTurnOnBluetooth",
)

ALL_SUPPORTED_FOR_RUNNER: tuple[str, ...] = (
    MAIN_SIX + BACKUP_TASKS + SANITY_CALIBRATION_TASKS
)
