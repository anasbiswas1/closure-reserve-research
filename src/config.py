"""Central config. Import paths and host presets from here; never hardcode them in notebooks."""
import os
from pathlib import Path

PROJECT = "CLOSURE"
REPO_NAME = "closure-reserve-research"
GH_USER = "anasbiswas1"

DRIVE_ROOT = Path("/content/drive/MyDrive") / f"{PROJECT}_Research"
REPO_DIR = Path(os.environ.get("CLOSURE_REPO_ROOT", str(DRIVE_ROOT / REPO_NAME)))

NOTEBOOKS = REPO_DIR / "notebooks"
SRC = REPO_DIR / "src"
REPORTS = REPO_DIR / "reports"
FIGURES = REPO_DIR / "figures"
DATA = REPO_DIR / "data"
EXTERNAL = REPO_DIR / "external"
SUITE_DATA = REPO_DIR / "bookings_data"

AGENTDOJO_VERSION = "0.1.35"
RUNTIME_VERSION = "2"      # bumped whenever the recording runtime changes; stale records are re-recorded automatically
CLOSURE_VERSION = "3"      # bumped whenever the closure or the gate changes; stale closure tables are rebuilt automatically

# Host presets (plan v2.1, Section 2.3). Costs are in token-equivalents (TE).
PRESETS = {
    "w": 100,            # TE charged per tool call
    "m_out": 400,        # max output tokens of a main-loop model call
    "m_out_c": 300,      # max output tokens of the closure report call
    "k_max": 3,          # max tool calls executed per turn
    "L": 1200,           # per-return cap in tokens, trusted and untrusted alike
    "H_max": 2,          # max holds in HELD state at once
    "Q_max": 3,          # max queued consequential calls carried into closure
    "M": 40,             # max loop iterations; budget, not iteration count, should bind
    "record_ceiling": 80000,   # TE ceiling for unconstrained recording runs
    "P": 300,            # padding tokens per attacked field (attack A1)
}

# Tools whose execution changes the world (plan Section 3.2).
CONSEQUENTIAL = {"hold_booking", "confirm_booking", "cancel_hold", "send_money", "reply_email"}
# Consequential tools that require a prior verification of their arguments (HostVerifyRule).
VERIFY_REQUIRED = {"confirm_booking": "verify_hold", "send_money": "verify_payee"}
# Tools whose returns carry attacker-controlled fields.
UNTRUSTED_TOOLS = {"search_hotels", "get_hotel_terms", "search_cars", "read_inbox", "search_inbox"}

CONDITIONS = ["A0", "A1", "A2", "A3", "A123", "A5", "A6"]
ATTACK_BY_CONDITION = {"A0": None, "A1": "padding", "A2": "transient_error",
                       "A3": "clarification_bait", "A123": "combined", "A4": "pending_flood",
                       "A5": "chaining", "A6": "chaining_padding"}


def ensure_dirs():
    for p in (NOTEBOOKS, SRC, REPORTS, FIGURES, DATA, EXTERNAL, SUITE_DATA):
        p.mkdir(parents=True, exist_ok=True)
