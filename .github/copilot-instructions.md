# Copilot / AI assistant quick-start for this repository

This file gives focused, actionable pointers so an AI coding agent can be productive quickly in this repo.

High-level system overview
- This repository mixes two major concerns:
  - ArduPilot flight-control firmware (C++): under `ardupilot/` (many subprojects like `ArduCopter/`, `ArduPlane/`, `Rover/`). This is a large, modular C++ codebase using a custom build flow (waf/Make). Look for `waf`, `Makefile.waf`, and `wscript` inside `ardupilot/`.
  - Local tooling and vision/inference scripts (Python): top-level directories like `script/`, `rdk/`, and `model/` contain Python tools for dataset handling, model conversion and live inference (examples: `script/live-inference.py`, `script/main.py`, `rdk/export_to_onnx.py`, `rdk/calib_maker.py`). These are kept separate from the flight control code.

Key integration points and patterns
- MAVLink / GCS integration: look for `GCS_Mavlink.cpp` / `GCS_Mavlink.h` in `AntennaTracker/` and `ArduCopter/` — these are good anchors for code interacting with ground stations.
- Flight modes are structured by vehicle in `ArduCopter/` (e.g., `mode_auto.cpp`, `mode_manual.cpp`) — adding or changing behavior typically means editing or adding a `mode_*.cpp` and corresponding header.
- Many core subsystems follow the AP_ prefix naming (AP_Arming, AP_State, AP_Nav*) and are split into paired .cpp/.h files. Follow existing naming and file-location conventions.

Developer workflows (what to run)
- Building ArduPilot (C++): inspect `ardupilot/` for the recommended flow. Typical commands (run from repo root):
  - cd into `ardupilot/` and use the included waf wrapper: `./waf configure --board <board>` then `./waf build` or `./waf`. There is also a `Makefile` and `Makefile.waf` for CI and legacy flows. If you see failures, consult `ardupilot/BUILD.md` and the developer wiki linked in `ardupilot/README.md`.
- Running Python tools / inference:
  - Use a virtualenv: `python3 -m venv .venv && source .venv/bin/activate` then `pip install -r requirements.txt` if the project contains one, or inspect `script/` and `rdk/` for imports to determine packages.
  - Example entry points: `script/main.py`, `script/live-inference.py`, `script/generate-object.py`. `rdk/export_to_onnx.py` is used for model conversion.

Project-specific conventions and notes for edits
- C++ naming: files and classes use descriptive subsystem prefixes (AP_, GCS_, mode_, etc.). New subsystems or modes should follow the same file pattern and live under the matching vehicle folder (e.g., new copter code → `ArduCopter/`).
- Parameter and board configs: see `mav.parm` and `ardupilot/mav.parm` files — parameter sets are stored in root-level parm files.
- Avoid changing broad build infrastructure in one PR. Small, targeted changes (single subsystem or script) are preferred.

Where to look for examples
- Flight-code examples: `ArduCopter/` (modes, fail-safes, logging). Use `Log.cpp` and `events.cpp` for logging/telemetry patterns.
- Ground/control integrations: `AntennaTracker/GCS_*.cpp` and `ArduCopter/GCS_Mavlink.cpp`.
- Vision/tools: `script/` and `rdk/` (model, export and calibration utilities).

What an AI agent should do first when editing code here
1. Locate the vehicle/subsystem relevant to the change (search for AP_* or mode_* in `ardupilot/`).
2. Find and run a minimal build locally for that component (`./waf build` in `ardupilot/`), or run Python tools in a virtualenv to verify runtime behavior.
3. When adding features, use existing naming and file placement conventions and include a short comment referencing the related module/file (e.g., `// See ArduCopter/mode_auto.cpp`).

Files referenced in this guidance (examples to cite in PRs)
- `ardupilot/` (C++ flight code, waf build)
- `ardupilot/ArduCopter/` (copter-specific modes & logic)
- `AntennaTracker/GCS_Mavlink.cpp` (ground station comms)
- `script/live-inference.py`, `script/main.py` (vision/inference entry points)
- `rdk/export_to_onnx.py`, `rdk/calib_maker.py` (model tooling)

If something is ambiguous
- Ask the human maintainer for: preferred board/target for testing, the Python dependency list (requirements.txt) or how the project is run in CI, and whether changes to ArduPilot build scripts are allowed in this repository copy.

Done. If you want, I can now:
- open a draft PR with this file added, or
- update the draft with examples of exact commands you use locally (board names, Python env setup) if you provide them.

Please review and tell me any missing commands or local conventions to include.
