# soarmmoce-openclaw-skill

OpenClaw tool-call adapter layer for `soarmmoce-sdk`.

This package only does:
- OpenAI-style tool schemas
- Argument validation
- Dispatch to `soarmmoce_sdk.Robot` APIs
- Structured tool results (`ok/result/error`)

It does **not** include GUI/VTK/PyBullet workflows or prompt engineering logic.

## Install

```bash
pip install soarmmoce-openclaw-skill
```

Dependency:
- `soarmmoce-sdk>=0.1.0`

## Configuration

Runtime config priority:
1. `SOARMMOCE_CONFIG` (path to yaml)
2. `SOARMMOCE_TRANSPORT` (`mock`/`tcp`/`serial`)
3. fallback: `Robot()` default config from SDK

Examples:

```bash
export SOARMMOCE_CONFIG=/abs/path/to/soarm_moce.yaml
export SOARMMOCE_TRANSPORT=mock
```

## Python usage

```python
from soarmmoce_openclaw_skill import dispatch, TOOL_SCHEMAS

resp = dispatch("get_robot_state", {})
print(resp)
```

## CLI usage

```bash
soarmmoce-skill get_robot_state
soarmmoce-skill move_robot_arm --x 0.30 --y 0.00 --z 0.20 --wait true
soarmmoce-skill set_gripper --open-ratio 0.5
soarmmoce-skill stop_robot
```

## Development

```bash
python -m pytest -q
python -m build
```
