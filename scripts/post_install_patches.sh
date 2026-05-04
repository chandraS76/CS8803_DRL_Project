#!/usr/bin/env bash
# Compatibility patches for site-packages, run once after pip install.

set -euo pipefail

# numpy 1.24 removed np.bool; patch mlagents_envs/rpc_utils.py.
python - <<'PY'
import pathlib, re
import mlagents_envs
rpc_utils = pathlib.Path(mlagents_envs.__file__).parent / "rpc_utils.py"
src = rpc_utils.read_text()
patched = re.sub(r"\bnp\.bool\b", "bool", src)
if patched != src:
    rpc_utils.write_text(patched)
    print(f"patched np.bool in {rpc_utils}")
PY

# macOS Unity binary path fix.
if [[ "$(uname -s)" == "Darwin" ]]; then
    python - <<'PY'
import pathlib
import soccer_twos
pkg = pathlib.Path(soccer_twos.__file__).parent / "package.py"
src = pkg.read_text()
replacements = [
    ("mac_os/soccer-twos.app/Contents/MacOS/UnityEnvironment", "mac_os/soccer-twos"),
]
patched = src
for old, new in replacements:
    patched = patched.replace(old, new)
if patched != src:
    pkg.write_text(patched)
    print(f"patched macOS Unity path in {pkg}")
PY
fi
