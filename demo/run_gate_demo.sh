#!/bin/bash
# M8 demo: the same CI gate passes the committed policy and rejects the unfair one.
cd "$(dirname "$0")/.."
echo "=== Committed policy (models/fair_policy.json) ==="; python3 src/gate.py; echo "exit code: $?"
echo; echo "=== Deliberately unfair commit (demo/unfair_commit_policy.json) ==="
python3 src/gate.py --policy demo/unfair_commit_policy.json; echo "exit code: $?"
