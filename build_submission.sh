#!/usr/bin/env bash
set -euo pipefail
TEAM_NAME="${1:-team_innovators}"
python utils/package_submission.py --team-name "$TEAM_NAME"
