#!/usr/bin/env bash
set -euo pipefail

PHASE="$1"; shift || true
TITLE="$1"; shift || true
DURATION_START="${1:-$(date +%F)}"; shift || true
DURATION_END="${1:-$(date +%F)}"; shift || true

if [[ -z "${PHASE:-}" || -z "${TITLE:-}" ]]; then
  echo "Usage: $0 <phase> <title> [start-date YYYY-MM-DD] [end-date YYYY-MM-DD]" >&2
  exit 2
fi

cat >> history.md <<EOF

## ${PHASE}: ${TITLE}

- **Duration:** ${DURATION_START} ~ ${DURATION_END}
- **Completed Work:**
  - [Fill in]
- **Issues Encountered:**
  - [Fill in]
- **Retrospective:**
  - **What went well:** [Fill in]
  - **What to improve:** [Fill in]
EOF

echo "Appended skeleton entry to history.md for ${PHASE}: ${TITLE}" >&2
