#!/bin/bash
# Deprecated: COMBO paper replication uses full D4RL medium-expert, not sparse data.
echo "NOTE: COMBO_HOPPER_SPARSE.sh redirects to COMBO_HOPPER.sh (full D4RL)."
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/COMBO_HOPPER.sh" "$@"
