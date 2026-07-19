#!/bin/bash
# Deprecated: COMBO paper replication uses full D4RL medium-expert, not sparse data.
echo "NOTE: COMBO_WALKER2D_SPARSE.sh redirects to COMBO_WALKER2D.sh (full D4RL)."
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/COMBO_WALKER2D.sh" "$@"
