#!/usr/bin/env bash

REMOTE="ulrikh@dnat.simula.no"
REMOTE_DIR="/home/ulrikh/D1/DLSCA/models"
LOCAL_DIR="."

mkdir -p "$LOCAL_DIR"

PREFIX="fixslice-sbox-byte"
SUFFIX="-zhang-1150_1600-s777"

SSH_OPTS="-Y -A -C -2 -p 60441"
SCP_OPTS="-P 60441 -o ForwardX11=yes -o ForwardAgent=yes -o Compression=yes -o Protocol=2"


# list only folders matching COMMON_NAME{number}
#ssh $SSH_OPTS "$REMOTE" "ls -1 $REMOTE_DIR | grep '^${PREFIX}[0-9]\+${SUFFIX}'" | while read -r MODEL; do
ssh $SSH_OPTS "$REMOTE" "ls -1 $REMOTE_DIR | grep fixslice-sbox-byte1-zhang-0_1000-s777" | while read -r MODEL; do
    echo "Processing $MODEL"

    mkdir -p "$LOCAL_DIR/$MODEL"

    # copy metadata.json
    scp $SCP_OPTS "$REMOTE:$REMOTE_DIR/$MODEL/metadata.json" "$LOCAL_DIR/$MODEL/"

    META="$LOCAL_DIR/$MODEL/metadata.json"

    BEST_EPOCH=$(jq '
        .scores[1]
        | to_entries
        | min_by(.value)
        | .key
    ' "$META")

    echo "Best epoch: $BEST_EPOCH"

    scp $SCP_OPTS "$REMOTE:$REMOTE_DIR/$MODEL/epoch$BEST_EPOCH.pt" "$LOCAL_DIR/$MODEL/"
done