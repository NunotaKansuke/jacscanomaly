#!/usr/bin/env bash
set -u

RESULT_DIR=/moao39_13/nunota/rges-data/anomaly_finder_result
PORTAL_DIR=/rogue1_8/nunota/html_portal/rges_anomaly_finder
SYNC_SCRIPT=/rogue1_8/nunota/html_portal/tool/request_sync.sh
LOG_FILE=/moao39_13/nunota/rges-data/rges_f146_realtime_sync.log
STATE_FILE=/moao39_13/nunota/rges-data/rges_f146_realtime_sync.state
SCAN_SESSION=rges_f146_serial_current

mkdir -p "$PORTAL_DIR"
touch "$STATE_FILE"

process_pending() {
    while IFS= read -r -d '' event_json; do
        event_key=${event_json#"$RESULT_DIR"/}
        if grep -Fqx "$event_key" "$STATE_FILE"; then
            continue
        fi
        # Build each page locally while the scan is running.  Publishing here
        # used to invoke a full portal rsync once per event, which made a
        # 2k-event run generate thousands of overlapping sync requests.
        if python tools/build_rges_anomaly_html.py \
            --result-dir "$RESULT_DIR" \
            --out-dir "$PORTAL_DIR" \
            --event-json "$event_json" >> "$LOG_FILE" 2>&1; then
            printf '%s\n' "$event_key" >> "$STATE_FILE"
            printf '[realtime-sync] built %s (publish deferred)\n' "$event_key" >> "$LOG_FILE"
        else
            printf '[realtime-sync] failed %s; will retry\n' "$event_key" >> "$LOG_FILE"
        fi
    done < <(
        find "$RESULT_DIR/events" -mindepth 2 -maxdepth 2 -type f -name '*.json' -print0 \
            | sort -z
    )
}

while tmux has-session -t "$SCAN_SESSION" 2>/dev/null; do
    process_pending
    sleep 20
done

process_pending
python tools/build_rges_anomaly_html.py \
    --result-dir "$RESULT_DIR" \
    --out-dir "$PORTAL_DIR" >> "$LOG_FILE" 2>&1 \
    && "$SYNC_SCRIPT" >> "$LOG_FILE" 2>&1
