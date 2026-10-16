#!/usr/bin/env bash
# Refresh every input the models use (nflverse + sportsdataverse). ~250 MB, about a minute.
set -euo pipefail
NFL="${NFL_DATA_DIR:-/tmp/nfl}"; CFB="${CFB_DATA_DIR:-/tmp/cfb}"
mkdir -p "$NFL" "$CFB"
REL="https://github.com/nflverse/nflverse-data/releases/download"
curl -sSfL -o "$NFL/games.csv" "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
for y in 2023 2024 2025 2026; do curl -sSfL -o "$NFL/play_by_play_$y.parquet" "$REL/pbp/play_by_play_$y.parquet"; done
for y in 2024 2025 2026; do
  curl -sSfL -o "$NFL/injuries_$y.csv" "$REL/injuries/injuries_$y.csv"
  curl -sSfL -o "$NFL/depth_charts_$y.csv" "$REL/depth_charts/depth_charts_$y.csv"
done
curl -sSfL -o "$NFL/roster_weekly_2026.csv" "$REL/weekly_rosters/roster_weekly_2026.csv"
curl -sSfL -o "$CFB/schedules_2026.parquet" "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/parquet/cfb_schedules_2026.parquet"
curl -sSfL -o "$CFB/espn_pbp_2026.parquet" "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_pbp/play_by_play_2026.parquet"
ls -la "$NFL" "$CFB"
