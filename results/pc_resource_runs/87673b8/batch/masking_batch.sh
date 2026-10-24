#!/usr/bin/env bash
# Live batch for configs/resource_masking_experiment.json, in the declared order.
# Usage: masking_batch.sh PREFIX [phase]   phase: all (default), a (required conditions), b (history only)
# A run cut short by an invalid episode is continued from the next seed; nothing is re-run or replaced.
set -u
PREFIX="$1"; PHASE="${2:-all}"
S="C:/Users/ethan/AppData/Local/Temp/claude/C--Users-ethan-PycharmProjects-RL-Minecraft-project/e8caeeaf-7c0d-42d8-a0e0-47974dda527c/scratchpad"
cd "C:/Users/ethan/PycharmProjects/RL-Minecraft-project" || exit 1
P=./.venv/Scripts/python.exe
if [ -n "$(git status --porcelain -- minecraft_bridge infrastructure minecraft_rl configs scripts)" ]; then echo "refusing: uncommitted code"; exit 1; fi
STATUS="$S/$PREFIX-status.txt"
echo "revision $(git rev-parse --short HEAD) phase $PHASE start $(date +%H:%M:%S)" >> "$STATUS"

# folders NAME -> number of episode folders already attempted under runs/NAME*
attempted() { local n=0; for d in runs/$1 runs/$1-part*; do [ -d "$d" ] && n=$((n + $(ls -d "$d"/episode-* 2>/dev/null | wc -l))); done; echo $n; }
latest_policy() { local last="runs/$1"; for d in runs/$1-part*; do [ -d "$d" ] && last="$d"; done; echo "$last/policy.json"; }

# train NAME CONDITION ALGORITHM EPISODES FIRST_SEED
train() {
  local name=$1 condition=$2 algorithm=$3 total=$4 first=$5 part=0
  for d in runs/$name runs/$name-part*; do [ -e "$d" ] && { echo "refusing: $d exists"; exit 1; }; done
  while :; do
    local done_n; done_n=$(attempted "$name")
    [ "$done_n" -ge "$total" ] && break
    [ "$part" -ge 5 ] && { echo "$name GAVE UP after $part continuations at $done_n/$total" >> "$STATUS"; break; }
    local remaining=$((total - done_n)) folder="runs/$name"
    [ "$part" -gt 0 ] && folder="runs/$name-part$part"
    # The declared schedule is linear from 0.6 to 0.1 over all episodes; a continuation resumes it exactly.
    local eps; eps=$("$P" -c "t=$total; k=$done_n+1; print(round(0.6 + (0.1 - 0.6) * (k - 1) / max(1, t - 1), 6))")
    local args=(resource-train --condition "$condition" --algorithm "$algorithm" --episodes "$remaining"
                --world-seed $((first + done_n)) --seed 0 --max-steps 60 --pause 0 --epsilon "$eps" --epsilon-end 0.1 --out "$folder")
    if [ "$part" -gt 0 ]; then args=(resource-train --policy "$(latest_policy "$name")" --algorithm "$algorithm" --episodes "$remaining"
                --world-seed $((first + done_n)) --seed 0 --max-steps 60 --pause 0 --epsilon "$eps" --epsilon-end 0.1 --out "$folder"); fi
    "$P" -m infrastructure "${args[@]}" > "$S/$PREFIX-$(basename "$folder").log" 2>&1
    echo "$(basename "$folder") EXIT=$? attempted=$(attempted "$name")/$total $(date +%H:%M:%S)" >> "$STATUS"
    part=$((part + 1))
  done
}

# evaluate NAME EPISODES FIRST_SEED (--policy PATH | --random-baseline --condition C)
evaluate() {
  local name=$1 total=$2 first=$3; shift 3
  for d in runs/$name runs/$name-part*; do [ -e "$d" ] && { echo "refusing: $d exists"; exit 1; }; done
  local part=0
  while :; do
    local done_n; done_n=$(attempted "$name")
    [ "$done_n" -ge "$total" ] && break
    [ "$part" -ge 5 ] && { echo "$name GAVE UP after $part continuations at $done_n/$total" >> "$STATUS"; break; }
    local folder="runs/$name"; [ "$part" -gt 0 ] && folder="runs/$name-part$part"
    "$P" -m infrastructure resource-demo "$@" --episodes $((total - done_n)) --world-seed $((first + done_n)) \
      --seed 0 --max-steps 60 --pause 0 --out "$folder" > "$S/$PREFIX-$(basename "$folder").log" 2>&1
    echo "$(basename "$folder") EXIT=$? attempted=$(attempted "$name")/$total $(date +%H:%M:%S)" >> "$STATUS"
    part=$((part + 1))
  done
}

if [ "$PHASE" = all ] || [ "$PHASE" = a ]; then
  train "$PREFIX-masked-q-train" masked q_learning 20 100
  train "$PREFIX-masked-sarsa-train" masked sarsa 20 100
  train "$PREFIX-unmasked-q-train" unmasked q_learning 50 100
  train "$PREFIX-unmasked-sarsa-train" unmasked sarsa 50 100
  evaluate "$PREFIX-masked-q-test" 20 20000 --policy "$(latest_policy "$PREFIX-masked-q-train")"
  evaluate "$PREFIX-masked-sarsa-test" 20 20000 --policy "$(latest_policy "$PREFIX-masked-sarsa-train")"
  evaluate "$PREFIX-masked-random-test" 20 20000 --random-baseline --condition masked
  evaluate "$PREFIX-unmasked-q-test" 20 20000 --policy "$(latest_policy "$PREFIX-unmasked-q-train")"
  evaluate "$PREFIX-unmasked-sarsa-test" 20 20000 --policy "$(latest_policy "$PREFIX-unmasked-sarsa-train")"
  evaluate "$PREFIX-unmasked-random-test" 20 20000 --random-baseline --condition unmasked
  echo "PHASE A DONE $(date +%H:%M:%S)" >> "$STATUS"
fi
if [ "$PHASE" = all ] || [ "$PHASE" = b ]; then
  train "$PREFIX-history-q-train" unmasked_history q_learning 50 100
  train "$PREFIX-history-sarsa-train" unmasked_history sarsa 50 100
  evaluate "$PREFIX-history-q-test" 20 20000 --policy "$(latest_policy "$PREFIX-history-q-train")"
  evaluate "$PREFIX-history-sarsa-test" 20 20000 --policy "$(latest_policy "$PREFIX-history-sarsa-train")"
  echo "PHASE B DONE $(date +%H:%M:%S)" >> "$STATUS"
fi
echo "BATCH DONE $(date +%H:%M:%S)" >> "$STATUS"
