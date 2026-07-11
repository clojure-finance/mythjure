#!/usr/bin/env bash
#
# run_all.sh — reproduce the whole mythjure research pipeline in one go.
#
# Runs the test suite (grad checks) and then every experiment script in paper
# order (§3 → §7.6; see the README "Experiments" table). Each Clojure script writes its own .edn result; this
# wrapper captures each run's stdout+stderr to scripts/<name>.log and prints a
# timed PASS/FAIL summary at the end.
#
# Usage:
#   scripts/run_all.sh              # tests + all experiments
#   scripts/run_all.sh --no-test    # skip the test suite
#   scripts/run_all.sh --quick      # skip the slow LM/curriculum runs
#
# Everything is pure clojure.core on the base classpath (no native deps), so a
# JVM + the Clojure CLI is all that's required. Expect this to take a while:
# the experiments are CPU-bound BPTT at toy scale.

set -euo pipefail

# Resolve project root as the parent of this script's dir, so it runs from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

RUN_TESTS=1
QUICK=0
for arg in "$@"; do
  case "$arg" in
    --no-test) RUN_TESTS=0 ;;
    --quick)   QUICK=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg (see --help)" >&2; exit 2 ;;
  esac
done

command -v clojure >/dev/null || { echo "error: clojure CLI not found on PATH" >&2; exit 1; }

# The LM experiments read the char-level corpus; fail early with a clear message
# rather than deep inside a JVM stack trace if it's missing.
[ -f resources/tinyshakespeare.txt ] || {
  echo "error: resources/tinyshakespeare.txt missing (needed by data.clj)" >&2
  exit 1
}

# Experiment scripts in paper-section order (see README "Experiments" table).
# --quick drops the multi-stage curriculum / LM runs that dominate wall-clock.
SCRIPTS=(
  "foundations"           # §3–§5 discretization/spiral/Lipschitz diagnostics (fast, deterministic)
  "spectral"              # §5.3  ρ_spec vs ‖J_G‖₂ (power iteration on JVPs) + injection ratio + seed robustness (slow: ~80 min, settle cap on non-convergent seeds)
  "seed_verify"           # §5.3  dense cross-check of ρ_spec/‖J_G‖₂ across 10 seeds at ρ=0.3,0.9 (slow: ~35 min)
  "unconstrained_carry"   # §7.1  constrained vs free carry across lr
  "train_run"             # §7.2  Tiny Shakespeare LM; carry self-corrects
  "copy_sweep"            # §7.3  full-attention copy sweep
  "window_feasibility"    # §7.4  can w=1 learn a short gap at all?
  "window_curriculum"     # §7.4  windowed copy curriculum
  "windowed_hard"         # §7.4  k=2..6 unconfounded, finer curriculum
  "ablation"              # §7.5  ablate block's view of H_t (k=2)
  "ablation_k4"           # §7.5  same ablation at a 4-hop gap
  "faithful_check"        # note  finding holds under faithful Parcae arch
  "diagonal_memory"       # §7.6  axis-aligned task: carry IS recruited
)
QUICK_SKIP=("spectral" "seed_verify" "train_run" "window_curriculum" "windowed_hard" "diagonal_memory")

declare -a RESULTS

fmt_secs() { local s=$1; printf '%dm%02ds' $((s/60)) $((s%60)); }

run_step() {
  local name="$1"; shift
  local log="scripts/${name}.log"
  printf '\n\033[1m▶ %s\033[0m  → %s\n' "$name" "$log"
  local start; start=$(date +%s)
  if "$@" >"$log" 2>&1; then
    local dur=$(( $(date +%s) - start ))
    printf '  \033[32m✓ pass\033[0m  (%s)\n' "$(fmt_secs "$dur")"
    RESULTS+=("PASS  $(fmt_secs "$dur")  $name")
  else
    local code=$? dur=$(( $(date +%s) - start ))
    printf '  \033[31m✗ FAIL (exit %d)\033[0m  (%s) — see %s\n' "$code" "$(fmt_secs "$dur")" "$log"
    RESULTS+=("FAIL  $(fmt_secs "$dur")  $name")
    tail -n 15 "$log" | sed 's/^/    | /'
  fi
}

PIPELINE_START=$(date +%s)

if [ "$RUN_TESTS" -eq 1 ]; then
  run_step "test_suite" clojure -M:test
fi

for name in "${SCRIPTS[@]}"; do
  if [ "$QUICK" -eq 1 ] && printf '%s\n' "${QUICK_SKIP[@]}" | grep -qx "$name"; then
    printf '\n\033[2m⏭ %s (skipped: --quick)\033[0m\n' "$name"
    RESULTS+=("SKIP  --      $name")
    continue
  fi
  run_step "$name" clojure -M "scripts/${name}.clj"
done

# ---- summary ----
printf '\n\033[1m═══ pipeline summary ═══\033[0m\n'
for r in "${RESULTS[@]}"; do printf '  %s\n' "$r"; done
printf '  ─────────────────────\n  total: %s\n' "$(fmt_secs $(( $(date +%s) - PIPELINE_START )))"

# Non-zero exit if anything failed, so CI / && chaining behaves.
if printf '%s\n' "${RESULTS[@]}" | grep -q '^FAIL'; then exit 1; fi
