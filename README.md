# mythjure

[![arXiv](https://img.shields.io/badge/arXiv-2607.10681-b31b1b.svg)](https://arxiv.org/abs/2607.10681)

Building a mini **looped (recurrent-depth) transformer** from scratch in Clojure,
REPL-first, on a CPU — to understand the Parcae / S4 stability mathematics
hands-on rather than at scale.

> The interesting question isn't "can we match a frontier lab's compute" (we
> can't). It's: *what does the stability-constrained loop actually do?* Watch an
> unconstrained loop explode; apply the negative-diagonal discretization; watch
> it stay bounded. That insight costs minutes on a laptop, not a GPU cluster.

## Approach

Pure Clojure, manual backprop (BPTT through one shared weight block — a small,
regular set of gradients). No native dependencies: the linear algebra is dense
`clojure.core`, fast enough at toy scale and completely explicit. The goal is to
build the stability mathematics from scratch — discretization, the gated
recurrence, hand-derived gradients, and a full training loop — to understand the
mechanisms directly rather than to reach for scale.

## Layout (`src/`)

| Path | What |
|---|---|
| `dynamics.clj` | Step 0: discretization + stability dynamics, pure `clojure.core`. Runnable now. |
| `nonlinear.clj` | Step 1: nonlinear `R(h,e)=tanh(Wh+e)` in the loop; reproduces the Parcae spiral geometry. |
| `linalg.clj` | Dense linear algebra: matmul, softmax, LayerNorm, GELU, causal + **windowed** masks. |
| `block.clj` | Step 2: real decoder block — multi-head attention (causal or windowed via `:window`) + GELU MLP, pre-LN + residuals. Forward. |
| `looped.clj` | Step 3: drives the block recurrently; shows looped residual growth and the (simplified, B̄=0) Parcae carry bounding it. |
| `lipschitz.clj` | Empirical local Lipschitz: random probes + `channel-lipschitz`/`lipschitz-robust` (axis-aware, for diagonal-carry maps). |
| `analysis.clj` | Stability diagnostics / training monitors: `scale-profile`, `gated-contraction`, `per-channel-contraction`; plus `spectral-radius`/`operator-norm`/`spectral-analysis` (power iteration on JVPs — ρ_spec vs ‖J_G‖₂ non-normality), `cf-analysis` (true self-stabilization constant C/F), `injection-ratio` (‖H*+E‖/‖H*‖), and `spectral-seed-sweep` (ρ_spec robustness across weight seeds). |
| `backprop.clj` | Step 4: manual reverse-mode VJPs for the block + BPTT through the loop (incl. `B̄e` injection and the recurrence ablation). |
| `data.clj` | Char-level Tiny Shakespeare (`resources/tinyshakespeare.txt`): load, vocab, encode, batch sampler. |
| `model.clj` | Step 5: full LM — Prelude → looped block → Coda → cross-entropy. Three carry modes: `:parcae` (ā=exp(−exp log_A)), `:parcae :faithful?` (learnable Δ + B̄ injection + prelude LN, Parcae Eq. 3), `:free` (unconstrained ā). |
| `train.clj` | Adam, minibatch loop, and the three carry diagnostics (ā distribution, dL/dā sign, per-channel convergence ratio). |
| `copytask.clj` | Synthetic copy-across-a-gap task `[A][PAD×k][QUERY]` with a single dial `k`. |

**Tests** (`test/`): `backprop_test.clj` (primitive VJPs, attention, block, BPTT) + `model_test.clj` (full-model finite-diff over every leaf × standard/windowed/ablated/faithful). Run `clojure -M:test` → **7 tests, 100 assertions**, analytic vs finite-diff to <1e-6.

## Experiments (`scripts/`)

These scripts reproduce the experiments in the companion paper
[*"LayerNorm as Implicit Gain Control in Looped Transformers"*](https://arxiv.org/abs/2607.10681)
(arXiv:2607.10681); the table below maps each script to its result and paper section. Each writes a `.log` and
(most) a `.edn`. The outputs recorded for the paper are pinned at tag
[`paper-v1`](https://github.com/clojure-finance/mythjure/releases/tag/paper-v1)
(untracked on `main` so ongoing development doesn't churn result files).
The `*_torch.clj` variants rerun the slowest experiments on the torch backend
(~11–86× faster) and verify they reproduce the recorded results exactly.

Reproduce everything in one go with `scripts/run_all.sh` — it runs the test suite
(the grad checks) and then every experiment below in paper order, capturing each
run's output to `scripts/<name>.log` and printing a timed PASS/FAIL summary. Flags:
`--no-test` skips the suite, `--quick` skips the slow LM/curriculum runs. To run a
single experiment: `clojure -M scripts/<name>.clj`.

| Script | Paper § | Finding |
|---|---|---|
| `spectral.clj` | §5.3 (Tab. spectral/cf/seed-robustness), App. A | Power iteration on Jacobian–vector products at residual-verified fixed points. **J_G is non-normal**: ‖J_G‖₂>1 in every instance that converges (9/9 seeds at ρ=0.3, 8/8 at ρ=0.9) while ρ_spec<1 at every verified fixed point — confirms the spiral step-size rebound. The finite-diff probe under-estimates operator norms ~2.6×, so the true self-stabilization constant is **C/F≈2.5, not ≈1** (`cf-analysis`). Seed sweep: the spectral margin collapses monotonically as ρ→1 (0.159→0.012), and **a minority of seeds (1–2/10 at every carry tested) never converge to a fixed point at all** (bounded, non-settling orbits) — the diagonal constraint ρ(Ā)<1 is necessary but not sufficient for convergence of the full recurrence. Plus injection ratio ‖H*+E‖/‖H*‖ across ρ (App. A). |
| `seed_verify.clj` | §5.3 (Tab. seed-robustness) | Dense cross-check of ρ_spec (validates the matrix-free estimate to ≤0.004) and ‖J_G‖₂ per seed at ρ=0.3, 0.9 across 10 seeds; settles each seed to a residual-verified fixed point and reports non-convergent seeds separately. |
| `foundations.clj` | §3–§5 (Tab. 1, 5, 13, Fig. 1–2) | The analytical/illustrative numbers: unconstrained loop explodes vs Parcae-constrained stays bounded, the Banach contraction threshold (§4.1), the spiral geometry (§5.1, Fig. 1), `Lip(block) ∝ 1/‖H‖` (§4, Tab. 1), the shrinking contraction margin as ρ→1 (§5.3 / App. C probe record, Tab. 13, Fig. 2), and per-channel carry governed by the slowest channel (§5.4, Tab. 5). Deterministic, seconds to run. |
| `unconstrained_carry.clj` | §7.1 (Tab. 6, Fig. 3) | `:parcae` vs `:free` across lr: constrained ρ(Ā)<1 always; **free learns ρ>1 at high lr** with residual explosion — reproduces Parcae Fig. 3 / Table 2. |
| `train_run.clj` | §7.2 (Tab. 7) | Trains the LM on Tiny Shakespeare; ā **self-corrects** (stays ~0.2, no cliff) at low lr. |
| `copy_sweep.clj` | §7.3 (Tab. 8) | Full-attention copy sweep: ā ignores gap `k` — attention (position-memory) solves it, carry (depth-memory) idle. |
| `window_feasibility.clj` / `window_curriculum.clj` | §7.4 (Tab. 9, Fig. 4) | Windowed (w=1) copy: carry stays flat; depth-memory flows through the block's **nonlinear recurrence**, not the linear carry. |
| `windowed_hard.clj` | §7.4 (Tab. 9, Fig. 4) | Removes the k≥6 learnability confound (finer curriculum, d=32): solves **k=2→6**, ā flat/declining (0.52→0.49) & `dL/dā` anti-cliff even at k=6 (predicted 0.83). |
| `ablation.clj` / `ablation_k4.clj` | §7.5 (Tab. 10, Fig. 5) | Ablate the block's view of `H_t` ⇒ windowed copy fails (full attn still solves) — confirms the nonlinear recurrence is the mechanism. |
| `faithful_check.clj` | §7.4 note (Parcae relationship, §1) | Central finding (§7.4) holds under the **faithful** Parcae architecture (learnable Δ + B̄e + prelude LN): ā flat, `dL/dā` anti-cliff at k=2,4. |
| `diagonal_memory.clj` | §7.6 (Tab. 11, Fig. 6) | **Boundary of the "free constraint":** on an axis-aligned per-channel task, the carry *is* recruited — learned ā tracks per-channel targets (corr 0.92). Block subsumes carry only for cross-channel/position tasks. |

## Torch backend (post-paper development)

Everything above — and every number in the paper — runs on the dependency-free
pure-Clojure implementation. Since `paper-v1`, the repo also carries an
optional **PyTorch backend** behind the same function signatures, for speed
(~86× on a full training step at paper scale, CPU-only): `mythjure.torch.*` is
a thin Clojure façade over [libpython-clj](https://github.com/clj-python/libpython-clj),
and `block-torch` / `backprop-torch` / `model-torch` / `train-torch` mirror
their pure namespaces one-to-one using the same hand-derived VJPs translated
to tensor ops — no autograd anywhere in the experiment path. The pure
implementation remains the correctness oracle: the torch backend's forward,
gradients, and full Adam training trajectories are pinned to it by tests
(`clojure -M:test-torch`, float64, agreement ≤1e-12; needs a Python with
`torch` installed — see *Torch backend setup* below). The plain
`clojure -M:test` suite and base classpath stay dependency-free.

Beyond the experiment mirrors, the façade carries three general-purpose
surfaces: `mythjure.torch.autograd` (opt-in reverse-mode autograd —
`backward!`, `no-grad`, param-map grad helpers — validated leaf-for-leaf to
machine ε against the manual VJPs across every model mode),
`mythjure.torch.module` (`nn.Module`/`state_dict` interop: a Python module's
weights become a nested Clojure param map, loadable/savable and trainable
end-to-end through the façade's optimizer and autograd helpers), and
`mythjure.torch.op` (op coverage in three tiers: curated named wrappers
declared as one-line `defop` table rows with explicit coercion, return, and
aliasing/mutation specs — every row is enforced-tested; a generic
`torch-fn`/`nn-fn`/`method` dispatch surface that reaches any torch op
today, no wrapper needed; and `require-python` for docstring-level discovery
at the REPL — see the namespace docstring for the promotion policy between
tiers).

The first post-paper experiment built on this speed is
`scripts/spectral_dsweep_torch.clj` (`clojure -M:torch scripts/spectral_dsweep_torch.clj`),
a budget-guarded width ladder for the paper's §5.3 fixed-point/spectral
claims: d ∈ {32, 64, 128, 256} at head-dim 8 (heads = d/8, d-ff = 4d),
10 seeds × ρ ∈ {0.3, 0.9}, the paper's schedules, with the d=32 rung diffed
against the recorded `seed_verify.edn` as a reproduce-the-paper control.
Preliminary finding (n=10 seeds/cell): the qualitative story is width-robust
(non-normality gap, non-convergent seed minority, C/F drifting ≈2.3→2.9),
the ρ=0.3 margin plateaus near 0.10, and the ρ=0.9 spectral margin shrinks
roughly 2× per doubling of d — zero within estimator noise (~0.005) at
d=256, where residual-verified fixed points sit at marginal stability.
Outputs stay untracked like all script outputs.

### Torch backend setup

You need a Python 3 built with `--enable-shared` that can `import torch`
(CPU wheels are fine: `pip install torch`). That's it — `initialize!` finds
`python3` on `PATH`, asks it (in a subprocess) where its shared `libpython`
lives via `sysconfig`, and embeds the pair. pyenv shims and venv wrappers
resolve to the right interpreter automatically. To pick a *specific* Python,
set the env vars (they override discovery):

```bash
export MYTHJURE_PYTHON=~/.pyenv/versions/3.12.11/bin/python3
export MYTHJURE_LIBPYTHON=~/.pyenv/versions/3.12.11/lib/libpython3.12.so   # optional; derived from MYTHJURE_PYTHON if unset
clojure -M:nrepl:torch     # REPL with the backend
clojure -M:test-torch      # pure + torch suites
```

When anything misbehaves, run the doctor first — it checks the whole chain
(executable → shared build → libpython pairing → torch importability →
embedded state) in a subprocess, so it's safe even when initialization
itself is broken:

```clojure
(require '[mythjure.torch.doctor :as doctor])
(doctor/doctor)
;; mythjure.torch doctor
;; ─────────────────────
;;  ✓ python executable  /home/you/.pyenv/versions/3.12.11/bin/python3 (from env)
;;  ✓ interpreter        Python 3.12.11
;;  ✓ shared build       --enable-shared ✓
;;  ✓ libpython          …/lib/libpython3.12.so (from env, matches interpreter)
;;  ✓ torch              2.9.1
;;  ✓ embedded           initialized; embedded …/bin/python3 (torch 2.9.1)
```

Misconfigurations fail *before* CPython is embedded, with the fix in the
error message — including the classic pyenv trap (an executable from one
Python installation paired with a `libpython` from another, which would
otherwise embed the wrong interpreter with the wrong `site-packages`).

**Deploying an app on the backend:** an uberjar does not bundle Python — the
runtime host needs the same two things (shared-build Python 3 + torch), and
discovery/env vars work identically there. For a pinned, reproducible unit,
use a container: start from `python:3.12-slim` (shared-build Python), add a
JDK 21 and `pip install torch --index-url https://download.pytorch.org/whl/cpu`,
and run the uberjar — no env vars needed, discovery finds the image's Python.
For GPU work later, swap in a CUDA-enabled torch wheel/base image; the bridge
is device-agnostic.

### Mutation & views

The façade keeps torch's aliasing semantics faithfully — some ops return
**views** that share storage with their input (`transpose`, `permute`,
`narrow`, `split`/`chunk`/`unbind`, `broadcast-to`, `squeeze`/`unsqueeze`),
and some alias *depending on their input*: `reshape`/`flatten` are views only
when the input is contiguous (silent copies otherwise), `einsum` on
pure-permutation equations lowers to a view. Since aliasing is only
observable through mutation, the convention is:

- **No unmarked op mutates its arguments.** Mutation is opt-in via the
  `!`-suffixed ops (`add!`, `mul!`, `clamp!`, `zero!`, `fill!`, …). If you
  never call a `!`-op, you have value semantics.
- **`!`-ops are guarded.** Mutating an autograd *intermediate* (a tensor with
  a `grad_fn`) throws a readable error at the mutation site — instead of
  torch's version-counter error later at `backward!`, far from the bug. Pass
  `:unsafe true` to bypass when you know no backward will run. Leaves keep
  torch's own rules: refused under grad mode, legal under `no-grad` (the
  optimizer pattern).
- **`clone` breaks aliasing, `contiguous` normalizes layout, `data-ptr`
  diagnoses sharing.** `mythjure.torch.tensor/aliasing-table` is the
  greppable record of every op that can alias; each claim is empirically
  pinned by `torch_mutation_test`, so a torch upgrade that changes an op's
  aliasing fails a test instead of corrupting weights.
- **Detaching is not copying.** `autograd/detach` and `module/state-dict`
  cut the autograd *graph*, not the *storage* — their results still share
  memory with the live tensors, and (having no `grad_fn`) are invisible to
  the `!`-guard. `clone` the values if you need a real snapshot.

## REPL

Start an nREPL server (the `:nrepl` alias binds an auto-assigned port and writes
it to `.nrepl-port`, which your editor picks up):

```bash
clojure -M:nrepl
```

Then, in the REPL (or via your editor):

```clojure
(require '[mythjure.dynamics :as dyn] :reload)
(dyn/demo-explosion)   ; unconstrained loop diverges
(dyn/demo-stable)      ; Parcae-constrained loop stays bounded
```

## deps.edn aliases

- `:nrepl` — nREPL server on an auto-assigned port (see `.nrepl-port`), for editor connection.
- `:test` — cognitect test-runner (pure suite only).
- `:torch` — adds libpython-clj for the optional PyTorch backend (`clojure -M:nrepl:torch`).
- `:test-torch` — runs the pure and torch suites together.

The base classpath is dependency-free on purpose: the whole project is pure
`clojure.core`, needs nothing native, and the REPL always starts clean.

## Citation

If you use this code or build on the paper, please cite:

```bibtex
@misc{buehlmaier2026layernorm,
  title         = {LayerNorm as Implicit Gain Control in Looped Transformers},
  author        = {Matthias M. M. Buehlmaier},
  year          = {2026},
  eprint        = {2607.10681},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.10681}
}
```

The exact code state used for the paper's experiments is pinned at tag
[`paper-v1`](https://github.com/clojure-finance/mythjure/releases/tag/paper-v1).
