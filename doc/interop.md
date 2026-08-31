# PyTorch interop: edge cases and conventions

Everything the `mythjure.torch.*` façade learned the hard way about running
PyTorch through [libpython-clj](https://github.com/clj-python/libpython-clj).
Each item below cost real debugging; most are baked into guards, coercion
helpers, or tests so that user code cannot re-discover them the hard way —
this page is the consolidated map. Pointers like `core/py-tuple` refer to
`mythjure.torch.core` etc.; the last section says which test pins what.

Validated envelope: float64 on CPU (the oracle-comparison configuration).
The bridge itself is device- and dtype-agnostic, but the tight tolerances
quoted in the README are f64/CPU facts.

## Setup and the pyenv trap

Covered in the README (*Torch backend setup*): discovery finds `python3` on
`PATH`, interrogates it in a subprocess, and derives the matching
`libpython` from the interpreter's own `sysconfig`. The env vars
`MYTHJURE_PYTHON` / `MYTHJURE_LIBPYTHON` are overrides, not requirements.

The trap this design removes: with only an executable configured,
libpython-clj could silently load the *system* `libpythonX.Y.so` — wrong
interpreter, wrong `site-packages`. Symptom: `ModuleNotFoundError` for a
package that `python3 -c "import torch"` clearly has. An explicit libpython
from a *different* installation than the executable's own is detected and
refused before anything embeds. When anything misbehaves, run
`(mythjure.torch.doctor/doctor)` first — it is safe even when
initialization itself is broken.

## REPL lifecycle

- **Never `:reload-all`** a namespace while torch is loaded. It re-evaluates
  libpython-clj's protocol definitions and orphans every live Python handle
  (symptom: `No implementation of :get-attr`). Restart the REPL instead.
  Plain `:reload` of mythjure namespaces is safe.
- **An embedded CPython cannot be un-embedded.** Changing the Python
  configuration requires a JVM restart; `doctor` cross-checks the embedded
  interpreter against the currently-resolved configuration and warns on
  drift. A *failed* `initialize!` resets the init flag, so a corrected
  configuration can retry in the same JVM.

## Scalars across the bridge

libpython-clj auto-converts Python numbers and strings to JVM values on
return. Feeding such a JVM scalar back into the raw FFI is interpreted as a
**pointer — and segfaults the JVM**. The façade guards this class out:
`core/call` / `core/call-kw` / `core/attr` throw a readable error on plain
JVM values instead, and `tensor/item` is idempotent on numbers. If you work
below the façade (raw libpython-clj), this is the first thing that will
kill your REPL.

## Collections and coercion

Whether torch's C++ layer accepts an auto-bridged JVM collection is
**per-op luck**: `cat`/`stack` happen to take bridged vectors,
`layer_norm`/`load_state_dict` reject them. So coercion is *explicit*,
always:

- shapes and int-tuples → `core/py-tuple` (constructors, `reshape`,
  `layer_norm` all type-check tuple-of-ints in C++),
- lists of tensors → `core/py-list`,
- dicts → `core/py-dict` (auto-bridged maps are rejected by
  `load_state_dict`).

The generic tier (`op/torch-fn` / `op/nn-fn` / `op/method`) deliberately
never guesses a coercion — a wrong guess is silent breakage. Curated
`defop` rows declare theirs per argument.

Return direction:

- **Tuple returns** (e.g. `split`, `topk`): use `core/py-vec` — elements
  stay *live* tensors. Bridged tuples are `java.util.List` and look
  `coll?`-ish, which is why the helper branches before the pyobj guard.
- **Dict `keys()` views do not auto-convert** — they come back as opaque
  pointers. `core/py-keys` materializes via Python's `list()` first.
- **`torch.__version__` is a `str` subclass** — the generic `->jvm` walk
  turns it into a seq of characters. Call `(str …)` on it.

## Numeric semantics pins

Places where torch's defaults silently diverge from the pure-Clojure
oracle, pinned in the façade:

- **`.float()` is a float32 DOWNCAST**, not a no-op cast — `inspect` stats
  go through `.double()`.
- **`nn/gelu` pins `approximate="tanh"`** (torch defaults to erf; the
  oracle uses tanh — unpinned, they diverge silently at ~1e-3).
- `nn/layer-norm` eps defaults to 1e-5, `nn/softmax` normalizes over the
  last dim — both matching `mythjure.linalg`.
- Tensor constructors default to float32; pass `:dtype :float64` when
  comparing against the oracle.
- `nn/sdpa` exists for the general-purpose path only — a fused kernel
  cannot be compared term-by-term against the oracle, so it never appears
  in the experiment path.

## Mutation, views, and aliasing

The façade keeps torch's aliasing semantics faithfully — no defensive
copying. The load-bearing rule: **aliasing is only observable through
mutation, so “no `!`, no aliasing bugs.”** The README's *Mutation & views*
section is the user-facing statement; the interop details:

- No unmarked op mutates its arguments, façade-wide (`module/to-dtype!` is
  bang-marked because **`nn.Module.to` mutates the module** — unlike
  `Tensor.to`, which returns self-or-copy).
- `!`-ops are guarded at the mutation site: mutating an autograd
  *intermediate* (a tensor with a `grad_fn`) throws readably **before any
  write** — torch's own version-counter error would otherwise fire later at
  `backward!`, far from the bug. `:unsafe true` bypasses. Leaves keep
  torch's rules: refused under grad mode, legal under `no-grad` (the
  optimizer pattern).
- `tensor/aliasing-table` is the greppable record of every op that can
  share storage. The input-dependent ones bite hardest: `reshape`/`flatten`
  are views **iff the input is contiguous** (`transpose` → `reshape` is a
  silent copy), `contiguous` returns its input when already contiguous,
  `einsum` on pure-permutation equations (`"ij->ji"`, even `"ij->ij"`)
  lowers to a view while contractions allocate.
- **Detaching is not copying.** `autograd/detach` *and* `module/state-dict`
  (even by default) cut the autograd graph, not the storage — their results
  alias the live weights and, having no `grad_fn`, are invisible to the
  `!`-guard. `tensor/clone` is the firebreak (fresh storage, still
  differentiable); `tensor/data-ptr` diagnoses sharing (equal ⇒ shared;
  unequal proves nothing — offset views).

## Autograd semantics

Torch's rules, surfaced through the bridge (they are torch's, not ours):

- `backward!` **accumulates** into `.grad` — clear between steps
  (`clear-grads!` or `optim/zero-grad!`).
- The graph is **freed** by `backward!` unless `:retain-graph` — a second
  call on the same graph is a Python-side error.
- **Tied parameters** (one tensor reachable via several param-map paths)
  accumulate their gradient contributions automatically under autograd. On
  the manual path, `optim/set-grads!` and `all-param-tensors`
  identity-deduplicate — without that, Adam would step a tied tensor twice
  (silent 2× learning rate). Tied params must be the *same object* at init.
- `.grad` is `None` until first backward — it crosses the bridge as `nil`;
  clearing is `__setattr__ "grad" nil`.

## Error rendering

Python-side failures come back as `ex-info` whose first line is the actual
Python error (the last traceback line when a traceback comes through) plus
the shapes of all tensor arguments, e.g.
`call .matmul failed: RuntimeError: mat1 and mat2 shapes cannot be
multiplied (2x3 and 2x3) | tensor shapes [[0 [2 3]] [1 [2 3]]]` — original
exception kept as the cause. The scalar/FFI pointer guard fires before any
of this can be reached.

## Op coverage: three tiers

1. **Curated** — named, tested wrappers (hand-written or one-line `defop`
   rows with coercion/return/aliasing specs).
2. **Generic** — `op/torch-fn` / `nn-fn` / `method` reach any torch op
   today, explicit coercion, no wrapper needed.
3. **Discovery** — libpython-clj's `require-python` generates a namespace
   from the torch module for REPL exploration (Python docstrings included).
   Use it to *find* ops; *call* them through tiers 1–2 (generated fns have
   none of the façade's hardening).

Promotion policy: when a tier-2 call site stabilizes, add a `defop` row —
the table-driven tests insist every registered row has a passing example.
Details in the `mythjure.torch.op` namespace docstring.

## Where each claim is pinned

| Concern | Test file (`test-torch/mythjure/`) |
| --- | --- |
| Oracle agreement of core ops, scalar guard, coercion helpers | `torch_facade_test.clj` |
| Every `defop` row has a passing example (two-way) | `torch_op_test.clj` |
| Every aliasing claim has a passing witness (two-way) | `torch_mutation_test.clj` |
| Autograd vs. manual VJPs, accumulate/free/tied semantics | `torch_autograd_test.clj` |
| state_dict round-trip, live-weights training | `torch_module_test.clj` |
| Discovery, precedence, pyenv-trap detection, readable errors | `torch_doctor_test.clj` |
| Forward/backward/training trajectory vs. oracle | `model_torch_test.clj`, `backprop_torch_test.clj`, `train_torch_test.clj` |
