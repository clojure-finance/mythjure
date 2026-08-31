(ns mythjure.torch.tmd
  "scicloj bridge: tech.ml.dataset ⇄ torch tensors (direction doc §5.1
  item 4 — the ecosystem integration; boundary work under the §4 doctrine).

  Requires the :tmd alias in addition to :torch
  (clojure -M:nrepl:torch:tmd) — tech.ml.dataset is NOT a dependency of the
  torch facade itself, only of this namespace.

  Value semantics, BOTH directions (pinned by torch_tmd_test): the bridge
  COPIES at the JVM↔Python edge — mutating a tensor produced by ds->tensor
  never touches the source dataset, and a dataset produced by tensor->ds
  never sees later tensor mutation. The facade's aliasing rule (\"no !, no
  aliasing bugs\") therefore extends across the ecosystem boundary with no
  caveats. The copy is buffer-level (dtype-next ⇄ numpy), not element-wise
  Clojure — this is the fast lane the raw-bytes gap (doc/interop.md,
  bulk ingest) points to.

  Column names do not survive into a tensor; tensor->ds takes
  :column-names to restore them. Numeric columns only — select/encode
  with tech.v3.dataset first (categorical → int columns, etc.).

  Facade-internal namespace: allowed to touch libpython-clj directly (the
  dtype-next ⇄ numpy zero-conversion lane lives there)."
  (:require [libpython-clj2.python :as py]
            [libpython-clj2.python.np-array]   ; enables numpy ⇄ dtype-next
            [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.tensor :as ds-tens]))

(defn ds->tensor
  "A dataset's numeric columns as one [rows cols] tensor (a COPY — see the
  ns docstring). Options:
    :columns  subset + order of columns (default: all, dataset order)
    :dtype    tensor dtype keyword (default :float32; use :float64 against
              the oracle, :int64 for label columns)."
  [dataset & {:keys [columns dtype] :or {dtype :float32}}]
  (let [d (if columns (ds/select-columns dataset columns) dataset)]
    (t/from-numpy (py/->python (ds-tens/dataset->tensor d dtype)))))

(defn tensor->ds
  "A [rows cols] (or [rows], reshaped to one column) tensor as a
  tech.ml.dataset dataset (a COPY — see the ns docstring). Works on
  autograd tensors (detached internally; detaching before copying is
  semantics-free). Option :column-names replaces the default 0..C-1 names."
  [tensor & {:keys [column-names]}]
  (let [t2  (if (= 1 (count (t/shape tensor)))
              (t/reshape tensor [(first (t/shape tensor)) 1])
              tensor)
        dtt (py/->jvm (core/call (core/call t2 "detach") "numpy"))
        d   (ds-tens/tensor->dataset dtt)]
    (if column-names
      (ds/rename-columns d (zipmap (ds/column-names d) column-names))
      d)))
