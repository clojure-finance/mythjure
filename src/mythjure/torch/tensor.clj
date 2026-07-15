(ns mythjure.torch.tensor
  "Tensor creation and manipulation — the Clojure face of torch tensors.

  Conventions mirror mythjure.linalg where they overlap: a matrix is
  [rows × cols], `transpose` with no dims swaps the last two. Constructors
  default to core/*device* and :float32; pass :dtype :float64 when comparing
  against the pure-Clojure oracle."
  (:refer-clojure :exclude [get cat abs flatten chunk])
  (:require [mythjure.torch.core :as core]
            [mythjure.torch.op :as op :refer [defop]]))

;; ---------------------------------------------------------------------------
;; Construction
;; ---------------------------------------------------------------------------

(defn- make-opts [{:keys [device dtype requires-grad]
                   :or {device core/*device* dtype :float32 requires-grad false}}]
  {:device device :dtype (core/dtype dtype) :requires_grad requires-grad})

(defn randn
  "Standard-normal tensor of the given shape."
  [shape & {:as opts}]
  (core/call-kw core/torch "randn" [(core/py-tuple shape)] (make-opts opts)))

(defn zeros [shape & {:as opts}]
  (core/call-kw core/torch "zeros" [(core/py-tuple shape)] (make-opts opts)))

(defn ones [shape & {:as opts}]
  (core/call-kw core/torch "ones" [(core/py-tuple shape)] (make-opts opts)))

(defn arange [n & {:as opts}]
  (core/call-kw core/torch "arange" [n] (make-opts opts)))

(defn eye [n & {:as opts}]
  (core/call-kw core/torch "eye" [n] (make-opts opts)))

(defn from-clj
  "Build a tensor from (nested) Clojure vectors or a scalar."
  [data & {:as opts}]
  (core/call-kw core/torch "tensor" [data] (make-opts opts)))

(defn zeros-like
  "Zeros with the same shape/dtype/device as t."
  [t]
  (core/call core/torch "zeros_like" t))

(defn ones-like [t] (core/call core/torch "ones_like" t))

(defn rsqrt "1/√t elementwise." [t] (core/call core/torch "rsqrt" t))

;; ---------------------------------------------------------------------------
;; Inspection / conversion
;; ---------------------------------------------------------------------------

(defn shape [t]
  (vec (core/->jvm (core/attr t "shape"))))

(defn item
  "Extract the single value of a 0-dim/1-element tensor as a Clojure number.
  Idempotent on values libpython-clj already auto-converted to JVM numbers."
  [t]
  (if (number? t) t (core/->jvm (core/call t "item"))))

(defn to-clj
  "Pull a tensor back into nested Clojure vectors. For validation and
  monitoring only — never call on big tensors inside a training step."
  [t]
  (let [x (core/->jvm (core/call (core/call (core/call t "detach") "cpu") "tolist"))]
    (if (sequential? x) (clojure.walk/postwalk #(if (sequential? %) (vec %) %) x) x)))

(defn to-device [t device] (core/call t "to" device))

;; ---------------------------------------------------------------------------
;; Elementwise + linear algebra
;; ---------------------------------------------------------------------------

(defn matmul [a b] (core/call core/torch "matmul" a b))
(defn add [a b] (core/call core/torch "add" a b))
(defn sub [a b] (core/call core/torch "sub" a b))
(defn mul [a b] (core/call core/torch "mul" a b))
(defn div [a b] (core/call core/torch "div" a b))
(defn exp [t] (core/call core/torch "exp" t))
(defn tanh [t] (core/call core/torch "tanh" t))
(defn sqrt [t] (core/call core/torch "sqrt" t))
(defn neg [t] (core/call core/torch "neg" t))

(defn transpose
  "Swap two dims; with no dims, the last two (matrix transpose)."
  ([t] (core/call core/torch "transpose" t -2 -1))
  ([t d0 d1] (core/call core/torch "transpose" t d0 d1)))

(defn reshape [t shape] (core/call t "reshape" (core/py-tuple shape)))

(defn norm
  "Frobenius/L2 norm as a Clojure double."
  [t]
  (item (core/call core/torch "norm" t)))

(defn sum
  ([t] (core/call core/torch "sum" t))
  ([t dim] (core/call-kw core/torch "sum" [t] {:dim dim})))

(defn sum-keep
  "Sum over dim, keeping it (size 1) for broadcasting."
  [t dim]
  (core/call-kw core/torch "sum" [t] {:dim dim :keepdim true}))

(defn mean-keep
  [t dim]
  (core/call-kw core/torch "mean" [t] {:dim dim :keepdim true}))

(defn log [t] (core/call core/torch "log" t))
(defn sigmoid [t] (core/call core/torch "sigmoid" t))

(defn tmax [t] (core/call core/torch "max" t))
(defn tmin [t] (core/call core/torch "min" t))

;; ---------------------------------------------------------------------------
;; Shape surgery / indexing / masking (attention support)
;; ---------------------------------------------------------------------------

(defn unsqueeze [t dim] (core/call t "unsqueeze" dim))
(defn squeeze [t dim] (core/call t "squeeze" dim))

(defn index-select
  "Rows (dim 0) / columns (dim 1) picked by an int64 index tensor."
  [t dim idx]
  (core/call core/torch "index_select" t dim idx))

(defn narrow
  "Slice `length` elements of dim starting at `start` (a view, no copy)."
  [t dim start length]
  (core/call core/torch "narrow" t dim start length))

(defn gather
  "torch.gather — e.g. per-row column pick: (gather lsm 1 (unsqueeze idx 1))."
  [t dim index]
  (core/call core/torch "gather" t dim index))

(defn gt [a b] (core/call core/torch "gt" a b))
(defn lt [a b] (core/call core/torch "lt" a b))
(defn logical-or [a b] (core/call core/torch "logical_or" a b))

(defn masked-fill
  "Set entries where the bool mask is true to `value` (e.g. ##-Inf)."
  [t mask value]
  (core/call t "masked_fill" mask value))

(defn scatter
  "Out-of-place scatter of a scalar value: result[i, index[i,j]] = value
  along `dim` (used to build one-hots)."
  [t dim index value]
  (core/call t "scatter" dim index value))

(defn index-add
  "Out-of-place index_add: rows of `src` added into `t` at positions `idx`
  along `dim` (the embedding-gradient scatter)."
  [t dim idx src]
  (core/call t "index_add" dim idx src))

;; ---------------------------------------------------------------------------
;; Curated op table (op/defop rows — direction doc §5.1 item 1).
;; One line per op; coercions/return conversions are declared, not coded.
;; Every row registers in op/op-registry and the table-driven test in
;; torch_op_test requires a test entry for each — keep them in sync.
;; ---------------------------------------------------------------------------

;; -- shape combinators --------------------------------------------------------
(defop cat          "Concatenate tensors along :dim (default 0)."             [tensors] {:coerce {0 :list}})
(defop stack        "Stack tensors along a NEW dim :dim (default 0)."         [tensors] {:coerce {0 :list}})
(defop split        "Split along :dim into chunks of `size` (int, or vector of sizes); returns a vector of tensors." [t size] {:ret :vec})
(defop chunk        "Split along :dim into `n` roughly equal chunks; returns a vector of tensors." [t n] {:ret :vec})
(defop unbind       "Remove :dim (default 0), returning its slices as a vector of tensors." [t] {:ret :vec})
(defop permute      "Reorder dims: (permute t [2 0 1])."                      [t dims] {:coerce {1 :tuple}})
(defop flatten      "Flatten dims :start_dim..:end_dim (defaults 0..-1) into one." [t] {})
(defop broadcast-to "Broadcast t to `shape` (expanded view, no copy)."        [t shape] {:coerce {1 :tuple}})

;; -- reductions ---------------------------------------------------------------
(defop mean     "Mean over all elements, or over :dim (:keepdim)."            [t] {})
(defop prod     "Product over all elements, or over :dim (:keepdim)."         [t] {})
(defop cumsum   "Cumulative sum along `dim`."                                 [t dim] {})
(defop argmax   "Index of the max element (flat), or along :dim (:keepdim)."  [t] {})
(defop argmin   "Index of the min element (flat), or along :dim (:keepdim)."  [t] {})
(defop topk     "[values indices] of the k largest (:dim/:largest/:sorted)."  [t k] {:ret :vec})
(defop variance "Variance; :dim/:keepdim/:correction (default correction 1 = unbiased, torch's default)." [t] {:py-name "var"})
(defop std      "Standard deviation; :dim/:keepdim/:correction as variance."  [t] {})

;; -- elementwise --------------------------------------------------------------
(defop abs     "Absolute value elementwise."                                  [t] {})
(defop pow     "t^p elementwise; p is a scalar or a tensor."                  [t p] {})
(defop clamp   "Clamp into [:min, :max] (either may be omitted)."             [t] {})
(defop where   "Elementwise (if condition x y); condition is a bool tensor."  [condition x y] {})
(defop floor   "Floor elementwise."                                           [t] {})
(defop ceil    "Ceiling elementwise."                                         [t] {})
(defop round   "Round-half-to-even elementwise."                              [t] {})
(defop sign    "Sign (−1/0/+1) elementwise."                                  [t] {})
(defop log1p   "log(1+t) elementwise."                                        [t] {})
(defop expm1   "exp(t)−1 elementwise."                                        [t] {})
(defop erf     "Gauss error function elementwise (torch's exact erf — NOT the gelu-tanh path; nn/gelu stays pinned to the oracle)." [t] {})
(defop maximum "Elementwise max of two tensors."                              [a b] {})
(defop minimum "Elementwise min of two tensors."                              [a b] {})

;; -- comparison / logical / selection -----------------------------------------
(defop eq            "Elementwise equality (bool tensor); cf. allclose."      [a b] {})
(defop ne            "Elementwise inequality (bool tensor)."                  [a b] {})
(defop ge            "Elementwise >= (bool tensor)."                          [a b] {})
(defop le            "Elementwise <= (bool tensor)."                          [a b] {})
(defop logical-and   "Elementwise boolean AND."                               [a b] {})
(defop logical-not   "Elementwise boolean NOT."                               [t] {})
(defop masked-select "1-D tensor of t's entries where the bool mask is true." [t mask] {})
(defop allclose      "True if all entries agree within :rtol/:atol."          [a b] {})

;; einsum is variadic, so it stays a hand-written tier-2 call:
(defn einsum
  "torch.einsum: (einsum \"ij,jk->ik\" a b)."
  [equation & tensors]
  (op/torch-fn "einsum" (cons equation tensors)))
