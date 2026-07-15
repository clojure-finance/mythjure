(ns mythjure.torch.tensor
  "Tensor creation and manipulation — the Clojure face of torch tensors.

  Conventions mirror mythjure.linalg where they overlap: a matrix is
  [rows × cols], `transpose` with no dims swaps the last two. Constructors
  default to core/*device* and :float32; pass :dtype :float64 when comparing
  against the pure-Clojure oracle.

  Mutation rules (direction doc §5.1 item 2): no unmarked op in this
  namespace mutates its arguments — mutation is opt-in via the !-suffixed
  ops, which are guarded against corrupting autograd graphs (op/defop).
  Torch aliasing is kept faithfully: some ops return VIEWS sharing the
  input's storage, so a ! op can write through them into the source. The
  rule of thumb: no !, no aliasing bugs. `aliasing-table` (bottom of this
  file) is the greppable record of which ops can alias; `clone` breaks
  aliasing, `data-ptr` diagnoses it."
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

(defn to-device
  "Move t to a device. Returns t ITSELF when it is already there (torch's
  .to is a no-op then) — a copy only when a move actually happens."
  [t device]
  (core/call t "to" device))

(defn data-ptr
  "Address of the first element of t's storage, as a Clojure long — the
  aliasing diagnostic: equal data-ptrs ⇒ shared buffer. The converse does
  NOT hold (an offset view, e.g. a mid-tensor narrow, overlaps its base at
  a different address), so unequal ptrs prove nothing."
  [t]
  (item (core/call t "data_ptr")))

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
  "Swap two dims; with no dims, the last two (matrix transpose).
  Returns a VIEW — shares storage with t."
  ([t] (core/call core/torch "transpose" t -2 -1))
  ([t d0 d1] (core/call core/torch "transpose" t d0 d1)))

(defn reshape
  "New shape, same elements. A VIEW when t is contiguous, a silent COPY
  otherwise (torch semantics — aliasing is decided by t's memory layout at
  runtime, not by this op). `clone` if you need guaranteed independence."
  [t shape]
  (core/call t "reshape" (core/py-tuple shape)))

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

(defn unsqueeze "Insert a size-1 dim (a view)." [t dim] (core/call t "unsqueeze" dim))
(defn squeeze "Drop a size-1 dim (a view)." [t dim] (core/call t "squeeze" dim))

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
(defop split        "Split along :dim into chunks of `size` (int, or vector of sizes); returns a vector of VIEWS." [t size] {:ret :vec :aliasing :view})
(defop chunk        "Split along :dim into `n` roughly equal chunks; returns a vector of VIEWS." [t n] {:ret :vec :aliasing :view})
(defop unbind       "Remove :dim (default 0), returning its slices as a vector of VIEWS." [t] {:ret :vec :aliasing :view})
(defop permute      "Reorder dims: (permute t [2 0 1]). A view."              [t dims] {:coerce {1 :tuple} :aliasing :view})
(defop flatten      "Flatten dims :start_dim..:end_dim (defaults 0..-1) into one. View when t is contiguous, copy otherwise (like reshape)." [t] {:aliasing :input-dependent})
(defop broadcast-to "Broadcast t to `shape` (expanded view, no copy)."        [t shape] {:coerce {1 :tuple} :aliasing :view})

;; -- copies / layout ----------------------------------------------------------
(defop clone      "Deep copy: fresh storage, breaks all aliasing. Differentiable (taped as identity) — `autograd/detach` for the opposite trade." [t] {:aliasing :copy})
(defop contiguous "t in contiguous layout: returns t ITSELF when already contiguous, a fresh copy otherwise." [t] {:target :method :aliasing :input-dependent})

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

;; -- in-place (the ONLY mutating ops in the façade; ns docstring has the rules;
;;    all guard-throw on autograd intermediates, take :unsafe true to bypass,
;;    and return their mutated target for threading) --------------------------
(defop add!   "In-place t += other (elementwise or broadcast)."               [t other] {:target :method})
(defop sub!   "In-place t -= other."                                          [t other] {:target :method})
(defop mul!   "In-place t *= other."                                          [t other] {:target :method})
(defop div!   "In-place t /= other."                                          [t other] {:target :method})
(defop clamp! "In-place clamp into [:min, :max] (either may be omitted)."     [t] {:target :method})
(defop zero!  "In-place fill with zeros."                                     [t] {:target :method})
(defop fill!  "In-place fill with a scalar."                                  [t value] {:target :method})

;; einsum is variadic, so it stays a hand-written tier-2 call:
(defn einsum
  "torch.einsum: (einsum \"ij,jk->ik\" a b). Contractions allocate, but a
  pure-permutation equation (\"ij->ji\", even \"ij->ij\") lowers to a VIEW."
  [equation & tensors]
  (op/torch-fn "einsum" (cons equation tensors)))

;; ---------------------------------------------------------------------------
;; Aliasing table — which ops can return a tensor sharing storage with an
;; input (or mutate one). Everything NOT listed returns fresh storage.
;; torch_mutation_test pins every claim empirically (data-ptr / mutation
;; witnesses), so a torch upgrade that changes an op's aliasing fails a test.
;; ---------------------------------------------------------------------------

(def hand-written-aliasing
  "Aliasing facts for the hand-written (non-defop) ops in this namespace;
  defop rows carry theirs as :aliasing in op/op-registry."
  '{transpose :view
    narrow    :view
    squeeze   :view
    unsqueeze :view
    reshape   :input-dependent
    to-device :input-dependent
    einsum    :input-dependent})

(defn aliasing-table
  "Every op in this namespace whose result can share storage with (or
  mutates) an input → :view | :input-dependent | :copy | :in-place.
  Merges the defop rows' :aliasing declarations with the hand-written map."
  []
  (into hand-written-aliasing
        (keep (fn [[sym row]]
                (when (and (= "mythjure.torch.tensor" (namespace sym))
                           (:aliasing row))
                  [(symbol (name sym)) (:aliasing row)])))
        @op/op-registry))
