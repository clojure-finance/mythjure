(ns mythjure.torch.op
  "Generic dispatch surface + declarative op table for the torch façade
  (direction doc §5.1 item 1 — the op-coverage strategy).

  Three tiers of coverage:

  1. CURATED — named fns in mythjure.torch.{tensor,nn,…}, hand-written or
     declared with `defop` below. Documented, tested, semantics pinned where
     torch's defaults diverge from the oracle (e.g. nn/gelu's tanh pin).
     Mutation is opt-in and !-marked at this tier (guarded in-place rows,
     :aliasing declared per op — see the defop docstring and the mutation
     rules in mythjure.torch.tensor's ns docstring).

  2. GENERIC — `torch-fn` / `nn-fn` / `method` here call ANY torch op by
     name today, no wrapper needed. Coercion is EXPLICIT at this tier: wrap
     shape args in core/py-tuple, tensor lists in core/py-list, dicts in
     core/py-dict yourself. The generic layer never guesses — whether torch's
     C++ layer accepts an auto-bridged JVM collection is per-op luck
     (cat/stack do, layer_norm/load_state_dict don't), and a wrong guess is
     silent breakage.

  3. DISCOVERY — libpython-clj can generate a whole namespace from the torch
     module for REPL exploration, python docstrings included:

       (require '[libpython-clj2.require :refer [require-python]])
       (require-python '[torch :as pt])
       (clojure.repl/doc pt/topk)

     Use it to FIND ops; CALL them through tier 1/2 (require-python fns have
     none of the façade's coercion/scalar hardening).

  Promotion policy (the library's law, formerly raw-call's docstring): when
  a tier-2 call site stabilizes, add a one-line `defop` row in the owning
  namespace — the row lands in `op-registry`, and the table-driven test in
  torch_op_test insists every row mythjure.torch.tensor registers has a test
  entry (extend that pattern per namespace as tables grow)."
  (:require [clojure.string :as str]
            [mythjure.torch.core :as core]))

;; ---------------------------------------------------------------------------
;; Generic dispatch (tier 2)
;; ---------------------------------------------------------------------------

(defn -module!
  "Internal: a module handle, or a readable error before initialize!."
  [m what]
  (when (nil? m)
    (throw (ex-info (str what " module handle is nil — call "
                         "(mythjure.torch.core/initialize!) first")
                    {:module what})))
  m)

(defn -dispatch
  "Internal: positional-or-kwargs call on a Python object."
  [obj py-name pos kw]
  (if (seq kw)
    (core/call-kw obj py-name (vec pos) kw)
    (apply core/call obj py-name pos)))

(defn torch-fn
  "Call torch.<py-name> with positional args and optional kwargs:
    (torch-fn \"einsum\" [\"ij,jk->ik\" a b])
    (torch-fn \"full\" [(core/py-tuple [2 2]) 3.0] {:dtype (core/dtype :float64)})
  Coercion is explicit — see the namespace docstring."
  ([py-name pos-args] (torch-fn py-name pos-args nil))
  ([py-name pos-args kw-args]
   (-dispatch (-module! core/torch "torch") py-name pos-args kw-args)))

(defn nn-fn
  "Call torch.nn.functional.<py-name>; same contract as torch-fn."
  ([py-name pos-args] (nn-fn py-name pos-args nil))
  ([py-name pos-args kw-args]
   (-dispatch (-module! core/F "torch.nn.functional") py-name pos-args kw-args)))

(defn method
  "Call a method on a tensor/module/any Python object; same contract:
    (method t \"to\" [\"cpu\"])"
  ([obj py-name pos-args] (method obj py-name pos-args nil))
  ([obj py-name pos-args kw-args]
   (-dispatch obj py-name pos-args kw-args)))

;; ---------------------------------------------------------------------------
;; defop — the declarative op table (tier 1 growth mechanism)
;; ---------------------------------------------------------------------------

(defn op-name->py-name
  "Clojure op name → python name: dashes → underscores, a trailing !
  (in-place op) → trailing underscore. (op-name->py-name \"logical-not\")
  = \"logical_not\"; \"add!\" = \"add_\"."
  [s]
  (-> (name s)
      (str/replace #"!$" "_")
      (str/replace "-" "_")))

(defn -coerce
  "Internal: explicit positional coercion for defop rows."
  [v kind]
  (case kind
    :tuple (core/py-tuple v)
    :list  (core/py-list v)
    :dict  (core/py-dict v)))

(defn -ret
  "Internal: return conversion for defop rows."
  [r kind]
  (case kind
    :item (if (number? r) r (core/->jvm (core/call r "item")))
    :vec  (core/py-vec r)
    nil   r))

(defn -guard-inplace!
  "Internal: refuse in-place mutation of an autograd INTERMEDIATE (a tensor
  with a grad_fn). Some backward pass may have saved that tensor; mutating it
  makes backward! fail with torch's version-counter error — far from the
  mutation. This guard moves the error to the mutation site. Leaves are not
  guarded: torch itself refuses in-place on a requires-grad leaf under grad
  mode, and allows it under no-grad (the optimizer pattern)."
  [obj op-name unsafe?]
  (when-not unsafe?
    (when-let [gf (core/attr obj "grad_fn")]
      (let [nm   (core/attr (core/attr gf "__class__") "__name__")
            node (if (string? nm) nm (core/->jvm nm))]
        (throw (ex-info (str op-name " would mutate an autograd intermediate in "
                             "place (grad_fn " node ") — a backward pass that "
                             "saved this tensor would fail with torch's "
                             "version-counter error at backward!, far from "
                             "here. Clone first (mythjure.torch.tensor/clone) "
                             "and mutate the copy, or pass :unsafe true if no "
                             "backward will traverse the current graph.")
                        {:op op-name :grad-fn node}))))))

(defonce op-registry
  ;; qualified op symbol → its defop spec; the greppable table, and what the
  ;; table-driven test iterates to enforce every-registered-op-is-tested.
  (atom (sorted-map)))

(defmacro defop
  "One row of the curated op table:

    (defop cat
      \"Concatenate tensors along :dim (default 0).\"
      [tensors]
      {:coerce {0 :list}})

  expands to (defn cat [tensors & {:as kw}] …) calling torch.cat with the
  coerced positionals and kw passed through as Python kwargs, and registers
  the row in `op-registry`.

  argv: fixed positional params (symbols only); every generated fn also
  accepts trailing kwargs.

  A trailing ! in the op name declares an IN-PLACE op (py-name munges to the
  trailing-underscore torch form, e.g. add! → add_): the generated fn mutates
  its first arg, returns it (torch returns self — threads), and guard-throws
  at the call site when the target is an autograd intermediate (see
  -guard-inplace!). The caller-only kwarg :unsafe true bypasses the guard and
  is stripped before dispatch.

  spec keys:
    :target     :torch (default) → torch.<py-name>
                :nn              → torch.nn.functional.<py-name>
                :method          → (first arg).<py-name>(rest…)
    :py-name    override; default = (op-name->py-name name)
    :coerce     {argv-index :tuple|:list|:dict} — explicit coercions
    :default-kw {kwarg default} — kwargs sent unless the caller overrides;
                semantic pins live here (e.g. gelu's {:approximate \"tanh\"})
    :ret        nil (live pyobject, default)
                :item (0-dim tensor → Clojure number)
                :vec  (python tuple → vector of live pyobjects)
    :aliasing   whether the result can SHARE STORAGE with an input; absent =
                fresh storage (the torch default for out-of-place ops)
                :view            (always shares)
                :input-dependent (view or copy, decided at runtime — e.g.
                                  reshape/flatten by contiguity)
                :copy            (explicitly fresh, e.g. clone)
                :in-place        (set automatically for ! ops)"
  [op-name doc argv spec]
  (assert (vector? argv) "defop: argv must be a vector")
  (assert (every? symbol? argv) "defop: argv is fixed positional symbols only")
  (let [{:keys [target py-name coerce default-kw ret aliasing]
         :or   {target :torch}} spec
        in-place? (str/ends-with? (name op-name) "!")
        aliasing  (if in-place? :in-place aliasing)]
    (assert (every? #{:target :py-name :coerce :default-kw :ret :aliasing} (keys spec))
            (str "defop " op-name ": unknown spec key"))
    (assert (#{:torch :nn :method} target) (str "defop " op-name ": bad :target"))
    (assert (contains? #{nil :item :vec} ret) (str "defop " op-name ": bad :ret"))
    (assert (contains? #{nil :view :input-dependent :copy :in-place} aliasing)
            (str "defop " op-name ": bad :aliasing"))
    (assert (or (not in-place?) (contains? #{nil :in-place} (:aliasing spec)))
            (str "defop " op-name ": a ! op is :aliasing :in-place by definition"))
    (assert (or (not in-place?) (seq argv))
            (str "defop " op-name ": an in-place op needs a target arg"))
    (assert (or (not in-place?) (not (contains? coerce 0)))
            (str "defop " op-name ": :coerce on the mutation target — the "
                 "guard would check the uncoerced value"))
    (assert (every? #{:tuple :list :dict} (vals coerce))
            (str "defop " op-name ": bad :coerce kind"))
    (assert (every? #(< -1 % (count argv)) (keys coerce))
            (str "defop " op-name ": :coerce index out of range"))
    (let [py-name (or py-name (op-name->py-name op-name))
          kw-sym  (gensym "kw")
          coerced (vec (map-indexed
                        (fn [i s] (if-let [k (get coerce i)] `(-coerce ~s ~k) s))
                        argv))
          [obj pos] (if (= target :method)
                      [(first coerced) (vec (rest coerced))]
                      [(if (= target :nn)
                         `(-module! core/F "torch.nn.functional")
                         `(-module! core/torch "torch"))
                       coerced])
          kw-expr  (if default-kw `(merge ~default-kw ~kw-sym) kw-sym)
          dispatch `(-ret (-dispatch ~obj ~py-name ~pos ~kw-expr) ~ret)
          body     (if in-place?
                     `(let [unsafe# (:unsafe ~kw-sym)
                            ~kw-sym (not-empty (dissoc ~kw-sym :unsafe))]
                        (-guard-inplace! ~(first argv) ~(str op-name) unsafe#)
                        ~dispatch)
                     dispatch)]
      `(do (defn ~op-name ~doc [~@argv & {:as ~kw-sym}] ~body)
           (swap! op-registry assoc
                  '~(symbol (str (ns-name *ns*)) (str op-name))
                  {:py-name ~py-name :target ~target
                   :coerce ~coerce :default-kw ~default-kw :ret ~ret
                   :aliasing ~aliasing :in-place? ~in-place?})
           (var ~op-name)))))
