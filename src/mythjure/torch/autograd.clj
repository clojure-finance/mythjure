(ns mythjure.torch.autograd
  "Opt-in autograd surface over torch tensors — the general-purpose gradient
  path of the façade.

  Positioning (direction doc §1.4 / §4): mythjure's own experiments keep using
  the hand-derived VJPs (mythjure.backprop-torch) — they are the
  validation-grade path, oracle-pinned to the pure-Clojure implementation.
  This namespace exposes torch's reverse-mode autograd for everything else,
  and is itself validated against those VJPs: full-model gradients agree
  leaf-for-leaf to machine ε (see mythjure.torch-autograd-test). Nothing in
  the forward code changes — any graph built from façade ops is differentiable
  as-is.

  Semantics to know (pinned by tests, they are torch's, not ours):
   - `backward!` ACCUMULATES into `.grad` — clear between steps
     (`clear-grad!` / `clear-grads!`, or optim/zero-grad!).
   - A graph is freed by `backward!` unless :retain-graph is passed; calling
     it twice on the same graph without it is a Python-side error.
   - Tied parameters (one tensor, several paths in a param map) accumulate
     their gradient contributions automatically — no identity-dedup needed on
     this path, unlike optim/set-grads!.

  Param-map helpers mirror the optim convention: params are a nested Clojure
  map with tensor leaves; grads come back as the same tree."
  (:require [mythjure.torch.core :as core]
            [mythjure.torch.optim :as opt]))

;; ---------------------------------------------------------------------------
;; Per-tensor flags and gradient access
;; ---------------------------------------------------------------------------

(defn requires-grad!
  "Flag a tensor (in place) as a leaf that accumulates gradients. Returns t."
  ([t] (requires-grad! t true))
  ([t flag] (core/call t "requires_grad_" flag) t))

(defn requires-grad? [t]
  (boolean (core/->jvm (core/attr t "requires_grad"))))

(defn grad
  "The accumulated gradient tensor of a leaf, or nil if none yet."
  [t]
  (core/attr t "grad"))

(defn clear-grad!
  "Reset a leaf's .grad to None (torch's recommended zero_grad(set_to_none))."
  [t]
  (core/call t "__setattr__" "grad" nil)
  nil)

(defn detach
  "A view of t cut out of the graph: shares storage, requires_grad false."
  [t]
  (core/call t "detach"))

(defn grad-fn
  "Name of the autograd node that produced t (e.g. \"MulBackward0\"), or nil
  for leaves / detached tensors. Inspection only — plain Clojure data out."
  [t]
  (when-let [gf (core/attr t "grad_fn")]
    (let [nm (core/attr (core/attr gf "__class__") "__name__")]
      (if (string? nm) nm (core/->jvm nm)))))

;; ---------------------------------------------------------------------------
;; Backward
;; ---------------------------------------------------------------------------

(defn backward!
  "Reverse-mode through the graph rooted at t, accumulating into each leaf's
  .grad. t must be a scalar unless :gradient (the incoming vjp, same shape as
  t) is given. :retain-graph keeps the graph alive for another backward!;
  :create-graph tapes the backward itself (higher-order grads)."
  [t & {:keys [gradient retain-graph create-graph]}]
  (core/call-kw t "backward"
                (if gradient [gradient] [])
                (cond-> {}
                  retain-graph (assoc :retain_graph true)
                  create-graph (assoc :create_graph true)))
  nil)

;; ---------------------------------------------------------------------------
;; Grad-mode scoping
;; ---------------------------------------------------------------------------

(defn call-with-grad-mode
  "Run thunk inside a torch grad-mode context manager (\"no_grad\" /
  \"enable_grad\"), restoring the previous mode even on exception."
  [ctor-name thunk]
  (let [ctx (core/call core/torch ctor-name)]
    (core/call ctx "__enter__")
    (try (thunk)
         (finally (core/call ctx "__exit__" nil nil nil)))))

(defmacro no-grad
  "Evaluate body with taping off — ops build no graph (inference, updates)."
  [& body]
  `(call-with-grad-mode "no_grad" (fn [] ~@body)))

(defmacro enable-grad
  "Re-enable taping inside an enclosing no-grad."
  [& body]
  `(call-with-grad-mode "enable_grad" (fn [] ~@body)))

;; ---------------------------------------------------------------------------
;; Param-map helpers (nested Clojure map, tensor leaves — optim convention)
;; ---------------------------------------------------------------------------

(defn requires-grad-params!
  "Flag every unique tensor leaf of a param map as requiring grad.
  Returns params."
  [params]
  (doseq [t (opt/all-param-tensors params)]
    (requires-grad! t))
  params)

(defn clear-grads!
  "Reset .grad on every tensor leaf of a param map."
  [params]
  (doseq [t (opt/all-param-tensors params)]
    (clear-grad! t))
  nil)

(defn param-grads
  "Grads tree mirroring a param map: every tensor leaf replaced by its .grad;
  gradless leaves and non-tensor entries (config scalars…) are dropped. Tied
  leaves surface the SAME accumulated grad tensor at each of their paths."
  [m]
  (cond
    (map? m) (not-empty
              (into {} (keep (fn [[k v]]
                               (when-some [g (param-grads v)] [k g])))
                    m))
    (core/tensor? m) (grad m)
    :else nil))
