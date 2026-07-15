(ns mythjure.torch.module
  "Python nn.Module interop — the pretrained-weights door of the façade.

  The point: a module's weights become a transparent nested Clojure map
  (dotted state_dict keys split into keyword segments, numeric segments into
  longs), so the rest of the façade works on Python-defined models unchanged —
  optim/adam and autograd/param-grads over the live parameters,
  inspect/param-report for observability, plain get-in/assoc-in for surgery.

  state_dict vs live parameters: torch's state_dict includes buffers (running
  stats…) and by default returns DETACHED copies — right for serialization.
  Pass :keep-vars true (or use `params`) to get the LIVE tensors — right for
  optimization. Buffers surfacing in `params` are harmless to optimizers:
  they never receive a .grad, and Adam skips gradless tensors."
  (:require [clojure.string :as str]
            [mythjure.torch.core :as core]))

(defn- nn
  "torch.nn, reached as an attribute of the torch handle so nothing here
  needs its own import (and nothing loads before core/initialize!)."
  []
  (core/attr core/torch "nn"))

;; ---------------------------------------------------------------------------
;; Construction and invocation
;; ---------------------------------------------------------------------------

(defn make
  "Construct a torch.nn module by class name:
  (make \"Linear\" [5 3]), (make \"GELU\" [] {:approximate \"tanh\"})."
  ([class-name pos-args] (make class-name pos-args {}))
  ([class-name pos-args kw-args]
   (core/call-kw (nn) class-name pos-args kw-args)))

(defn sequential
  "nn.Sequential of the given modules (a Clojure seq)."
  [modules]
  (apply core/call (nn) "Sequential" modules))

(defn call
  "Apply a module to inputs — module(x …), i.e. __call__, so hooks run
  (unlike calling .forward directly)."
  [m & inputs]
  (apply core/call m "__call__" inputs))

(defn to-dtype!
  "Cast every parameter/buffer of m: (to-dtype! m :float64). IN PLACE —
  nn.Module.to mutates the module (unlike Tensor.to) — and returns m.
  Modules construct as float32 — cast before comparing against the float64
  oracle."
  [m dtype-kw]
  (core/call m "to" (core/dtype dtype-kw)))

(defn train-mode! [m] (core/call m "train") m)
(defn eval-mode!  [m] (core/call m "eval") m)

;; ---------------------------------------------------------------------------
;; state_dict ↔ Clojure param map
;; ---------------------------------------------------------------------------

(defn state-dict
  "Flat {\"dotted.key\" tensor} of a module's parameters and buffers.
  Detached by default; :keep-vars true → the live autograd tensors.
  CAUTION: detached ≠ independent — torch's state_dict detaches the GRAPH,
  not the STORAGE, so even the default tensors share memory with the live
  weights: a !-op on one writes into the module (and no guard can see it —
  detached tensors have no grad_fn). tensor/clone each value for a snapshot."
  [m & {:keys [keep-vars]}]
  (let [sd (if keep-vars
             (core/call-kw m "state_dict" [] {:keep_vars true})
             (core/call m "state_dict"))]
    (into {} (map (fn [k] [k (core/call sd "__getitem__" k)]))
          (core/py-keys sd))))

(defn- seg->key [s] (if (re-matches #"\d+" s) (Long/parseLong s) (keyword s)))
(defn- key->seg [k] (if (keyword? k) (name k) (str k)))

(defn state-dict->params
  "Nest a flat state dict into a Clojure param map:
  \"block.0.weight\" → {:block {0 {:weight T}}}."
  [flat]
  (reduce-kv (fn [acc k v] (assoc-in acc (mapv seg->key (str/split k #"\.")) v))
             {} flat))

(defn params->state-dict
  "Inverse of state-dict->params: flatten a nested param map to
  {\"dotted.key\" tensor}. Non-tensor leaves are dropped."
  [params]
  (letfn [(walk [m path]
            (cond (map? m) (mapcat (fn [[k v]] (walk v (conj path k))) m)
                  (core/tensor? m) [[(str/join "." (map key->seg path)) m]]
                  :else nil))]
    (into {} (walk params []))))

(defn params
  "The LIVE parameters (+ buffers) of a module as a nested Clojure map —
  feed it to optim/adam, autograd/param-grads, inspect/param-report."
  [m]
  (state-dict->params (state-dict m :keep-vars true)))

(defn- flat? [dict] (and (map? dict) (every? string? (keys dict))))

(defn load-state-dict!
  "Load weights into a module from a flat {\"dotted.key\" tensor} map or a
  nested param map. Strict by default (torch raises on any key mismatch);
  :strict false permits partial loads. Returns m."
  [m dict & {:keys [strict] :or {strict true}}]
  (let [flat (if (flat? dict) dict (params->state-dict dict))]
    (core/call-kw m "load_state_dict" [(core/py-dict flat)] {:strict strict}))
  m)

;; ---------------------------------------------------------------------------
;; Serialization (torch.save / torch.load)
;; ---------------------------------------------------------------------------

(defn save!
  "torch.save the weights of x — a module, a nested param map, or a flat
  state dict — to path (conventionally .pt). Returns path."
  [x path]
  (let [flat (cond (flat? x) x
                   (map? x)  (params->state-dict x)
                   :else     (state-dict x))]
    (core/call core/torch "save" (core/py-dict flat) path))
  path)

(defn load-params
  "torch.load a state dict from path (weights_only — tensors, not pickled
  code) as a nested Clojure param map. Load into a module with
  load-state-dict!."
  [path]
  (let [ld (core/call-kw core/torch "load" [path] {:weights_only true})]
    (state-dict->params
     (into {} (map (fn [k] [k (core/call ld "__getitem__" k)]))
           (core/py-keys ld)))))
