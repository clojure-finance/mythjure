(ns mythjure.torch.nn
  "Neural-net ops, with semantics pinned to the mythjure.linalg oracle:

   - gelu is the TANH APPROXIMATION (linalg/gelu), not torch's erf default.
   - layer-norm eps defaults to 1e-5, matching linalg/layernorm.
   - softmax normalizes over the last dim by default (row-wise).

  sdpa is provided for later speed work but must NOT be used until the manual
  attention path is oracle-validated — it's a fused kernel whose computation
  path can't be compared term-by-term against mythjure.backprop."
  (:require [mythjure.torch.core :as core]))

(defn layer-norm
  "LayerNorm over the last dim(s). normalized-shape is a vector, e.g. [d-model]."
  [input normalized-shape & {:keys [weight bias eps] :or {eps 1e-5}}]
  (core/call-kw core/F "layer_norm" [input (core/py-tuple normalized-shape)]
                (cond-> {:eps eps}
                  weight (assoc :weight weight)
                  bias   (assoc :bias bias))))

(defn gelu
  "GELU, tanh approximation — matches linalg/gelu (the GPT variant)."
  [x]
  (core/call-kw core/F "gelu" [x] {:approximate "tanh"}))

(defn softmax
  ([x] (softmax x -1))
  ([x dim] (core/call-kw core/F "softmax" [x] {:dim dim})))

(defn cross-entropy
  "Mean cross-entropy. logits [N × vocab], targets [N] of class ids."
  [logits targets]
  (core/call core/F "cross_entropy" logits targets))

(defn log-softmax
  ([x] (log-softmax x -1))
  ([x dim] (core/call-kw core/F "log_softmax" [x] {:dim dim})))

(defn softplus [x] (core/call core/F "softplus" x))

(defn sdpa
  "Fused scaled-dot-product attention. DO NOT use before the manual attention
  path passes oracle validation (see direction doc §1.4)."
  [q k v & {:keys [causal? dropout] :or {causal? true dropout 0.0}}]
  (core/call-kw core/F "scaled_dot_product_attention" [q k v]
                {:is_causal causal? :dropout_p dropout}))
