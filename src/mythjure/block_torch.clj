(ns mythjure.block-torch
  "Torch backend for mythjure.block/forward — same param structure (tensor
  leaves via mythjure.model-torch/params->torch), same math, same result.

  The oracle splits Q/K/V into per-head column slabs and runs each head
  separately; here the identical split is a [S, H, d_head] reshape (row-major
  ⇒ contiguous columns per head) followed by one batched matmul over the head
  dimension. Attention is MANUAL — Q·Kᵀ/√d → mask → softmax → ·V — never the
  fused SDPA, so the computation stays term-for-term comparable to
  mythjure.backprop for Phase 3 (direction doc §1.4)."
  (:require [mythjure.torch.tensor :as t]
            [mythjure.torch.nn :as nn]))

(defn attn-mask
  "Bool [S×S], true where attention is FORBIDDEN. window nil ⇒ causal (j>i);
  integer w ⇒ banded causal (j>i or j<i−w), matching linalg's masks."
  [S window]
  (let [i (t/unsqueeze (t/arange S :dtype :int64) 1)
        j (t/unsqueeze (t/arange S :dtype :int64) 0)]
    (if window
      (t/logical-or (t/gt j i) (t/lt j (t/sub i window)))
      (t/gt j i))))

(defn multi-head-attention
  "Multi-head causal (or windowed) self-attention over X [S × d_model].
  Returns the mixed output [S × d_model]."
  [{:keys [n-heads window] {:keys [wq wk wv wo]} :attn} X]
  (let [[S d-model] (t/shape X)
        d-head (quot d-model n-heads)
        proj (fn [w] (-> (t/matmul X w)
                         (t/reshape [S n-heads d-head])
                         (t/transpose 0 1)))          ; [H, S, d_head]
        Q (proj wq) K (proj wk) V (proj wv)
        scores (t/mul (t/matmul Q (t/transpose K))    ; [H, S, S]
                      (/ 1.0 (Math/sqrt d-head)))
        masked (t/masked-fill scores (attn-mask S window) ##-Inf)
        weights (nn/softmax masked)
        out (-> (t/matmul weights V)                  ; [H, S, d_head]
                (t/transpose 0 1)
                (t/reshape [S d-model]))]
    (t/matmul out wo)))

(defn mlp
  "Position-wise GELU MLP: GELU(X·W1+b1)·W2+b2 (bias adds broadcast row-wise)."
  [{{:keys [w1 b1 w2 b2]} :mlp} X]
  (-> (t/add (t/matmul X w1) b1)
      nn/gelu
      (t/matmul w2)
      (t/add b2)))

(defn forward
  "Pre-LN block: a = X + MHA(LN1(X)); Y = a + MLP(LN2(a)). Same signature and
  result as mythjure.block/forward."
  [{:keys [ln1 ln2 d-model] :as params} X]
  (let [d (or d-model (last (t/shape X)))
        a (t/add X (multi-head-attention
                    params (nn/layer-norm X [d] :weight (:gamma ln1) :bias (:beta ln1))))
        m (mlp params (nn/layer-norm a [d] :weight (:gamma ln2) :bias (:beta ln2)))]
    (t/add a m)))
