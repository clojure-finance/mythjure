(ns mythjure.block
  "Step 2: a real (decoder) transformer block, pure clojure.core on top of
  mythjure.linalg. This is the R that goes inside the loop — the thing the
  recurrent-depth model applies over and over.

  Architecture (pre-LayerNorm, the modern default):

      a = X + MultiHeadCausalAttention(LN1(X))
      Y = a + MLP(LN2(a))

  with MLP(z) = GELU(z·W1 + b1)·W2 + b2.

  X is a [seq_len × d_model] matrix: one row per token position. Attention is
  CAUSAL (a token sees only itself and earlier tokens) so the block is suitable
  for autoregressive language modeling. Multi-head: d_model is split into
  n_heads contiguous slices of width d_head = d_model/n_heads; each head runs its
  own scaled dot-product attention; heads are concatenated and mixed by Wo.

  No backprop yet — this is the forward pass, built to be inspected in the REPL
  and verified piece by piece. Manual BPTT comes next."
  (:require [mythjure.linalg :as la]))

;; ---------------------------------------------------------------------------
;; Parameter initialization (one block)
;; ---------------------------------------------------------------------------

(defn init-params
  "Deterministically initialize one transformer block.

  config keys: :d-model (default 32) :n-heads (4) :d-ff (4×d-model) :seed (0)
               :scale (input-projection init std, default 1/√d-model)
               :out-scale (output-projection init std, default scale/√2)

  Output projections (Wo, W2) get a smaller init so the residual stream doesn't
  blow up immediately — important once this block is LOOPED."
  [{:keys [d-model n-heads d-ff seed scale out-scale window]
    :or {d-model 32 n-heads 4 seed 0}}]
  (let [d-ff      (or d-ff (* 4 d-model))
        scale     (or scale (/ 1.0 (Math/sqrt d-model)))
        out-scale (or out-scale (/ scale (Math/sqrt 2.0)))
        ;; distinct seeds per matrix so they aren't identical
        rm        (fn [s r c sc] (la/rand-matrix (+ seed s) r c sc))]
    {:n-heads n-heads
     :d-model d-model
     :window window   ; nil ⇒ full causal attention; integer w ⇒ banded (windowed)
     :ln1 {:gamma (la/ones d-model) :beta (la/zeros d-model)}
     :ln2 {:gamma (la/ones d-model) :beta (la/zeros d-model)}
     :attn {:wq (rm 1 d-model d-model scale)
            :wk (rm 2 d-model d-model scale)
            :wv (rm 3 d-model d-model scale)
            :wo (rm 4 d-model d-model out-scale)}
     :mlp {:w1 (rm 5 d-model d-ff scale)
           :b1 (la/zeros d-ff)
           :w2 (rm 6 d-ff d-model out-scale)
           :b2 (la/zeros d-model)}}))

;; ---------------------------------------------------------------------------
;; Multi-head self-attention (causal, or windowed via the block's :window)
;; ---------------------------------------------------------------------------

(defn col-slice
  "Columns [a, b) of every row — used to carve a matrix into per-head slabs."
  [M a b]
  (mapv #(subvec % a b) M))

(defn head-slices
  "Split a [seq × d_model] matrix into n-heads slabs of width d_head."
  [M n-heads]
  (let [d-model (count (first M))]
    (assert (zero? (mod d-model n-heads))
            (str "d-model (" d-model ") must be divisible by n-heads (" n-heads ")"))
    (let [d-head (quot d-model n-heads)]
      (mapv (fn [h] (col-slice M (* h d-head) (* (inc h) d-head)))
            (range n-heads)))))

(defn scaled-dot-attention
  "Single-head scaled dot-product attention. Q,K,V are [seq × d_head]. `window`:
   nil ⇒ full causal mask; integer w ⇒ banded (windowed) causal mask. Returns
   [seq × d_head] plus the attention weights."
  ([Q K V] (scaled-dot-attention Q K V nil))
  ([Q K V window]
   (let [d-head (count (first Q))
         scale  (/ 1.0 (Math/sqrt d-head))
         scores (la/scale-mat scale (la/matmul Q (la/transpose K)))
         masked (if window (la/apply-windowed-mask scores window) (la/apply-causal-mask scores))
         weights (la/softmax-rows masked)
         out    (la/matmul weights V)]
     {:out out :weights weights})))

(defn multi-head-attention
  "Full multi-head self-attention over X [seq × d_model]. Respects the block's
   `:window` (nil ⇒ full causal). Returns {:out [seq × d_model] :weights ...}."
  [{:keys [n-heads window] {:keys [wq wk wv wo]} :attn} X]
  (let [Q (la/matmul X wq)
        K (la/matmul X wk)
        V (la/matmul X wv)
        heads (mapv #(scaled-dot-attention %1 %2 %3 window)
                    (head-slices Q n-heads)
                    (head-slices K n-heads)
                    (head-slices V n-heads))
        ;; concat head outputs along the feature axis, then mix with Wo
        concat (apply mapv (fn [& rows] (vec (apply clojure.core/concat rows)))
                      (mapv :out heads))            ; row-wise concat of slabs
        out (la/matmul concat wo)]
    {:out out :weights (mapv :weights heads)}))

;; ---------------------------------------------------------------------------
;; Position-wise MLP
;; ---------------------------------------------------------------------------

(defn mlp
  "GELU MLP applied independently to each position: GELU(X·W1+b1)·W2+b2."
  [{{:keys [w1 b1 w2 b2]} :mlp} X]
  (-> (la/linear X w1 b1)
      (la/gelu-mat)
      (la/linear w2 b2)))

;; ---------------------------------------------------------------------------
;; The block
;; ---------------------------------------------------------------------------

(defn forward
  "Pre-LN transformer block forward pass.
     a = X + MHA(LN1(X))
     Y = a + MLP(LN2(a))
   Returns the refined sequence Y [seq × d_model]."
  [{:keys [ln1 ln2] :as params} X]
  (let [attn-out (:out (multi-head-attention params (la/layernorm X (:gamma ln1) (:beta ln1))))
        a (la/mat+ X attn-out)
        mlp-out (mlp params (la/layernorm a (:gamma ln2) (:beta ln2)))]
    (la/mat+ a mlp-out)))

(defn forward+weights
  "Like `forward` but also returns the attention weights for inspection."
  [{:keys [ln1 ln2] :as params} X]
  (let [{attn-out :out w :weights} (multi-head-attention params (la/layernorm X (:gamma ln1) (:beta ln1)))
        a (la/mat+ X attn-out)
        mlp-out (mlp params (la/layernorm a (:gamma ln2) (:beta ln2)))]
    {:out (la/mat+ a mlp-out) :weights w}))

(comment
  (require '[mythjure.linalg :as la])
  (def p (init-params {:d-model 32 :n-heads 4 :seed 7}))
  (def X (la/rand-matrix 99 16 32 1.0))          ; seq_len=16, d_model=32
  (la/shape (forward p X))                        ; => [16 32], shape preserved
  ;; attention weights are row-stochastic and causal:
  (-> (forward+weights p X) :weights first first) ; head 0, query 0 -> [1 0 0 ...]
  )
