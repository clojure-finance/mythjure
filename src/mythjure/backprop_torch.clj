(ns mythjure.backprop-torch
  "Torch backend for mythjure.backprop: the same hand-derived VJPs and BPTT,
  with torch tensors as the substrate — NO autograd anywhere. Every function
  mirrors its oracle namesake one-to-one (same cache keys, same grad tree),
  so ladder steps 2–3 can compare leaf-for-leaf.

  Where the oracle loops over heads/rows, this batches: heads live in a
  leading [H, S, d_head] dimension, LayerNorm/softmax VJPs reduce over the
  last dim with keepdim broadcasting. The math is identical."
  (:require [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]
            [mythjure.torch.nn :as nn]
            [mythjure.block-torch :as blkt]))

(def ^:private eps 1e-5)

;; ---------------------------------------------------------------------------
;; Elementwise / per-row VJPs
;; ---------------------------------------------------------------------------

(def ^:private gelu-c (Math/sqrt (/ 2.0 Math/PI)))
(def ^:private gelu-a 0.044715)

(defn gelu'
  "Derivative of the tanh-approximation GELU, elementwise on a tensor."
  [x]
  (let [x2 (t/mul x x)
        u  (t/mul (t/add x (t/mul (t/mul x x2) gelu-a)) gelu-c)
        th (t/tanh u)
        u' (t/mul (t/add (t/mul x2 (* 3.0 gelu-a)) 1.0) gelu-c)]
    (t/add (t/mul (t/add th 1.0) 0.5)
           (t/mul (t/mul (t/mul x 0.5) (t/sub (t/ones-like th) (t/mul th th))) u'))))

(defn softmax-backward
  "dz = s ⊙ (g − ⟨g,s⟩) over the last dim; batched. Masked entries (s=0)
  get zero gradient automatically."
  [g s]
  (t/mul s (t/sub g (t/sum-keep (t/mul g s) -1))))

(defn ln-backward
  "LayerNorm backward over the last dim (rows). Returns {:dX :dgamma :dbeta};
  dgamma/dbeta summed across rows, matching the oracle."
  [G X gamma eps*]
  (let [d     (last (t/shape X))
        mu    (t/mean-keep X -1)
        xc    (t/sub X mu)
        var*  (t/mean-keep (t/mul xc xc) -1)
        inv   (t/rsqrt (t/add var* eps*))
        xhat  (t/mul xc inv)
        dxhat (t/mul G gamma)
        s1    (t/sum-keep dxhat -1)
        s2    (t/sum-keep (t/mul dxhat xhat) -1)
        dX    (t/mul (t/div inv d)
                     (t/sub (t/sub (t/mul dxhat d) s1) (t/mul xhat s2)))]
    {:dX dX
     :dgamma (t/sum (t/mul G xhat) 0)
     :dbeta  (t/sum G 0)}))

;; ---------------------------------------------------------------------------
;; Multi-head attention: cached forward + backward (batched over heads)
;; ---------------------------------------------------------------------------

(defn- split-heads [M S n-heads d-head]
  (-> M (t/reshape [S n-heads d-head]) (t/transpose 0 1)))     ; [H, S, dh]

(defn- merge-heads [Mh S d-model]
  (-> Mh (t/transpose 0 1) (t/reshape [S d-model])))           ; [S, D]

(defn attn-forward
  "Forward caching everything backward needs. window nil ⇒ full causal."
  ([X wq wk wv wo n-heads] (attn-forward X wq wk wv wo n-heads nil))
  ([X wq wk wv wo n-heads window]
   (let [[S d-model] (t/shape X)
         d-head (quot d-model n-heads)
         scale  (/ 1.0 (Math/sqrt d-head))
         Qh (split-heads (t/matmul X wq) S n-heads d-head)
         Kh (split-heads (t/matmul X wk) S n-heads d-head)
         Vh (split-heads (t/matmul X wv) S n-heads d-head)
         scores (t/mul (t/matmul Qh (t/transpose Kh)) scale)
         masked (t/masked-fill scores (blkt/attn-mask S window) ##-Inf)
         W (nn/softmax masked)                                 ; [H, S, S]
         Oh (t/matmul W Vh)                                    ; [H, S, dh]
         Z (merge-heads Oh S d-model)]
     {:attn (t/matmul Z wo)
      :cache {:X X :Qh Qh :Kh Kh :Vh Vh :W W :Z Z :scale scale
              :wq wq :wk wk :wv wv :wo wo :nh n-heads :dh d-head}})))

(defn attn-backward
  "Given dL/d(attn), return dX and the projection-weight gradients."
  [dAttn {:keys [X Qh Kh Vh W Z scale wq wk wv wo nh dh]}]
  (let [[S d-model] (t/shape X)
        dZ  (t/matmul dAttn (t/transpose wo))
        dWo (t/matmul (t/transpose Z) dAttn)
        dOh (split-heads dZ S nh dh)                           ; [H, S, dh]
        dW  (t/matmul dOh (t/transpose Vh))                    ; into softmax weights
        dVh (t/matmul (t/transpose W) dOh)
        dScores (softmax-backward dW W)
        dRaw (t/mul dScores scale)
        dQh (t/matmul dRaw Kh)
        dKh (t/matmul (t/transpose dRaw) Qh)
        dQ (merge-heads dQh S d-model)
        dK (merge-heads dKh S d-model)
        dV (merge-heads dVh S d-model)]
    {:dX (-> (t/matmul dQ (t/transpose wq))
             (t/add (t/matmul dK (t/transpose wk)))
             (t/add (t/matmul dV (t/transpose wv))))
     :dwq (t/matmul (t/transpose X) dQ)
     :dwk (t/matmul (t/transpose X) dK)
     :dwv (t/matmul (t/transpose X) dV)
     :dwo dWo}))

;; ---------------------------------------------------------------------------
;; Pre-LN block: cached forward + backward
;; ---------------------------------------------------------------------------

(defn block-forward
  "Same math as block-torch/forward, caching intermediates. {:Y :cache}."
  [{:keys [ln1 ln2 attn mlp n-heads window d-model]} X]
  (let [d (or d-model (last (t/shape X)))
        n1 (nn/layer-norm X [d] :weight (:gamma ln1) :bias (:beta ln1))
        {att :attn ac :cache} (attn-forward n1 (:wq attn) (:wk attn) (:wv attn)
                                            (:wo attn) n-heads window)
        a  (t/add X att)
        n2 (nn/layer-norm a [d] :weight (:gamma ln2) :bias (:beta ln2))
        z1 (t/add (t/matmul n2 (:w1 mlp)) (:b1 mlp))
        h2 (nn/gelu z1)
        m  (t/add (t/matmul h2 (:w2 mlp)) (:b2 mlp))]
    {:Y (t/add a m) :cache {:X X :n1 n1 :ac ac :a a :n2 n2 :z1 z1 :h2 h2}}))

(defn block-backward
  "Given dL/dY, return {:dX :grads} with the oracle's grad-tree shape."
  [{:keys [ln1 ln2 mlp]} dY {:keys [X ac a n2 z1 h2]}]
  (let [dm  dY
        dW2 (t/matmul (t/transpose h2) dm)
        db2 (t/sum dm 0)
        dh2 (t/matmul dm (t/transpose (:w2 mlp)))
        dz1 (t/mul dh2 (gelu' z1))
        dW1 (t/matmul (t/transpose n2) dz1)
        db1 (t/sum dz1 0)
        dn2 (t/matmul dz1 (t/transpose (:w1 mlp)))
        l2  (ln-backward dn2 a (:gamma ln2) eps)
        da  (t/add dY (:dX l2))
        ab  (attn-backward da ac)
        l1  (ln-backward (:dX ab) X (:gamma ln1) eps)
        dX  (t/add da (:dX l1))]
    {:dX dX
     :grads {:ln1 {:gamma (:dgamma l1) :beta (:dbeta l1)}
             :ln2 {:gamma (:dgamma l2) :beta (:dbeta l2)}
             :attn {:wq (:dwq ab) :wk (:dwk ab) :wv (:dwv ab) :wo (:dwo ab)}
             :mlp {:w1 dW1 :b1 db1 :w2 dW2 :b2 db2}}}))

;; ---------------------------------------------------------------------------
;; BPTT through the gated looped update
;; ---------------------------------------------------------------------------

(defn gadd
  "Deep-add two gradient trees with tensor leaves."
  [a b]
  (if (map? a) (merge-with gadd a b) (t/add a b)))

(defn recur-forward
  "H_{t+1} = ā⊙H_t + B̄⊙E + (block(H_t+E) − (H_t+E)), caching each step."
  [params E H0 a-bar b-bar T]
  (let [be (when b-bar (t/mul E b-bar))]
    (loop [H H0, caches [], i 0]
      (if (= i T)
        {:HT H :caches caches}
        (let [z (t/add H E)
              bf (block-forward params z)
              inc* (t/sub (:Y bf) z)
              carry (t/mul H a-bar)
              Hn (if be (t/add (t/add carry be) inc*) (t/add carry inc*))]
          (recur Hn (conj caches {:H H :cache (:cache bf)}) (inc i)))))))

(defn recur-backward
  "BPTT. Returns {:grads (summed block grads) :dH0 :dA-bar :dB-bar :dE}."
  [params E a-bar b-bar dHT caches]
  (let [d (last (t/shape E))]
    (loop [dH dHT, grads nil
           dA (t/zeros-like a-bar), dB (t/zeros-like a-bar), dE (t/zeros-like E)
           cs (rseq caches)]
      (if-let [{:keys [H cache]} (first cs)]
        (let [bb (block-backward params dH cache)
              dz (t/sub (:dX bb) dH)
              dHt (t/add (t/mul dH a-bar) dz)
              dA-c (t/sum (t/mul H dH) 0)
              dB-c (t/sum (t/mul E dH) 0)
              dE-step (if b-bar (t/add dz (t/mul dH b-bar)) dz)]
          (recur dHt
                 (if grads (gadd grads (:grads bb)) (:grads bb))
                 (t/add dA dA-c) (t/add dB dB-c) (t/add dE dE-step)
                 (next cs)))
        {:grads grads :dH0 dH :dA-bar dA :dB-bar dB :dE dE}))))

;; ---------------------------------------------------------------------------
;; Ablated recurrence: block reads only E ⇒ H_{t+1} = ā⊙H_t + (block(E) − E)
;; ---------------------------------------------------------------------------

(defn recur-forward-ablated [params E H0 a-bar T]
  (let [bf (block-forward params E)
        c  (t/sub (:Y bf) E)]
    (loop [states [H0], i 0]
      (if (= i T)
        {:HT (peek states) :states states :c c :block-cache (:cache bf)}
        (recur (conj states (t/add (t/mul (peek states) a-bar) c)) (inc i))))))

(defn recur-backward-ablated [params E a-bar dHT states block-cache]
  (let [T (dec (count states))
        r (loop [dH dHT, dc (t/zeros-like E), dA (t/zeros-like a-bar), ts (reverse (range T))]
            (if-let [t* (first ts)]
              (recur (t/mul dH a-bar)
                     (t/add dc dH)
                     (t/add dA (t/sum (t/mul (nth states t*) dH) 0))
                     (next ts))
              {:dH dH :dc dc :dA dA}))
        bb (block-backward params (:dc r) block-cache)]
    {:grads (:grads bb) :dH0 (:dH r) :dA-bar (:dA r)
     :dE (t/sub (:dX bb) (:dc r))}))
