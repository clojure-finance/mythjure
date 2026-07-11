(ns mythjure.backprop
  "Step 4: manual reverse-mode gradients (VJPs) for the block, and BPTT through
  the looped update — pure clojure.core, no autograd engine. Every gradient here
  is verified against finite differences in the test suite (mythjure.backprop-test
  and mythjure.model-test; all match to <1e-6, typically ~1e-9).

  Reverse-mode in one sentence: each forward op f has a backward (vector-Jacobian
  product) that maps the upstream gradient dL/d(output) to dL/d(inputs) and
  dL/d(params); compose them in reverse order of the forward pass. The VJPs used:

    Y = A·B            ⇒  dA = G·Bᵀ,        dB = Aᵀ·G
    Y = X·W + b        ⇒  dX = G·Wᵀ,        dW = Xᵀ·G,  db = Σrows G
    y = gelu(z)        ⇒  dz = G ⊙ gelu'(z)
    y = softmax(z)     ⇒  dz = s ⊙ (G − ⟨G,s⟩)            (masked entries: s=0 ⇒ 0)
    y = LayerNorm(x)   ⇒  dx = inv/d·(d·dx̂ − Σdx̂ − x̂·Σ(dx̂⊙x̂)),  dx̂ = G⊙γ

  The looped update G(H) = A_bar⊙H + (block(H+E) − (H+E)) shares ONE block across
  T iterations, so each weight's gradient is the SUM of its per-iteration
  contributions — we walk the iterations in reverse, carrying dH, and accumulate."
  (:require [mythjure.linalg :as la]
            [mythjure.block :as blk]))

;; ---------------------------------------------------------------------------
;; Elementwise / per-row VJPs
;; ---------------------------------------------------------------------------

(def ^:private gelu-c (Math/sqrt (/ 2.0 Math/PI)))
(def ^:private gelu-a 0.044715)

(defn gelu'
  "Derivative of the tanh-approximation GELU."
  [x]
  (let [u (* gelu-c (+ x (* gelu-a x x x)))
        t (Math/tanh u)
        u' (* gelu-c (+ 1.0 (* 3.0 gelu-a x x)))]
    (+ (* 0.5 (+ 1.0 t)) (* 0.5 x (- 1.0 (* t t)) u'))))

(defn softmax-row-backward
  "Given upstream g and the softmax output s, return dz = s ⊙ (g − ⟨g,s⟩).
  Masked positions have s=0, so their gradient is automatically 0."
  [g s]
  (let [gs (reduce + (map * g s))]
    (mapv (fn [si gi] (* si (- gi gs))) s g)))

(defn ln-row-backward
  "LayerNorm backward for one feature vector. Returns {:dx :dgamma :dbeta}."
  [g x gamma eps]
  (let [d (count x)
        mu (/ (reduce + x) (double d))
        xc (mapv #(- % mu) x)
        var (/ (reduce + (map #(* % %) xc)) (double d))
        inv (/ 1.0 (Math/sqrt (+ var eps)))
        xhat (mapv #(* % inv) xc)
        dxhat (mapv * g gamma)
        s1 (reduce + dxhat)
        s2 (reduce + (map * dxhat xhat))
        dx (mapv (fn [dxh xh] (* (/ inv d) (- (* d dxh) s1 (* xh s2)))) dxhat xhat)]
    {:dx dx :dgamma (mapv * g xhat) :dbeta g}))

(defn ln-backward
  "Row-wise LayerNorm backward over a matrix. dgamma/dbeta sum across rows."
  [G X gamma eps]
  (let [rows (mapv (fn [g x] (ln-row-backward g x gamma eps)) G X)
        z (la/zeros (count gamma))]
    {:dX (mapv :dx rows)
     :dgamma (reduce #(mapv + %1 (:dgamma %2)) z rows)
     :dbeta  (reduce #(mapv + %1 (:dbeta  %2)) z rows)}))

(defn- col-sums [G] (apply mapv + G))
(defn- concat-heads [head-mats]
  (apply mapv (fn [& rows] (vec (apply concat rows))) head-mats))

;; ---------------------------------------------------------------------------
;; Multi-head causal attention: cached forward + backward
;; ---------------------------------------------------------------------------

(defn attn-forward
  "Forward pass caching everything backward needs. Optional `window`: nil ⇒ full
  causal attention; an integer w ⇒ banded causal (each position sees only the w
  previous positions + itself). Backward is unchanged — the softmax VJP zeroes
  masked entries (s=0) for any mask."
  ([X wq wk wv wo n-heads] (attn-forward X wq wk wv wo n-heads nil))
  ([X wq wk wv wo n-heads window]
   (let [Q (la/matmul X wq)
         K (la/matmul X wk)
         V (la/matmul X wv)
         Qh (blk/head-slices Q n-heads)
         Kh (blk/head-slices K n-heads)
         Vh (blk/head-slices V n-heads)
         per (mapv (fn [q k v]
                     (let [scale (/ 1.0 (Math/sqrt (count (first q))))
                           scores (la/scale-mat scale (la/matmul q (la/transpose k)))
                           masked (if window
                                    (la/apply-windowed-mask scores window)
                                    (la/apply-causal-mask scores))
                           w (la/softmax-rows masked)]
                       {:w w :o (la/matmul w v) :scale scale}))
                   Qh Kh Vh)
         Z (concat-heads (mapv :o per))]
     {:attn (la/matmul Z wo)
      :cache {:X X :Qh Qh :Kh Kh :Vh Vh :per per :Z Z
              :wq wq :wk wk :wv wv :wo wo :nh n-heads}})))

(defn attn-backward
  "Given dL/d(attn), return dX and the four projection-weight gradients."
  [dAttn {:keys [X Qh Kh Vh per Z wq wk wv wo nh]}]
  (let [dZ (la/matmul dAttn (la/transpose wo))
        dWo (la/matmul (la/transpose Z) dAttn)
        dOut-heads (blk/head-slices dZ nh)
        grads (mapv (fn [{:keys [w scale]} dout q k v]
                      (let [dW (la/matmul dout (la/transpose v))      ; into weights
                            dV (la/matmul (la/transpose w) dout)
                            dScores (mapv softmax-row-backward dW w)  ; mask handled by s=0
                            dRaw (la/scale-mat scale dScores)]
                        {:dQ (la/matmul dRaw k)
                         :dK (la/matmul (la/transpose dRaw) q)
                         :dV dV}))
                    per dOut-heads Qh Kh Vh)
        dQ (concat-heads (mapv :dQ grads))
        dK (concat-heads (mapv :dK grads))
        dV (concat-heads (mapv :dV grads))]
    {:dX (reduce la/mat+ [(la/matmul dQ (la/transpose wq))
                          (la/matmul dK (la/transpose wk))
                          (la/matmul dV (la/transpose wv))])
     :dwq (la/matmul (la/transpose X) dQ)
     :dwk (la/matmul (la/transpose X) dK)
     :dwv (la/matmul (la/transpose X) dV)
     :dwo dWo}))

;; ---------------------------------------------------------------------------
;; Pre-LN transformer block: cached forward + backward
;; ---------------------------------------------------------------------------

(def ^:private eps 1e-5)

(defn block-forward
  "Block forward (identical math to mythjure.block/forward) but caching
  intermediates for backward. Returns {:Y :cache}."
  [{:keys [ln1 ln2 attn mlp n-heads window]} X]
  (let [n1 (la/layernorm X (:gamma ln1) (:beta ln1))
        {att :attn ac :cache} (attn-forward n1 (:wq attn) (:wk attn) (:wv attn) (:wo attn) n-heads window)
        a  (la/mat+ X att)
        n2 (la/layernorm a (:gamma ln2) (:beta ln2))
        z1 (la/linear n2 (:w1 mlp) (:b1 mlp))
        h2 (la/gelu-mat z1)
        m  (la/linear h2 (:w2 mlp) (:b2 mlp))]
    {:Y (la/mat+ a m) :cache {:X X :n1 n1 :ac ac :a a :n2 n2 :z1 z1 :h2 h2}}))

(defn block-backward
  "Given dL/dY, return {:dX dX :grads <same shape as params>}."
  [{:keys [ln1 ln2 attn mlp]} dY {:keys [X n1 ac a n2 z1 h2]}]
  (let [dm dY
        ;; MLP backward
        dW2 (la/matmul (la/transpose h2) dm)
        db2 (col-sums dm)
        dh2 (la/matmul dm (la/transpose (:w2 mlp)))
        dz1 (mapv (fn [zr gr] (mapv (fn [z g] (* g (gelu' z))) zr gr)) z1 dh2)
        dW1 (la/matmul (la/transpose n2) dz1)
        db1 (col-sums dz1)
        dn2 (la/matmul dz1 (la/transpose (:w1 mlp)))
        ;; LN2 backward (input a); a also receives the residual gradient dY
        l2 (ln-backward dn2 a (:gamma ln2) eps)
        da (la/mat+ dY (:dX l2))
        ;; attention backward (input n1); X also receives residual gradient da
        ab (attn-backward da ac)
        l1 (ln-backward (:dX ab) X (:gamma ln1) eps)
        dX (la/mat+ da (:dX l1))]
    {:dX dX
     :grads {:ln1 {:gamma (:dgamma l1) :beta (:dbeta l1)}
             :ln2 {:gamma (:dgamma l2) :beta (:dbeta l2)}
             :attn {:wq (:dwq ab) :wk (:dwk ab) :wv (:dwv ab) :wo (:dwo ab)}
             :mlp {:w1 dW1 :b1 db1 :w2 dW2 :b2 db2}}}))

;; ---------------------------------------------------------------------------
;; BPTT through the gated looped update
;; ---------------------------------------------------------------------------

(defn gadd
  "Add two gradient maps of identical structure (matrices add row-wise)."
  [a b]
  (cond (map? a) (merge-with gadd a b)
        (sequential? (first a)) (mapv #(mapv + %1 %2) a b)
        :else (mapv + a b)))

(defn- zeros-like [M] (mapv #(la/zeros (count %)) M))

(defn recur-forward
  "Looped forward (Parcae Eq. 3, addition injection):
     H_{t+1} = A_bar⊙H_t + B_bar⊙E + (block(H_t+E) − (H_t+E)),
  caching each step. `b-bar` (per-channel) is the discretized input injection;
  nil ⇒ B_bar=0 (the earlier simplified recurrence). Returns {:HT :caches}."
  [params E H0 a-bar b-bar T]
  (let [be (when b-bar (mapv #(la/hadamard % b-bar) E))
        r (reduce (fn [{:keys [H caches]} _]
                    (let [z (la/mat+ H E)
                          bf (block-forward params z)
                          carry (mapv #(la/hadamard % a-bar) H)
                          inc (la/mat- (:Y bf) z)
                          Hn (if be (la/mat+ (la/mat+ carry be) inc) (la/mat+ carry inc))]
                      {:H Hn :caches (conj caches {:H H :cache (:cache bf)})}))
                  {:H H0 :caches []} (range T))]
    {:HT (:H r) :caches (:caches r)}))

(defn recur-backward
  "BPTT for the Eq. 3 recurrence. Returns SUMMED block grads + {:dH0 :dA-bar
  :dB-bar :dE}. `b-bar` nil ⇒ no input-injection path (dB-bar still computed but
  unused by the caller)."
  [params E a-bar b-bar dHT caches]
  (let [r (reduce (fn [{:keys [dH grads dA dB dE]} {:keys [H cache]}]
                    (let [bb (block-backward params dH cache)
                          dz (la/mat- (:dX bb) dH)                        ; block path − z path
                          dHt (la/mat+ (mapv #(la/hadamard % a-bar) dH) dz) ; carry + z→H
                          dA-c (apply mapv + (mapv (fn [hr dr] (mapv * hr dr)) H dH)) ; Σ_pos H⊙dH
                          dB-c (apply mapv + (mapv (fn [er dr] (mapv * er dr)) E dH)) ; Σ_pos E⊙dH
                          dE-step (if b-bar (la/mat+ dz (mapv #(la/hadamard % b-bar) dH)) dz)]
                      {:dH dHt
                       :grads (if grads (gadd grads (:grads bb)) (:grads bb))
                       :dA (mapv + dA dA-c)
                       :dB (mapv + dB dB-c)
                       :dE (la/mat+ dE dE-step)}))
                  {:dH dHT :grads nil :dA (la/zeros (count a-bar))
                   :dB (la/zeros (count a-bar)) :dE (zeros-like E)}
                  (reverse caches))]
    {:grads (:grads r) :dH0 (:dH r) :dA-bar (:dA r) :dB-bar (:dB r) :dE (:dE r)}))

;; ---------------------------------------------------------------------------
;; ABLATION: block reads only E (not H_t+E). The nonlinear cross-loop path is
;; cut — H_t reaches H_{t+1} ONLY through the linear carry ā⊙H_t. Since block(E)
;; is constant across loops, the recurrence is H_{t+1} = ā⊙H_t + c, c=block(E)−E.
;; ---------------------------------------------------------------------------

(defn recur-forward-ablated [params E H0 a-bar T]
  (let [bf (block-forward params E)
        c  (la/mat- (:Y bf) E)
        states (reduce (fn [acc _]
                         (conj acc (la/mat+ (mapv #(la/hadamard % a-bar) (peek acc)) c)))
                       [H0] (range T))]
    {:HT (peek states) :states states :c c :block-cache (:cache bf)}))

(defn recur-backward-ablated [params E a-bar dHT states block-cache]
  (let [T (dec (count states))
        r (reduce (fn [{:keys [dH dc dA]} t]
                    (let [Ht (nth states t)]            ; dH here = dL/dH_{t+1}
                      {:dc (la/mat+ dc dH)
                       :dA (mapv + dA (apply mapv + (mapv (fn [hr dr] (mapv * hr dr)) Ht dH)))
                       :dH (mapv #(la/hadamard % a-bar) dH)}))   ; carry back ⇒ dL/dH_t
                  {:dH dHT :dc (zeros-like E) :dA (la/zeros (count a-bar))}
                  (reverse (range T)))
        bb (block-backward params (:dc r) block-cache)]   ; dc → block(E) backward
    {:grads (:grads bb) :dH0 (:dH r) :dA-bar (:dA r)
     :dE (la/mat- (:dX bb) (:dc r))}))                    ; c=block(E)−E ⇒ dE = dX_block − dc

;; ---------------------------------------------------------------------------
;; Finite-difference gradient checker (test utility)
;; ---------------------------------------------------------------------------

(defn fd-grad
  "Central-difference gradient of scalar f w.r.t. matrix X."
  [f X & {:keys [h] :or {h 1e-6}}]
  (vec (for [i (range (count X))]
         (vec (for [j (range (count (first X)))]
                (let [Xp (assoc-in X [i j] (+ (get-in X [i j]) h))
                      Xm (assoc-in X [i j] (- (get-in X [i j]) h))]
                  (/ (- (f Xp) (f Xm)) (* 2.0 h))))))))

(defn max-abs-diff [A B]
  (apply max (map (fn [ra rb] (apply max (map (fn [a b] (Math/abs (- a b))) ra rb))) A B)))

(defn frob-dot
  "Frobenius inner product ⟨P,Q⟩ = Σ P_ij Q_ij. Used as a generic scalar loss
  L = ⟨Y,R⟩ so dL/dY = R (an arbitrary downstream gradient) in gradient checks."
  [P Q]
  (reduce + (map (fn [rp rq] (reduce + (map * rp rq))) P Q)))

(comment
  ;; see `mythjure.backprop-test` for the full gradient checks
  (require '[mythjure.linalg :as la])
  (def p (blk/init-params {:d-model 8 :n-heads 2 :d-ff 16 :seed 5}))
  (def E (la/rand-matrix 40 4 8 1.0))
  (def H0 (la/rand-matrix 41 4 8 0.5))
  (def Ab (mapv #(+ 0.5 (* 0.4 (/ % 7.0))) (range 8)))
  (def R (la/rand-matrix 42 4 8 1.0))
  (let [{:keys [HT caches]} (recur-forward p E H0 Ab nil 3)]   ; nil ⇒ no B̄ injection
    (recur-backward p E Ab nil R caches)))
