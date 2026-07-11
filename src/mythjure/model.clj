(ns mythjure.model
  "The full trainable looped language model: Prelude (token+positional embedding)
  → recurrent block looped T times with input injection → Coda (final LayerNorm +
  unembed) → cross-entropy. Forward and the complete backward (cross-entropy →
  Coda → BPTT → Prelude), all on top of mythjure.backprop.

  Three carry modes (config):
   - :parcae (default) — ā = exp(−exp(log-A)) ∈ (0,1)^d, the stable Parcae
     parameterization; the optimizer moves log-A, dL/d(log-A) = dā·ā·ln ā.
   - :parcae with :faithful? — the full Parcae Eq. 3: learnable per-channel
     Δ=softplus(log-dt) with ā=exp(−Δ⊙exp(log-A)), a B̄=Δ⊙B input-injection term
     (H_{t+1}=ā⊙H_t + B̄⊙e + block-increment), and a prelude LayerNorm on e.
   - :free — ā is a raw unconstrained per-channel vector (ρ(Ā)=max|ā| can be ≥1);
     used to reproduce the paper's residual-explosion instability.

  Watching log-A (via ā) and its gradient is the whole point of the carry
  diagnostics."
  (:require [mythjure.linalg :as la]
            [mythjure.block :as blk]
            [mythjure.backprop :as bp]))

;; ---------------------------------------------------------------------------
;; Carry parameterization
;; ---------------------------------------------------------------------------

(defn a-bar
  "ā = exp(−exp(log-A)), per channel — guaranteed in (0,1)."
  [log-A]
  (mapv #(Math/exp (- (Math/exp %))) log-A))

(defn dlogA<-dabar
  "Chain rule dā→d(log-A):  dā/d(log-A) = ā·ln(ā). Holds with or without a
  learnable Δ, since Δ is absorbed into ā (∂ā/∂log-A = ā·ln ā either way)."
  [dabar ab]
  (mapv (fn [da a] (* da a (Math/log a))) dabar ab))

;; ---------------------------------------------------------------------------
;; Faithful Parcae extras (behind :faithful?): learnable Δ + B̄ input injection
;; ---------------------------------------------------------------------------

(defn softplus [x] (Math/log (+ 1.0 (Math/exp x))))
(defn sigmoid  [x] (/ 1.0 (+ 1.0 (Math/exp (- x)))))    ; = softplus'

(defn a-bar-parcae
  "Ā = exp(−Δ⊙exp(log-A)), Δ = softplus(log-dt). ∈(0,1) per channel."
  [log-A delta]
  (mapv (fn [la d] (Math/exp (- (* d (Math/exp la))))) log-A delta))

(defn b-bar
  "B̄ = Δ⊙B (Euler discretization of the input-injection matrix, diagonal here)."
  [B delta]
  (mapv * B delta))

(defn effective-abar
  "The ā actually used by a model, respecting :carry-mode and :faithful?. Mirrors
  the carry branch in `forward` — external callers (diagnostics) should use this
  rather than `a-bar` directly, which is only correct for the simple :parcae mode."
  [{:keys [config params]}]
  (let [{:keys [carry-mode faithful?]} config
        {:keys [log-A log-dt]} params]
    (cond (= carry-mode :free) log-A
          faithful? (a-bar-parcae log-A (mapv softplus log-dt))
          :else (a-bar log-A))))

;; ---------------------------------------------------------------------------
;; Initialization
;; ---------------------------------------------------------------------------

(defn init
  [{:keys [vocab-size d-model n-heads d-ff seq-len T seed log-A0 window ablate? carry-mode faithful?]
    :or {d-model 48 n-heads 4 seq-len 64 T 8 seed 0 log-A0 0.5 carry-mode :parcae}}]
  (assert (not (and (= carry-mode :free) faithful?))
          ":free carry (unconstrained) is incompatible with :faithful? (Δ-reparameterized carry)")
  (assert (not (and ablate? faithful?))
          ":ablate? is not supported with :faithful? (the ablated recurrence has no B̄e path)")
  (let [d-ff (or d-ff (* 4 d-model))
        emb-scale 0.02
        ;; log-dt₀ = softplus⁻¹(1) ⇒ Δ≈1; B₀=1 ⇒ B̄≈1 (Parcae 'addition' start point)
        log-dt0 (Math/log (- (Math/exp 1.0) 1.0))]
    {:config {:vocab-size vocab-size :d-model d-model :n-heads n-heads
              :seq-len seq-len :T T :ablate? ablate? :carry-mode carry-mode :faithful? faithful?}
     :params (cond-> {:wte   (la/rand-matrix (+ seed 1) vocab-size d-model emb-scale)
                      :wpe   (la/rand-matrix (+ seed 2) seq-len d-model emb-scale)
                      :block (blk/init-params {:d-model d-model :n-heads n-heads :d-ff d-ff :seed (+ seed 100) :window window})
                      ;; log-A0 ≈ 0.5 ⇒ ā ≈ exp(−1.65) ≈ 0.19 (short memory at init, well inside stable regime)
                      :log-A (vec (repeat d-model log-A0))
                      :ln-f  {:gamma (la/ones d-model) :beta (la/zeros d-model)}
                      :wu    (la/rand-matrix (+ seed 3) d-model vocab-size emb-scale)}
               faithful? (assoc :log-dt (vec (repeat d-model log-dt0))       ; learnable Δ
                                :B      (la/ones d-model)                     ; input injection
                                :ln-p   {:gamma (la/ones d-model) :beta (la/zeros d-model)}))})) ; prelude norm

;; ---------------------------------------------------------------------------
;; Prelude
;; ---------------------------------------------------------------------------

(defn embed
  "E = token-embedding[input_i] + positional-embedding[i], per position."
  [{:keys [wte wpe]} inputs]
  (mapv (fn [i tok] (la/v+ (nth wte tok) (nth wpe i)))
        (range (count inputs)) inputs))

;; ---------------------------------------------------------------------------
;; Forward (one sequence), caching everything backward needs
;; ---------------------------------------------------------------------------

(defn forward
  "Returns {:loss :cache}. inputs/targets are vectors of token ids. Optional
  `weights` is a per-position vector (default all 1s) — set it to weight the loss
  toward specific positions (e.g. only the query position of a copy task)."
  ([m inputs targets] (forward m inputs targets (vec (repeat (count inputs) 1.0))))
  ([{:keys [config params]} inputs targets weights]
   (let [{:keys [d-model T ablate? carry-mode faithful?]} config
         {:keys [block log-A ln-f wu ln-p B log-dt]} params
         delta (when faithful? (mapv softplus log-dt))
         ;; carry: :free ⇒ raw ā (unconstrained); :parcae faithful ⇒ exp(−Δ⊙exp(log-A));
         ;; :parcae simple ⇒ exp(−exp(log-A)) (Δ=1).
         ab (cond (= carry-mode :free) log-A
                  faithful? (a-bar-parcae log-A delta)
                  :else (a-bar log-A))
         bb (when faithful? (b-bar B delta))               ; B̄ input injection (nil ⇒ off)
         E0 (embed params inputs)
         E  (if faithful? (la/layernorm E0 (:gamma ln-p) (:beta ln-p)) E0)   ; prelude norm
         H0 (mapv (fn [_] (la/zeros d-model)) inputs)
         rf (if ablate? (bp/recur-forward-ablated block E H0 ab T)
                (bp/recur-forward block E H0 ab bb T))
         HT (:HT rf)
         nf (la/layernorm HT (:gamma ln-f) (:beta ln-f))
         logits (la/matmul nf wu)
         probs (la/softmax-rows logits)
         wsum (max 1e-12 (reduce + weights))
         loss (/ (reduce + (map (fn [p t w] (* w (- (Math/log (max 1e-12 (nth p t)))))) probs targets weights))
                 wsum)]
     {:loss loss
      :cache {:inputs inputs :targets targets :weights weights :wsum wsum
              :E0 E0 :E E :H0 H0 :ab ab :bb bb :delta delta :recur rf :HT HT :nf nf :probs probs}})))

;; ---------------------------------------------------------------------------
;; Backward
;; ---------------------------------------------------------------------------

(defn- add-to-row [M i v] (update M i #(la/v+ % v)))

(defn backward
  "Returns the gradient tree (same shape as params) plus {:dA-bar ..} for
  diagnostics."
  [{:keys [config params]} {:keys [inputs targets weights wsum E0 E ab bb delta recur HT nf probs]}]
  (let [{:keys [d-model vocab-size seq-len ablate? carry-mode faithful?]} config
        {:keys [block log-A log-dt B ln-f ln-p wu]} params
        ;; cross-entropy backward: dlogits_i = (w_i/wsum)·(softmax_i − onehot(t_i))
        dlogits (mapv (fn [p t w]
                        (let [s (/ w wsum)
                              row (la/scale s p)]
                          (update row t #(- % s))))
                      probs targets weights)
        ;; Coda backward
        dWu (la/matmul (la/transpose nf) dlogits)
        dnf (la/matmul dlogits (la/transpose wu))
        lf  (bp/ln-backward dnf HT (:gamma ln-f) 1e-5)
        dHT (:dX lf)
        ;; BPTT (ablated path: block reads only E, H_t flows only through carry)
        rb (if ablate?
             (bp/recur-backward-ablated block E ab dHT (:states recur) (:block-cache recur))
             (bp/recur-backward block E ab bb dHT (:caches recur)))
        {bg :grads dabar :dA-bar dbbar :dB-bar dE :dE} rb
        ;; carry: :free ⇒ dā directly; else dlog-A = dā·ā·ln ā (Δ-agnostic)
        dlogA (if (= carry-mode :free) dabar (dlogA<-dabar dabar ab))
        ;; faithful extras: log-dt (from Ā and B̄), B, and prelude LN on E
        a-exp   (when faithful? (mapv #(Math/exp %) log-A))         ; a = exp(log-A)
        sig     (when faithful? (mapv sigmoid log-dt))              ; σ(log-dt)=Δ'
        dlogdt  (when faithful?
                  (mapv (fn [dA ab' ac dB' Bc s]
                          (+ (* dA ab' (- ac) s)      ; ∂Ā/∂Δ = ā·(−a); Δ'=σ
                             (* dB' Bc s)))            ; ∂B̄/∂Δ = B; Δ'=σ
                        dabar ab a-exp dbbar B sig))
        dB      (when faithful? (mapv * dbbar delta))               ; B̄=Δ⊙B ⇒ dB=dB̄·Δ
        ;; prelude LN backward: dE flows through LN_p to E0(=embed) + γ_p,β_p
        lp      (when faithful? (bp/ln-backward dE E0 (:gamma ln-p) 1e-5))
        dEmbed  (if faithful? (:dX lp) dE)
        ;; Prelude backward (H0 is constant zeros ⇒ no grad path)
        zwte (vec (repeat vocab-size (la/zeros d-model)))
        zwpe (vec (repeat seq-len (la/zeros d-model)))
        dWte (reduce (fn [M [i tok]] (add-to-row M tok (nth dEmbed i)))
                     zwte (map-indexed vector inputs))
        dWpe (reduce (fn [M i] (add-to-row M i (nth dEmbed i))) zwpe (range (count inputs)))]
    {:grads (cond-> {:wte dWte :wpe dWpe :block bg :log-A dlogA
                     :ln-f {:gamma (:dgamma lf) :beta (:dbeta lf)} :wu dWu}
              faithful? (assoc :log-dt dlogdt :B dB
                               :ln-p {:gamma (:dgamma lp) :beta (:dbeta lp)}))
     :dA-bar dabar}))
