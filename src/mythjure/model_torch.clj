(ns mythjure.model-torch
  "Torch backend for the full LM (mythjure.model): Prelude → looped block →
  Coda → weighted cross-entropy, forward AND manual backward — no autograd.
  Same config/param structure as the oracle: params come from converting
  mythjure.model/init output via params->torch, so both backends share one
  init and gradients compare leaf-for-leaf.

  Supports all oracle modes: :parcae, :parcae + :faithful? (learnable Δ,
  B̄⊙e injection, prelude LN), :free, and :ablate?."
  (:require [clojure.walk :as walk]
            [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]
            [mythjure.torch.nn :as nn]
            [mythjure.backprop-torch :as bpt]))

;; ---------------------------------------------------------------------------
;; Param conversion (oracle init → tensor leaves)
;; ---------------------------------------------------------------------------

(defn- numvec? [x] (and (vector? x) (number? (first x))))
(defn- nummat? [x] (and (vector? x) (vector? (first x)) (number? (ffirst x))))

(defn params->torch
  "Convert every numeric vector/matrix leaf of an oracle param/model map to a
  tensor; scalars, keywords, nils (:window, :n-heads, config…) pass through.
  Top-down walk so matrices convert whole, not row-by-row."
  [params & {:keys [dtype] :or {dtype :float64}}]
  (walk/prewalk
   (fn [x]
     (if (and (not (map-entry? x)) (or (nummat? x) (numvec? x)))
       (t/from-clj x :dtype dtype)
       x))
   params))

;; ---------------------------------------------------------------------------
;; Carry parameterization (torch mirrors of model/a-bar etc.)
;; ---------------------------------------------------------------------------

(defn a-bar
  "ā = exp(−exp(log-A)) per channel."
  [log-A]
  (t/exp (t/neg (t/exp log-A))))

(defn a-bar-parcae
  "Ā = exp(−Δ⊙exp(log-A)), Δ = softplus(log-dt)."
  [log-A delta]
  (t/exp (t/neg (t/mul delta (t/exp log-A)))))

(defn b-bar [B delta] (t/mul B delta))

;; ---------------------------------------------------------------------------
;; Prelude
;; ---------------------------------------------------------------------------

(defn embed
  "E = wte[token_i] + wpe[i] per position. inputs: Clojure vector of ids."
  [{:keys [wte wpe]} inputs]
  (let [S (count inputs)
        idx (t/from-clj inputs :dtype :int64)]
    (t/add (t/index-select wte 0 idx)
           (t/narrow wpe 0 0 S))))

;; ---------------------------------------------------------------------------
;; Forward (caching everything backward needs, like the oracle)
;; ---------------------------------------------------------------------------

(defn forward
  "Returns {:loss <0-dim tensor> :cache}. Same loss as mythjure.model/forward:
  weighted mean of per-position −log p[target]."
  ([m inputs targets] (forward m inputs targets (vec (repeat (count inputs) 1.0))))
  ([{:keys [config params]} inputs targets weights]
   (let [{:keys [d-model T ablate? carry-mode faithful?]} config
         {:keys [block log-A ln-f wu ln-p B log-dt]} params
         delta (when faithful? (nn/softplus log-dt))
         ab (cond (= carry-mode :free) log-A
                  faithful? (a-bar-parcae log-A delta)
                  :else (a-bar log-A))
         bb (when faithful? (b-bar B delta))
         E0 (embed params inputs)
         E  (if faithful?
              (nn/layer-norm E0 [d-model] :weight (:gamma ln-p) :bias (:beta ln-p))
              E0)
         H0 (t/zeros-like E)
         rf (if ablate?
              (bpt/recur-forward-ablated block E H0 ab T)
              (bpt/recur-forward block E H0 ab bb T))
         HT (:HT rf)
         nf (nn/layer-norm HT [d-model] :weight (:gamma ln-f) :bias (:beta ln-f))
         logits (t/matmul nf wu)
         lsm (nn/log-softmax logits)
         picked (t/squeeze (t/gather lsm 1 (t/unsqueeze (t/from-clj targets :dtype :int64) 1)) 1)
         w (t/mul (t/from-clj weights :dtype :float64) (t/ones-like picked))
         wsum (max 1e-12 (reduce + weights))
         loss (t/div (t/neg (t/sum (t/mul w picked))) wsum)]
     {:loss loss
      :cache {:inputs inputs :targets targets :weights weights :wsum wsum
              :E0 E0 :E E :ab ab :bb bb :delta delta :recur rf :HT HT
              :nf nf :logits logits}})))

;; ---------------------------------------------------------------------------
;; Backward (mirrors mythjure.model/backward)
;; ---------------------------------------------------------------------------

(defn backward
  "Returns {:grads <same tree as params> :dA-bar} — every leaf a tensor."
  [{:keys [config params]}
   {:keys [inputs targets weights wsum E0 E ab bb recur HT nf logits]}]
  (let [{:keys [d-model ablate? carry-mode faithful?]} config
        {:keys [block log-A log-dt B ln-f ln-p wu wte wpe]} params
        S (count inputs)
        ;; cross-entropy backward: dlogits = (w/wsum)·(softmax − onehot)
        probs (nn/softmax logits)
        tgt (t/unsqueeze (t/from-clj targets :dtype :int64) 1)
        onehot (t/scatter (t/zeros-like logits) 1 tgt 1.0)
        sw (t/unsqueeze (t/div (t/from-clj weights :dtype :float64) wsum) 1)
        dlogits (t/mul (t/sub probs onehot) sw)
        ;; Coda backward
        dWu (t/matmul (t/transpose nf) dlogits)
        dnf (t/matmul dlogits (t/transpose wu))
        lf  (bpt/ln-backward dnf HT (:gamma ln-f) 1e-5)
        dHT (:dX lf)
        ;; BPTT
        rb (if ablate?
             (bpt/recur-backward-ablated block E ab dHT (:states recur) (:block-cache recur))
             (bpt/recur-backward block E ab bb dHT (:caches recur)))
        {bg :grads dabar :dA-bar dbbar :dB-bar dE :dE} rb
        ;; carry chain rule: :free ⇒ dā directly; else dlog-A = dā·ā·ln ā
        dlogA (if (= carry-mode :free)
                dabar
                (t/mul (t/mul dabar ab) (t/log ab)))
        ;; faithful extras
        a-exp  (when faithful? (t/exp log-A))
        sig    (when faithful? (t/sigmoid log-dt))
        delta  (when faithful? (nn/softplus log-dt))
        dlogdt (when faithful?
                 (t/add (t/mul (t/mul (t/mul dabar ab) (t/neg a-exp)) sig)
                        (t/mul (t/mul dbbar B) sig)))
        dB     (when faithful? (t/mul dbbar delta))
        lp     (when faithful? (bpt/ln-backward dE E0 (:gamma ln-p) 1e-5))
        dEmbed (if faithful? (:dX lp) dE)
        ;; Prelude backward: scatter dEmbed rows into wte/wpe gradients
        idx  (t/from-clj inputs :dtype :int64)
        dWte (t/index-add (t/zeros-like wte) 0 idx dEmbed)
        dWpe (t/index-add (t/zeros-like wpe) 0 (t/arange S :dtype :int64) dEmbed)]
    {:grads (cond-> {:wte dWte :wpe dWpe :block bg :log-A dlogA
                     :ln-f {:gamma (:dgamma lf) :beta (:dbeta lf)} :wu dWu}
              faithful? (assoc :log-dt dlogdt :B dB
                               :ln-p {:gamma (:dgamma lp) :beta (:dbeta lp)}))
     :dA-bar dabar}))
