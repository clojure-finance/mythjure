(ns mythjure.torch.looped
  "Autograd-native looped ENCODER: a continuous-input sequence (S×d-in matrix)
  through the paper's weight-tied looped block to a pooled embedding vector —
  the general-purpose entry point for training looped models on non-token data
  (e.g. a learned distance metric over return windows).

  Positioning (direction doc §4): the manual-VJP path (model-torch/backward +
  train-torch) stays paper-only and oracle-pinned; this namespace REUSES the
  looped core unchanged — backprop-torch/recur-forward is façade ops, so it is
  differentiable as-is under torch autograd — and swaps only the boundary
  pieces: a learned input projection instead of the token prelude, a pooled
  readout head instead of the LM coda, and autograd/backward! + optim/adam
  instead of manual BPTT. The loss stays caller-side: `forward` hands back a
  differentiable embedding vector.

  Model map: {:config … :params …} like the paper model; params are a nested
  Clojure map of tensor leaves already flagged requires-grad. All paper carry
  modes are supported (:parcae default, :parcae+:faithful?, :free), and the
  carry diagnostics stay first-class: train-torch/effective-abar and
  channel-convergence work unchanged on this model/cache, and `diagnostics`
  reproduces the full train-torch report with dL/dā recovered from the log-A
  leaf gradient (`abar-grad`) — there is no manual BPTT here to hand it over."
  (:require [mythjure.linalg :as la]
            [mythjure.block :as blk]
            [mythjure.backprop-torch :as bpt]
            [mythjure.model-torch :as mt]
            [mythjure.train-torch :as tt]
            [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]
            [mythjure.torch.nn :as nn]
            [mythjure.torch.autograd :as ag]
            [mythjure.torch.optim :as optim]))

;; ---------------------------------------------------------------------------
;; Initialization
;; ---------------------------------------------------------------------------

(defn init
  "Initialize a looped encoder. Config keys (defaults):
    :d-in        REQUIRED — input feature dimension (columns of X)
    :d-embed 16  — embedding dimension of the head
    :pool :last  — sequence→vector readout, :last or :mean
    :positional? true — learned wpe added to the projection, narrowed to S
                 (so any S ≤ :seq-len works — variable-length windows)
    :d-model 48 :n-heads 4 :d-ff (4×d-model) :seq-len 64 :T 8 :window nil
    :seed 0 :log-A0 0.5 :carry-mode :parcae :faithful? nil :dtype :float64
  Params are the paper-style init (mythjure.block/init-params + seeded
  rand-matrix) converted to tensors and flagged requires-grad — deterministic
  from :seed alone, no torch RNG involved. The input projection scales
  1/√d-in, the head 1/√d-model."
  [{:keys [d-in d-model n-heads d-ff seq-len T seed log-A0 window carry-mode
           faithful? d-embed pool positional? dtype]
    :or {d-model 48 n-heads 4 seq-len 64 T 8 seed 0 log-A0 0.5
         carry-mode :parcae d-embed 16 pool :last positional? true
         dtype :float64}}]
  (assert (pos-int? d-in) "init: :d-in (input feature dimension) is required")
  (assert (contains? #{:last :mean} pool) "init: :pool must be :last or :mean")
  (assert (not (and (= carry-mode :free) faithful?))
          ":free carry (unconstrained) is incompatible with :faithful? (Δ-reparameterized carry)")
  (let [d-ff (or d-ff (* 4 d-model))
        ;; log-dt₀ = softplus⁻¹(1) ⇒ Δ≈1 (Parcae 'addition' start point, as in model/init)
        log-dt0 (Math/log (- (Math/exp 1.0) 1.0))
        params (cond-> {:w-in  (la/rand-matrix (+ seed 1) d-in d-model
                                               (/ 1.0 (Math/sqrt d-in)))
                        :block (blk/init-params {:d-model d-model :n-heads n-heads
                                                 :d-ff d-ff :seed (+ seed 100)
                                                 :window window})
                        :log-A (vec (repeat d-model log-A0))
                        :ln-f  {:gamma (la/ones d-model) :beta (la/zeros d-model)}
                        :w-out (la/rand-matrix (+ seed 3) d-model d-embed
                                               (/ 1.0 (Math/sqrt d-model)))
                        :b-out (la/zeros d-embed)}
                 positional? (assoc :wpe (la/rand-matrix (+ seed 2) seq-len d-model 0.02))
                 faithful?   (assoc :log-dt (vec (repeat d-model log-dt0))
                                    :B      (la/ones d-model)
                                    :ln-p   {:gamma (la/ones d-model)
                                             :beta  (la/zeros d-model)}))]
    {:config {:d-in d-in :d-model d-model :n-heads n-heads :seq-len seq-len
              :T T :carry-mode carry-mode :faithful? faithful? :d-embed d-embed
              :pool pool :positional? positional? :dtype dtype}
     :params (ag/requires-grad-params! (mt/params->torch params :dtype dtype))}))

;; ---------------------------------------------------------------------------
;; Forward (continuous prelude → looped core → pooled head)
;; ---------------------------------------------------------------------------

(defn forward
  "Encode one sequence. X: S×d-in — a torch tensor or nested Clojure vectors
  (converted at the model's :dtype); any S ≤ :seq-len when :positional?.
  Prelude E = X·w-in (+ wpe rows 0..S-1); looped core exactly as the paper
  model (H_{t+1} = ā⊙H_t [+ B̄⊙E] + block(H_t+E) − (H_t+E)); head = pool
  (:last row or :mean over positions) → final LayerNorm → w-out·+b-out.
  Returns {:embedding <d-embed tensor, differentiable> :cache …}; the cache
  feeds train-torch/channel-convergence and `diagnostics` unchanged."
  [{:keys [config params]} X]
  (let [{:keys [d-in d-model T seq-len carry-mode faithful? pool positional? dtype]} config
        {:keys [w-in wpe block log-A ln-f w-out b-out ln-p B log-dt]} params
        X (if (core/tensor? X) X (t/from-clj X :dtype dtype))
        [S d-in-x] (t/shape X)]
    (when (not= d-in-x d-in)
      (throw (ex-info "forward: X's feature dimension ≠ config :d-in"
                      {:d-in d-in :x-shape (t/shape X)})))
    (when (and positional? (> S seq-len))
      (throw (ex-info "forward: S exceeds :seq-len (the positional table)"
                      {:S S :seq-len seq-len})))
    (let [delta (when faithful? (nn/softplus log-dt))
          ab (cond (= carry-mode :free) log-A
                   faithful? (mt/a-bar-parcae log-A delta)
                   :else (mt/a-bar log-A))
          bb (when faithful? (mt/b-bar B delta))
          E0 (cond-> (t/matmul X w-in)
               wpe (t/add (t/narrow wpe 0 0 S)))
          E  (if faithful?
               (nn/layer-norm E0 [d-model] :weight (:gamma ln-p) :bias (:beta ln-p))
               E0)
          H0 (t/zeros-like E)
          rf (bpt/recur-forward block E H0 ab bb T)
          HT (:HT rf)
          pooled (case pool
                   :last (t/squeeze (t/narrow HT 0 (dec S) 1) 0)
                   :mean (t/squeeze (t/mean-keep HT 0) 0))
          nf (nn/layer-norm pooled [d-model] :weight (:gamma ln-f) :bias (:beta ln-f))]
      {:embedding (t/add (t/matmul nf w-out) b-out)
       :cache {:X X :E0 E0 :E E :ab ab :bb bb :recur rf :HT HT
               :pooled pooled :nf nf}})))

;; ---------------------------------------------------------------------------
;; Checkpoint restore (module/save! serializes tensor leaves only)
;; ---------------------------------------------------------------------------

(defn restore
  "Merge a loaded checkpoint (module/load-params of a module/save! of this
  model's :params) into a model initialized with the SAME config: tensor
  leaves come from the checkpoint, the non-tensor block config (:n-heads
  :d-model :window — dropped by save!, which serializes tensors only) from
  the init. Loaded tensors are NOT flagged requires-grad — right for
  strategy-runtime inference; call autograd/requires-grad-params! on
  (:params result) to resume training."
  [m loaded]
  (letfn [(deep-merge [a b]
            (if (and (map? a) (map? b)) (merge-with deep-merge a b) b))]
    (update m :params deep-merge loaded)))

;; ---------------------------------------------------------------------------
;; Carry diagnostics on the autograd path
;; ---------------------------------------------------------------------------

(defn abar-grad
  "dL/dā recovered from the log-A leaf's accumulated .grad — read it between
  autograd/backward! and the optimizer step (nil before backward, or after
  grads are cleared). Exact in every carry mode: log-A reaches the loss only
  through ā, and ∂ā/∂log-A = ā·ln ā in both parcae parameterizations (so
  dL/dā = dL/dlog-A ÷ ā·ln ā); :free carry reads the leaf grad directly."
  [{:keys [config params]}]
  (when-let [g (ag/grad (:log-A params))]
    (if (= (:carry-mode config) :free)
      g
      (ag/no-grad
       (let [{:keys [log-A log-dt]} params
             ab (if (:faithful? config)
                  (mt/a-bar-parcae log-A (nn/softplus log-dt))
                  (mt/a-bar log-A))]
         (t/div g (t/mul ab (t/log ab))))))))

(defn diagnostics
  "The train-torch carry-stability report — ā stats, dL/dā sign counts,
  per-channel convergence, worst channel — for an autograd-trained looped
  encoder: delegates to train-torch/diagnostics with dL/dā from `abar-grad`.
  Call between backward! and the optimizer step; `cache` is any forward
  cache (a no-grad forward on a representative input is fine)."
  [m cache]
  (let [da (abar-grad m)]
    (when-not da
      (throw (ex-info "diagnostics: no gradient on log-A — call after autograd/backward!, before grads are cleared" {})))
    (tt/diagnostics m cache da)))

;; ---------------------------------------------------------------------------
;; Training loop (autograd + torch Adam; loss caller-side)
;; ---------------------------------------------------------------------------

(defn train!
  "Adam over the model's params with a caller-supplied loss.
  loss-fn: model → 0-dim loss tensor, built from `forward` embeddings over
  the caller's data (metric/contrastive, MSE, …). Param tensors are stepped
  IN PLACE (torch Adam); returns {:model m :history […]}.

  Options: :steps 200, :lr 1e-3, :log-every 20, :verbose? true,
  :clip nil (max grad norm, applied after the logged diagnostics),
  :diag-input nil — a sample X; when given, every logged step runs a no-grad
  forward on it and records the full carry-diagnostics report in the history
  (monitoring carry stability during training is the point of the looped
  architecture — watch ā and convergence, not just the loss)."
  [m loss-fn & {:keys [steps lr log-every verbose? clip diag-input]
                :or {steps 200 lr 1e-3 log-every 20 verbose? true}}]
  (let [opt (optim/adam (:params m) :lr lr)]
    (loop [step 1, hist []]
      (if (> step steps)
        {:model m :history hist}
        (let [_ (optim/zero-grad! opt)
              loss (loss-fn m)
              _ (ag/backward! loss)
              hist (if (or (= step 1) (zero? (mod step log-every)))
                     (let [l (t/item loss)
                           diag (when diag-input
                                  (diagnostics m (:cache (ag/no-grad (forward m diag-input)))))]
                       (when verbose?
                         (if diag
                           (println (format "step %4d  loss %.4f  ā[min %.3f mean %.3f max %.3f]  dL/dā[+%d/-%d]  conv[mean %.2f max %.2f]"
                                            step l
                                            (get-in diag [:a-bar :min]) (get-in diag [:a-bar :mean]) (get-in diag [:a-bar :max])
                                            (get-in diag [:dL/dā :n-pos]) (get-in diag [:dL/dā :n-neg])
                                            (get-in diag [:converge :mean]) (get-in diag [:converge :max])))
                           (println (format "step %4d  loss %.4f" step l))))
                       (conj hist (assoc (or diag {}) :step step :loss l)))
                     hist)
              _ (when clip (optim/clip-grad-norm! (:params m) clip))
              _ (optim/step! opt)]
          (recur (inc step) hist))))))
