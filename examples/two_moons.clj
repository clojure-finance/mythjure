;; Non-mythjure toy example (direction doc §5.1 item 4 / §4 gate): two-moons
;; binary classification with a small MLP, written the way an OUTSIDE user of
;; the mythjure.torch.* facade would — nothing from mythjure's experiment code
;; (no mythjure.model/backprop/train), only the general-purpose surfaces:
;; tensor ops, autograd, the optimizer plumbing, and nn.Module interop.
;; Trains the same architecture twice: (A) as a Clojure param map + forward
;; fn, (B) as a Python nn.Sequential driven through module/params — the two
;; model idioms the facade offers today.
;; Run:  clojure -M:torch examples/two_moons.clj
;;       (Python w/ torch auto-discovered; MYTHJURE_PYTHON/MYTHJURE_LIBPYTHON override)
(require '[mythjure.torch.core :as tc]
         '[mythjure.torch.tensor :as t]
         '[mythjure.torch.nn :as nn]
         '[mythjure.torch.autograd :as ag]
         '[mythjure.torch.optim :as optim]
         '[mythjure.torch.module :as module])

(tc/initialize!)

;; ---------------------------------------------------------------------------
;; Data: two interleaved half-moons, generated in pure Clojure (seeded).
;; ---------------------------------------------------------------------------

(defn two-moons
  "n points per class with gaussian noise; label 0 = outer moon, 1 = inner.
  Returns {:x [[x y] ...] :y [0 1 ...]} in a fixed deterministic order."
  [n noise seed]
  (let [rng (java.util.Random. (long seed))
        jitter #(+ % (* noise (.nextGaussian rng)))
        pt (fn [label i]
             (let [theta (* Math/PI (/ (double i) (dec n)))]
               (if (zero? label)
                 [[(jitter (Math/cos theta)) (jitter (Math/sin theta))] 0]
                 [[(jitter (- 1.0 (Math/cos theta))) (jitter (- 0.5 (Math/sin theta)))] 1])))
        pts (concat (map #(pt 0 %) (range n)) (map #(pt 1 %) (range n)))]
    {:x (mapv first pts) :y (mapv second pts)}))

(defn split-train-test
  "Every 5th point goes to test (deterministic, balanced along both arcs)."
  [{:keys [x y]}]
  (let [test? (fn [i] (zero? (rem i 5)))
        idx (range (count x))
        pick (fn [pred sel] (mapv sel (filter pred idx)))]
    {:train {:x (pick (complement test?) x) :y (pick (complement test?) y)}
     :test  {:x (pick test? x) :y (pick test? y)}}))

(defn ->tensors [{:keys [x y]}]
  {:x (t/from-clj x :dtype :float32)
   :y (t/from-clj y :dtype :int64)   ; cross-entropy targets must be int64
   :n (count y)})

(def data (split-train-test (two-moons 200 0.15 7)))
(def train (->tensors (:train data)))
(def test* (->tensors (:test data)))

;; ---------------------------------------------------------------------------
;; Shared pieces
;; ---------------------------------------------------------------------------

(t/manual-seed 42)

(defn accuracy
  "Fraction of rows where argmax(logits) equals the int64 target."
  [logits {:keys [y n]}]
  (/ (t/item (t/sum (t/eq (t/argmax logits :dim 1) y))) (double n)))

(defn train-loop!
  "Full-batch Adam; forward-fn: params -> x -> logits. Mutates the param
  tensors in place (torch optimizer under no-grad); returns nil."
  [params forward-fn steps lr]
  (let [opt (optim/adam params :lr lr)]
    (dotimes [step steps]
      (let [loss (nn/cross-entropy (forward-fn params (:x train)) (:y train))]
        (optim/zero-grad! opt)
        (ag/backward! loss)
        (optim/step! opt)
        (when (zero? (rem step 100))
          (println (format "  step %3d  loss %.4f" step (t/item loss))))))))

;; ---------------------------------------------------------------------------
;; Model A: Clojure-native — a param map + a forward function.
;; ---------------------------------------------------------------------------

(defn linear-init
  "He-ish init: W [in out] scaled 1/sqrt(in), zero bias."
  [in out]
  {:w (t/mul (t/randn [in out]) (/ 1.0 (Math/sqrt in)))
   :b (t/zeros [out])})

(defn init-mlp [] {:l1 (linear-init 2 16) :l2 (linear-init 16 16) :l3 (linear-init 16 2)})

(defn dense [x {:keys [w b]}] (t/add (t/matmul x w) b))

(defn mlp-forward [p x]
  (-> x
      (dense (:l1 p)) nn/gelu
      (dense (:l2 p)) nn/gelu
      (dense (:l3 p))))

(println "Model A — Clojure param map + forward fn")
(def params-a (ag/requires-grad-params! (init-mlp)))
(train-loop! params-a mlp-forward 500 1e-2)
(def acc-a {:train (accuracy (mlp-forward params-a (:x train)) train)
            :test  (accuracy (mlp-forward params-a (:x test*)) test*)})
(println (format "  accuracy: train %.3f  test %.3f" (:train acc-a) (:test acc-a)))

;; ---------------------------------------------------------------------------
;; Model B: the same architecture as a Python nn.Sequential, driven through
;; module/params — live weights in a Clojure map, same optimizer + autograd.
;; ---------------------------------------------------------------------------

(println "Model B — Python nn.Sequential via module interop")
(def net (module/sequential [(module/make "Linear" [2 16])
                             (module/make "GELU" [] {:approximate "tanh"})
                             (module/make "Linear" [16 16])
                             (module/make "GELU" [] {:approximate "tanh"})
                             (module/make "Linear" [16 2])]))
(train-loop! (module/params net) (fn [_ x] (module/call net x)) 500 1e-2)
(def acc-b {:train (accuracy (module/call net (:x train)) train)
            :test  (accuracy (module/call net (:x test*)) test*)})
(println (format "  accuracy: train %.3f  test %.3f" (:train acc-b) (:test acc-b)))

;; ---------------------------------------------------------------------------
;; Verify: two-moons at this noise is cleanly separable by a small MLP.
;; ---------------------------------------------------------------------------

(assert (>= (:test acc-a) 0.9) (str "model A test accuracy too low: " acc-a))
(assert (>= (:test acc-b) 0.9) (str "model B test accuracy too low: " acc-b))
(println "OK — both model idioms trained to >=0.9 test accuracy.")
