;; Looped-encoder template: continuous input → looped encoder → embedding →
;; caller-defined loss → train → checkpoint round trip. This is the pattern a
;; downstream project copies to train a learned distance metric on the
;; weight-tied looped block (T-deep with one layer's parameter count, the
;; Parcae carry as built-in regularization for tiny data): windows of
;; return vectors are encoded into an embedding space, a contrastive loss
;; shapes that space, and nearest-neighbor lookup runs on the embeddings.
;; Everything model-side lives in mythjure.torch.looped (autograd-native,
;; reusing the paper's looped core); the LOSS stays in this file — swap in
;; your own metric/contrastive/MSE objective without touching mythjure.
;;
;; Run:  clojure -M:torch examples/looped_encoder.clj
(require '[clojure.java.io :as io]
         '[mythjure.torch.core :as tc]
         '[mythjure.torch.tensor :as t]
         '[mythjure.torch.autograd :as ag]
         '[mythjure.torch.module :as module]
         '[mythjure.torch.looped :as lp])

(tc/initialize!)
(t/manual-seed 42)   ; curated seed — data + init below are Clojure-seeded,
                     ; this pins any torch-side randomness a variant adds

;; ---------------------------------------------------------------------------
;; Data: synthetic "monthly return" windows, 3 assets, two regimes — bull
;; (+1.5% drift) vs bear (−1.5% drift) under 2% noise. Window length k varies
;; 4..6, the variable-S case the encoder must handle (S ≤ seq-len). The task
;; is deliberately two-moons-scale simple: this file is the PLUMBING template
;; (encode → caller loss → train → checkpoint) — a real learned metric brings
;; its own data and objective and copies the structure, not the task.
;; ---------------------------------------------------------------------------

(defn window
  "One k×3 window of return vectors: per-regime drift + gaussian noise."
  [rng drift k]
  (vec (repeatedly k (fn [] (vec (repeatedly 3 #(+ drift (* 0.02 (.nextGaussian rng)))))))))

(defn make-set
  "n windows per regime; label 0 = bull, 1 = bear."
  [n seed]
  (let [rng (java.util.Random. (long seed))
        one (fn [label drift] (doall (repeatedly n #(vector (window rng drift (+ 4 (.nextInt rng 3))) label))))
        pts (vec (concat (one 0 0.015) (one 1 -0.015)))]
    {:x (mapv first pts) :y (mapv second pts)}))

(def train* (make-set 8 1))   ; 16 windows — tiny data is the design point
(def test*  (make-set 8 2))   ; 16 held-out windows

;; ---------------------------------------------------------------------------
;; Model: looped encoder, k×3 → 8-dim embedding.
;; ---------------------------------------------------------------------------

(def config {:d-in 3 :d-model 16 :n-heads 2 :seq-len 8 :T 4 :d-embed 8 :seed 7})
(def model (lp/init config))

(defn embed-all [m xs] (mapv #(:embedding (lp/forward m %)) xs))

;; ---------------------------------------------------------------------------
;; Caller-defined loss: pairwise cosine contrastive over the train set.
;; Same regime → pull cos-sim toward 1; different → push it below 0.
;; ---------------------------------------------------------------------------

(defn cos-sim [a b]
  (t/div (t/sum (t/mul a b))
         (t/add (t/mul (t/sqrt (t/sum (t/mul a a)))
                       (t/sqrt (t/sum (t/mul b b))))
                1e-12)))

(def pairs (let [n (count (:x train*))]
             (vec (for [i (range n), j (range (inc i) n)] [i j]))))

(defn contrastive-loss [m]
  (let [embs (embed-all m (:x train*))
        y    (:y train*)
        term (fn [[i j]]
               (let [c (cos-sim (embs i) (embs j))]
                 (if (= (y i) (y j))
                   (t/sub (t/ones-like c) c)
                   (t/clamp c :min 0.0))))]
    (t/div (reduce t/add (map term pairs)) (double (count pairs)))))

;; ---------------------------------------------------------------------------
;; Train — watching the carry diagnostics (ā, dL/dā, convergence) is the
;; point of the looped architecture, not a nicety: :diag-input logs them.
;; ---------------------------------------------------------------------------

(println "Training the looped encoder (contrastive, 16 windows)")
(def result (lp/train! model contrastive-loss
                       :steps 30 :lr 3e-3 :log-every 5
                       :diag-input (first (:x train*))))
(def history (:history result))

;; ---------------------------------------------------------------------------
;; Evaluate: 1-NN by cosine similarity in EMBEDDING space on held-out windows.
;; Inference runs under no-grad — no graphs get built while evaluating (the
;; pattern to copy at strategy runtime).
;; ---------------------------------------------------------------------------

(defn nn-accuracy [m]
  (let [tr (ag/no-grad (embed-all m (:x train*)))
        te (ag/no-grad (embed-all m (:x test*)))
        hit (fn [e l]
              (let [sims (mapv #(t/item (cos-sim e %)) tr)]
                (= l ((:y train*) (apply max-key sims (range (count sims)))))))]
    (/ (count (filter true? (map hit te (:y test*))))
       (double (count te)))))

(def acc (nn-accuracy model))
(println (format "1-NN accuracy in embedding space: %.3f" acc))

;; ---------------------------------------------------------------------------
;; Checkpoint round trip: save! → load-params → restore into a fresh init of
;; the same config (init seed irrelevant) — the strategy-runtime load path.
;; ---------------------------------------------------------------------------

(def ckpt "data/looped_encoder.pt")
(io/make-parents ckpt)
(module/save! (:params model) ckpt)
(def reloaded (lp/restore (lp/init (assoc config :seed 99))
                          (module/load-params ckpt)))
(def probe (first (:x test*)))
(assert (= (t/to-clj (:embedding (ag/no-grad (lp/forward model probe))))
           (t/to-clj (:embedding (lp/forward reloaded probe))))
        "checkpoint round trip changed the embeddings")

;; ---------------------------------------------------------------------------
;; Verify: training moved the loss, the metric separates the regimes, and
;; the carry stayed in the stable Parcae regime throughout.
;; ---------------------------------------------------------------------------

(let [l0 (:loss (first history)) l1 (:loss (last history))]
  (assert (< l1 (* 0.8 l0)) (str "loss did not train: " l0 " -> " l1)))
(assert (>= acc 0.9) (str "embedding-space 1-NN accuracy too low: " acc))
(assert (every? #(< (get-in % [:a-bar :max]) 1.0) history)
        "carry left the stable ā<1 regime")
(println "OK — looped encoder trained, 1-NN >=0.9, checkpoint round-trips exactly.")
