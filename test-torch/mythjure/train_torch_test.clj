(ns mythjure.train-torch-test
  "Validation ladder steps 4–5 (direction doc §1.5): the torch training loop
  reproduces the oracle's Adam trajectory step-for-step (same init, same seed,
  same batches), and a smoke train's loss actually decreases."
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [mythjure.model :as model]
            [mythjure.train :as train]
            [mythjure.torch.core :as core]
            [mythjure.model-torch :as mt]
            [mythjure.train-torch :as tt]))

(use-fixtures :once (fn [f] (core/initialize!) (f)))

(def ^:private cfg
  {:vocab-size 11 :d-model 8 :n-heads 2 :seq-len 6 :T 3 :seed 5})

(defn- synth-corpus [n vocab seed]
  (let [r (java.util.Random. (long seed))]
    (vec (repeatedly n #(.nextInt r (int vocab))))))

(deftest adam-trajectory-matches-oracle
  ;; Identical init + rng seed ⇒ identical batch sequence ⇒ the two backends
  ;; must produce the same loss trajectory and diagnostics to fp roundoff.
  (let [corpus (synth-corpus 300 11 42)
        m0  (model/init cfg)
        mt0 (update m0 :params mt/params->torch)
        ho  (:history (binding [*out* (java.io.StringWriter.)]
                        (train/train m0 corpus {:steps 10 :batch-size 2 :lr 1e-3 :seed 7 :log-every 2})))
        ht  (:history (tt/train mt0 corpus {:steps 10 :batch-size 2 :lr 1e-3 :seed 7 :log-every 2 :verbose? false}))]
    (is (= (count ho) (count ht)))
    (is (< (reduce max 0.0 (map (fn [a b] (Math/abs (- (:loss a) (:loss b)))) ho ht))
           1e-10))
    (is (< (Math/abs (- (get-in (last ho) [:converge :mean])
                        (get-in (last ht) [:converge :mean])))
           1e-10))
    (is (= (get-in (last ho) [:dL/dā :n-pos])
           (get-in (last ht) [:dL/dā :n-pos])))))

(deftest smoke-train-loss-decreases
  (let [corpus (synth-corpus 2000 11 43)
        ;; corpus with learnable structure: repeat a short motif with noise
        motif  (vec (take 2000 (cycle [1 2 3 4 5 6 7 8])))
        m0 (update (model/init cfg) :params mt/params->torch)
        r  (tt/train m0 motif {:steps 40 :batch-size 2 :lr 3e-3 :seed 9 :log-every 5 :verbose? false})
        losses (mapv :loss (:history r))]
    (is (< (last losses) (first losses))
        "training on a periodic corpus must reduce the loss")
    (is (< (last losses) 2.1)
        "period-8 corpus: loss must drop clearly below the ~2.39 start (measured ~1.91 at 40 steps)")))
