(ns mythjure.model-test
  "End-to-end finite-difference gradient checks for the FULL model (Prelude →
  looped block → Coda → cross-entropy), for every trainable parameter leaf, in
  three regimes: standard, windowed attention, and the recurrence ablation.
  Covers what backprop-test does not: wte, wpe, wu, log-A, final LN, MLP biases,
  and the ablated backward path."
  (:require [clojure.test :refer [deftest is testing]]
            [mythjure.model :as model]
            [mythjure.backprop :as bp]))

(def ^:private tol 1e-6)

;; every trainable leaf path in the param tree, tagged matrix? vs vector
(def ^:private paths
  [[[:wte] :m] [[:wpe] :m] [[:wu] :m]
   [[:log-A] :v] [[:ln-f :gamma] :v] [[:ln-f :beta] :v]
   [[:block :attn :wq] :m] [[:block :attn :wk] :m]
   [[:block :attn :wv] :m] [[:block :attn :wo] :m]
   [[:block :mlp :w1] :m] [[:block :mlp :w2] :m]
   [[:block :mlp :b1] :v] [[:block :mlp :b2] :v]
   [[:block :ln1 :gamma] :v] [[:block :ln1 :beta] :v]
   [[:block :ln2 :gamma] :v] [[:block :ln2 :beta] :v]])

(defn- check-all [m inputs targets weights]
  (let [loss-of (fn [mm] (:loss (model/forward mm inputs targets weights)))
        {:keys [cache]} (model/forward m inputs targets weights)
        {:keys [grads]} (model/backward m cache)]
    (doseq [[path kind] paths]
      (let [pp (into [:params] path)
            g  (get-in grads path)
            err (if (= kind :m)
                  (bp/max-abs-diff g (bp/fd-grad (fn [M] (loss-of (assoc-in m pp M))) (get-in m pp)))
                  (bp/max-abs-diff [g] (bp/fd-grad (fn [M] (loss-of (assoc-in m pp (first M)))) [(get-in m pp)])))]
        (is (< err tol) (str path " err=" err))))))

(def ^:private base
  {:vocab-size 5 :d-model 8 :n-heads 2 :d-ff 16 :seq-len 6 :T 3 :seed 2})
(def ^:private inputs  [1 3 0 2 4 0])
(def ^:private targets [3 0 2 4 0 1])
(def ^:private wts     (assoc (vec (repeat 6 0.0)) 3 1.0))   ; loss-masked (copy-task style)

(deftest full-model-gradients
  (testing "standard full-causal model, all trainable leaves"
    (check-all (model/init base) inputs targets wts)))

(deftest windowed-model-gradients
  (testing "windowed (w=1) attention, all trainable leaves"
    (check-all (model/init (assoc base :window 1)) inputs targets wts)))

(deftest ablated-model-gradients
  (testing "recurrence ablation (block reads only E), all trainable leaves"
    (check-all (model/init (assoc base :window 1 :ablate? true)) inputs targets wts)))

(def ^:private faithful-paths
  (into paths [[[:log-dt] :v] [[:B] :v] [[:ln-p :gamma] :v] [[:ln-p :beta] :v]]))

(deftest faithful-model-gradients
  (testing "faithful Parcae (learnable Δ + B̄ injection + prelude LN), all leaves"
    (let [m (model/init (assoc base :faithful? true))
          loss-of (fn [mm] (:loss (model/forward mm inputs targets wts)))
          {:keys [cache]} (model/forward m inputs targets wts)
          {:keys [grads]} (model/backward m cache)]
      (doseq [[path kind] faithful-paths]
        (let [pp (into [:params] path)
              g  (get-in grads path)
              err (if (= kind :m)
                    (bp/max-abs-diff g (bp/fd-grad (fn [M] (loss-of (assoc-in m pp M))) (get-in m pp)))
                    (bp/max-abs-diff [g] (bp/fd-grad (fn [M] (loss-of (assoc-in m pp (first M)))) [(get-in m pp)])))]
          (is (< err tol) (str path " err=" err)))))))
