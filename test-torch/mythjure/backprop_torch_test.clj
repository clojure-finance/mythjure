(ns mythjure.backprop-torch-test
  "Validation ladder steps 2–3 (direction doc §1.5): backward-pass agreement
  between the torch backend's hand-derived VJPs and the pure-Clojure oracle,
  at float64, leaf-for-leaf — block VJPs first, then full-model BPTT across
  every mode. The oracle's gradients are themselves finite-diff verified, so
  agreement here inherits that verification."
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [mythjure.linalg :as la]
            [mythjure.block :as blk]
            [mythjure.backprop :as bp]
            [mythjure.model :as model]
            [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]
            [mythjure.backprop-torch :as bpt]
            [mythjure.model-torch :as mt]))

(use-fixtures :once (fn [f] (core/initialize!) (f)))

(defn- mad [a b]
  (if (number? a)
    (Math/abs (- (double a) (double b)))
    (reduce max 0.0 (map mad a b))))

(defn- tree-mad
  "Max abs diff across two gradient trees: oracle (nested vectors) vs torch
  (tensor leaves). Missing keys count as mismatch via NPE — intentional."
  [a b]
  (cond (map? a) (reduce max 0.0 (map (fn [[k v]] (tree-mad v (get b k))) a))
        (nil? a) 0.0
        :else (mad (if (vector? a) a (t/to-clj a))
                   (if (vector? b) b (t/to-clj b)))))

(def ^:private TOL 1e-12)

;; ---------------------------------------------------------------------------
;; Ladder step 2: block backward
;; ---------------------------------------------------------------------------

(defn- block-bwd-diffs [p]
  (let [X (la/rand-matrix 40 4 8 1.0)
        R (la/rand-matrix 42 4 8 1.0)
        pt (mt/params->torch p)
        fo (bp/block-forward p X)
        ft (bpt/block-forward pt (t/from-clj X :dtype :float64))
        bo (bp/block-backward p R (:cache fo))
        bt (bpt/block-backward pt (t/from-clj R :dtype :float64) (:cache ft))]
    [(mad (:Y fo) (t/to-clj (:Y ft)))
     (mad (:dX bo) (t/to-clj (:dX bt)))
     (tree-mad (:grads bo) (:grads bt))]))

(deftest block-backward-agrees-full-causal
  (let [[y dx g] (block-bwd-diffs (blk/init-params {:d-model 8 :n-heads 2 :d-ff 16 :seed 5}))]
    (is (< y TOL)) (is (< dx TOL)) (is (< g TOL))))

(deftest block-backward-agrees-windowed
  (let [[y dx g] (block-bwd-diffs (blk/init-params {:d-model 8 :n-heads 2 :d-ff 16 :seed 5 :window 1}))]
    (is (< y TOL)) (is (< dx TOL)) (is (< g TOL))))

;; ---------------------------------------------------------------------------
;; Ladder step 3: full-model BPTT, every leaf, every mode
;; ---------------------------------------------------------------------------

(def ^:private cfg-base
  {:vocab-size 11 :d-model 8 :n-heads 2 :seq-len 6 :T 3 :seed 5})

(def ^:private inputs  [3 1 4 1 5 9])
(def ^:private targets [1 4 1 5 9 2])
(def ^:private wts     [0.0 1.0 0.5 1.0 0.0 2.0])

(defn- full-bwd-diffs [cfg]
  (let [m   (model/init cfg)
        mtb (update m :params mt/params->torch)
        fo  (model/forward m inputs targets wts)
        ft  (mt/forward mtb inputs targets wts)
        bo  (model/backward m (:cache fo))
        bt  (mt/backward mtb (:cache ft))]
    [(Math/abs (- (:loss fo) (t/item (:loss ft))))
     (tree-mad (:grads bo) (:grads bt))
     (mad (:dA-bar bo) (t/to-clj (:dA-bar bt)))]))

(defn- assert-mode [cfg]
  (let [[l g da] (full-bwd-diffs cfg)]
    (is (< l TOL)) (is (< g TOL)) (is (< da TOL))))

(deftest model-backward-agrees-standard (assert-mode cfg-base))
(deftest model-backward-agrees-windowed (assert-mode (assoc cfg-base :window 1)))
(deftest model-backward-agrees-ablated  (assert-mode (assoc cfg-base :ablate? true)))
(deftest model-backward-agrees-faithful (assert-mode (assoc cfg-base :faithful? true)))
(deftest model-backward-agrees-free     (assert-mode (assoc cfg-base :carry-mode :free :log-A0 0.7)))
