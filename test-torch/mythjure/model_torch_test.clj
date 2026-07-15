(ns mythjure.model-torch-test
  "Validation ladder step 1 (direction doc §1.5): forward-pass agreement
  between the torch backend and the pure-Clojure oracle, at float64, across
  every architectural variant the paper exercises. Both backends share the
  oracle's init (params->torch conversion), so agreement is leaf-for-leaf."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [mythjure.linalg :as la]
            [mythjure.block :as blk]
            [mythjure.model :as model]
            [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]
            [mythjure.torch.nn :as nn]
            [mythjure.block-torch :as blkt]
            [mythjure.model-torch :as mt]))

(use-fixtures :once (fn [f] (core/initialize!) (f)))

(defn- mad [a b]
  (if (number? a)
    (Math/abs (- (double a) (double b)))
    (reduce max 0.0 (map mad a b))))

(def ^:private TOL 1e-12)

;; ---------------------------------------------------------------------------
;; Block forward
;; ---------------------------------------------------------------------------

(deftest block-forward-agrees-full-causal
  (let [p (blk/init-params {:d-model 32 :n-heads 4 :seed 7})
        X (la/rand-matrix 99 16 32 1.0)]
    (is (< (mad (blk/forward p X)
                (t/to-clj (blkt/forward (mt/params->torch p)
                                        (t/from-clj X :dtype :float64))))
           TOL))))

(deftest block-forward-agrees-windowed
  ;; w=1 banded mask — the §7.4 configuration.
  (let [p (blk/init-params {:d-model 32 :n-heads 4 :seed 7 :window 1})
        X (la/rand-matrix 99 16 32 1.0)]
    (is (< (mad (blk/forward p X)
                (t/to-clj (blkt/forward (mt/params->torch p)
                                        (t/from-clj X :dtype :float64))))
           TOL))))

;; ---------------------------------------------------------------------------
;; Full model forward (loss), all modes, weighted + unweighted
;; ---------------------------------------------------------------------------

(def ^:private cfg-base
  {:vocab-size 11 :d-model 8 :n-heads 2 :seq-len 6 :T 3 :seed 5})

(def ^:private inputs  [3 1 4 1 5 9])
(def ^:private targets [1 4 1 5 9 2])
(def ^:private wts     [0.0 1.0 0.5 1.0 0.0 2.0])

(defn- loss-diffs [cfg]
  (let [m   (model/init cfg)
        mtb (update m :params mt/params->torch)]
    [(Math/abs (- (:loss (model/forward m inputs targets))
                  (t/item (:loss (mt/forward mtb inputs targets)))))
     (Math/abs (- (:loss (model/forward m inputs targets wts))
                  (t/item (:loss (mt/forward mtb inputs targets wts)))))]))

(deftest model-forward-agrees-standard
  (let [[d dw] (loss-diffs cfg-base)]
    (is (< d TOL)) (is (< dw TOL))))

(deftest model-forward-agrees-windowed
  (let [[d dw] (loss-diffs (assoc cfg-base :window 1))]
    (is (< d TOL)) (is (< dw TOL))))

(deftest model-forward-agrees-ablated
  (let [[d dw] (loss-diffs (assoc cfg-base :ablate? true))]
    (is (< d TOL)) (is (< dw TOL))))

(deftest model-forward-agrees-faithful
  ;; learnable Δ + B̄⊙e injection + prelude LN (Parcae Eq. 3 in full)
  (let [[d dw] (loss-diffs (assoc cfg-base :faithful? true))]
    (is (< d TOL)) (is (< dw TOL))))

(deftest model-forward-agrees-free-carry
  ;; unconstrained ā — the §7.1 instability configuration
  (let [[d dw] (loss-diffs (assoc cfg-base :carry-mode :free :log-A0 0.7))]
    (is (< d TOL)) (is (< dw TOL))))

;; ---------------------------------------------------------------------------
;; Pieces
;; ---------------------------------------------------------------------------

(deftest carry-parameterizations-agree
  (let [log-A [0.5 -0.3 1.2 0.0]
        log-dt [0.1 0.54 -0.2 0.0]]
    (is (< (mad (model/a-bar log-A)
                (t/to-clj (mt/a-bar (t/from-clj log-A :dtype :float64))))
           TOL))
    (is (< (mad (model/a-bar-parcae log-A (mapv model/softplus log-dt))
                (t/to-clj (mt/a-bar-parcae
                           (t/from-clj log-A :dtype :float64)
                           (nn/softplus (t/from-clj log-dt :dtype :float64)))))
           TOL))))

(deftest embed-agrees
  (let [m (model/init cfg-base)
        pt (mt/params->torch (:params m))]
    (is (< (mad (model/embed (:params m) inputs)
                (t/to-clj (mt/embed pt inputs)))
           TOL))))

(deftest params->torch-preserves-structure
  (let [p  (:params (model/init (assoc cfg-base :faithful? true)))
        pt (mt/params->torch p)]
    (testing "non-numeric leaves pass through"
      (is (= (get-in p [:block :n-heads]) (get-in pt [:block :n-heads])))
      (is (= (get-in p [:block :window]) (get-in pt [:block :window]))))
    (testing "numeric leaves become tensors of the right shape"
      (is (= [(count (:wte p)) (count (first (:wte p)))]
             (t/shape (:wte pt))))
      (is (= [(count (:log-A p))] (t/shape (:log-A pt)))))))
