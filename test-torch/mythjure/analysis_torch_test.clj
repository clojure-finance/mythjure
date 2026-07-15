(ns mythjure.analysis-torch-test
  "The spectral machinery's torch mirror agrees with mythjure.analysis: the
  update map G itself, a short settle, and a short-schedule spectral radius.
  (The full experiment-level reproduction is scripts/spectral_torch.clj /
  seed_verify_torch.clj, which diff against the committed paper .edn files.)"
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [mythjure.linalg :as la]
            [mythjure.block :as blk]
            [mythjure.analysis :as an]
            [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]
            [mythjure.analysis-torch :as ant]))

(use-fixtures :once (fn [f] (core/initialize!) (f)))

(defn- mad2 [a b]
  (reduce max 0.0
          (map (fn [ra rb] (reduce max 0.0 (map #(Math/abs (- (double %1) (double %2))) ra rb)))
               a b)))

(def ^:private p (delay (blk/init-params {:d-model 8 :n-heads 2 :d-ff 16 :seed 5})))
(def ^:private E (delay (la/rand-matrix 40 4 8 1.0)))

(deftest gated-update-agrees-with-oracle
  (let [Go (an/gated-update-fn @p @E 0.5)
        Gt (ant/gated-update-t @p @E 0.5)]
    (is (< (mad2 (Go @E) (t/to-clj (Gt (t/from-clj @E :dtype :float64))))
           1e-12))))

(deftest settle-t-matches-oracle-settle
  (let [Go (an/gated-update-fn @p @E 0.5)
        Gt (ant/gated-update-t @p @E 0.5)
        so (an/settle Go @E :tol 1e-9 :cap 2000)
        st (ant/settle-t Gt (t/from-clj @E :dtype :float64) :tol 1e-9 :cap 2000)]
    (is (= (:converged? so) (:converged? st)))
    (is (= (:iters so) (:iters st)))
    (is (< (mad2 (:H so) (t/to-clj (:H st))) 1e-9))))

(deftest spectral-radius-t-matches-oracle
  (let [Go (an/gated-update-fn @p @E 0.5)
        Gt (ant/gated-update-t @p @E 0.5)
        so (an/settle Go @E :tol 1e-9 :cap 2000)
        H* (:H so)
        ro (an/spectral-radius Go H* :iters 60 :warmup 25)
        rt (ant/spectral-radius-t Gt (t/from-clj H* :dtype :float64) :iters 60 :warmup 25)]
    (is (= (:rho-spec ro) (:rho-spec rt)))
    (is (= (:growth-max ro) (:growth-max rt)))))

(deftest dense-jacobian-and-operator-norm-match
  (let [Go (an/gated-update-fn @p @E 0.5)
        Gt (ant/gated-update-t @p @E 0.5)
        H* (:H (an/settle Go @E :tol 1e-9 :cap 2000))
        Jo (an/jacobian-matrix Go H*)
        Jt (ant/jacobian-matrix-t Gt (t/from-clj H* :dtype :float64))]
    (is (< (mad2 Jo (t/to-clj Jt)) 1e-9))
    (is (< (Math/abs (- (an/operator-norm Jo :iters 100)
                        (ant/operator-norm-t Jt :iters 100)))
           1e-9))))
