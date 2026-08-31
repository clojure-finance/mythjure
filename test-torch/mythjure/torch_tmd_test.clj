(ns mythjure.torch-tmd-test
  "Pins the scicloj bridge (mythjure.torch.tmd): round-trips, dtype and
  column handling, and — most load-bearing — the COPY semantics in both
  directions, so the facade's aliasing rule provably extends across the
  tech.ml.dataset boundary."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [mythjure.torch.autograd :as ag]
            [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]
            [mythjure.torch.tmd :as tmd]
            [tech.v3.dataset :as ds]))

(use-fixtures :once (fn [f] (core/initialize!) (f)))

(defn- D [] (ds/->dataset {:a [1.0 2.0 3.0] :b [4.0 5.0 6.0]}))

(deftest ds->tensor-values-shape-dtype
  (let [tt (tmd/ds->tensor (D) :dtype :float64)]
    (is (= [3 2] (t/shape tt)))
    (is (= [[1.0 4.0] [2.0 5.0] [3.0 6.0]] (t/to-clj tt)))
    (is (= "torch.float64" (str (core/attr tt "dtype")))))
  (testing "default dtype is float32"
    (is (= "torch.float32" (str (core/attr (tmd/ds->tensor (D)) "dtype")))))
  (testing "column selection controls subset AND order"
    (is (= [[4.0 1.0] [5.0 2.0] [6.0 3.0]]
           (t/to-clj (tmd/ds->tensor (D) :columns [:b :a] :dtype :float64)))))
  (testing "int64 label column"
    (let [tt (tmd/ds->tensor (ds/->dataset {:y [0 1 1 0]}) :dtype :int64)]
      (is (= "torch.int64" (str (core/attr tt "dtype"))))
      (is (= [[0] [1] [1] [0]] (t/to-clj tt))))))

(deftest tensor->ds-values-names-1d
  (let [d (tmd/tensor->ds (t/from-clj [[1.0 2.0] [3.0 4.0]] :dtype :float64)
                          :column-names [:x :y])]
    (is (= [:x :y] (vec (ds/column-names d))))
    (is (= [[1.0 2.0] [3.0 4.0]] (mapv vec (ds/rowvecs d)))))
  (testing "1-D tensor becomes a single column"
    (let [d (tmd/tensor->ds (t/from-clj [7.0 8.0] :dtype :float64))]
      (is (= 1 (count (ds/column-names d))))
      (is (= [[7.0] [8.0]] (mapv vec (ds/rowvecs d)))))))

(deftest bridge-copies-in-both-directions
  (testing "ds->tensor: mutating the tensor never touches the dataset"
    (let [d  (D)
          tt (tmd/ds->tensor d :dtype :float64)]
      (t/fill! tt 9.0)
      (is (= [1.0 2.0 3.0] (vec (d :a))))))
  (testing "tensor->ds: the dataset never sees later tensor mutation"
    (let [tt (t/from-clj [[1.0 2.0] [3.0 4.0]] :dtype :float64)
          d  (tmd/tensor->ds tt)]
      (t/fill! tt 7.0)
      (is (= [[1.0 2.0] [3.0 4.0]] (mapv vec (ds/rowvecs d)))))))

(deftest tensor->ds-handles-autograd-tensors
  (let [tt (ag/requires-grad! (t/from-clj [[1.0 2.0]] :dtype :float64))
        d  (tmd/tensor->ds (t/mul tt 2.0))]   ; an autograd INTERMEDIATE
    (is (= [[2.0 4.0]] (mapv vec (ds/rowvecs d))))))

(deftest round-trip-is-identity-on-values
  (let [d1 (D)
        d2 (tmd/tensor->ds (tmd/ds->tensor d1 :dtype :float64)
                           :column-names [:a :b])]
    (is (= (mapv vec (ds/rowvecs d1)) (mapv vec (ds/rowvecs d2))))))
