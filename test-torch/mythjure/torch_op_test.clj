(ns mythjure.torch-op-test
  "Op-coverage machinery tests (direction doc §5.1 item 1).

  Three layers, matching the strategy:
   1. GENERIC tier — torch-fn / nn-fn / method call arbitrary torch ops with
      explicit coercion.
   2. defop MACHINERY — name munging, :default-kw pins (caller-overridable),
      :ret conversions, :method targets, the registry.
   3. TABLE-DRIVEN coverage — every defop row registered by
      mythjure.torch.tensor MUST have an example here that passes; adding a
      row without a test entry fails `every-tensor-op-has-a-passing-example`.

  Runs under the :test-torch alias:  clojure -M:test-torch"
  (:require [clojure.set :as set]
            [clojure.test :refer [deftest is testing use-fixtures]]
            [mythjure.torch.core :as core]
            [mythjure.torch.nn :as nn]
            [mythjure.torch.op :as op]
            [mythjure.torch.tensor :as t]))

(use-fixtures :once (fn [f] (core/initialize!) (f)))

(def ^:private TOL 1e-12)

(defn- ->t64 [x] (t/from-clj x :dtype :float64))

(defn- max-abs-diff [a b]
  (if (number? a)
    (Math/abs (- (double a) (double b)))
    (reduce max 0.0 (map max-abs-diff a b))))

;; ---------------------------------------------------------------------------
;; 1. Generic dispatch tier
;; ---------------------------------------------------------------------------

(deftest generic-torch-fn-with-kwargs
  (is (= [[7.0 7.0] [7.0 7.0]]
         (t/to-clj (op/torch-fn "full" [(core/py-tuple [2 2]) 7.0]
                                {:dtype (core/dtype :float64)})))))

(deftest generic-nn-fn
  (is (= [[1.0 0.0] [3.0 0.0]]
         (t/to-clj (op/nn-fn "relu" [(->t64 [[1.0 -2.0] [3.0 -4.0]])])))))

(deftest generic-method
  (is (= [4 1]
         (t/shape (op/method (->t64 [[1.0 2.0] [3.0 4.0]])
                             "reshape" [(core/py-tuple [4 1])])))))

;; ---------------------------------------------------------------------------
;; 2. defop machinery (test-local rows; the tensor-ns coverage check below
;;    filters the registry by namespace, so these don't pollute it)
;; ---------------------------------------------------------------------------

(deftest py-name-munging
  (is (= "logical_not" (op/op-name->py-name "logical-not")))
  (is (= "add_"        (op/op-name->py-name "add!")))
  (is (= "cat"         (op/op-name->py-name "cat"))))

(op/defop gelu-pinned
  "Test row: :nn target + a :default-kw semantic pin."
  [x]
  {:target :nn :py-name "gelu" :default-kw {:approximate "tanh"}})

(op/defop dot-item
  "Test row: :ret :item."
  [a b]
  {:py-name "dot" :ret :item})

(op/defop reshape-method
  "Test row: :method target + :tuple coercion."
  [t shape]
  {:target :method :py-name "reshape" :coerce {1 :tuple}})

(deftest defop-default-kw-is-a-pin-the-caller-can-override
  (let [x (->t64 [[0.3 -1.2] [2.0 -0.4]])]
    ;; pin active: matches the hand-written nn/gelu (tanh approximation)
    (is (< (max-abs-diff (t/to-clj (gelu-pinned x)) (t/to-clj (nn/gelu x))) TOL))
    ;; caller override wins: erf gelu differs from the tanh approximation
    (is (> (max-abs-diff (t/to-clj (gelu-pinned x :approximate "none"))
                         (t/to-clj (nn/gelu x)))
           0.0))))

(deftest defop-ret-item-returns-a-clojure-number
  (let [r (dot-item (->t64 [1.0 2.0]) (->t64 [3.0 4.0]))]
    (is (number? r))
    (is (< (Math/abs (- r 11.0)) TOL))))

(deftest defop-method-target-with-tuple-coercion
  (is (= [4 1] (t/shape (reshape-method (->t64 [[1.0 2.0] [3.0 4.0]]) [4 1])))))

(deftest defop-registers-its-row
  (let [row (get @op/op-registry 'mythjure.torch.tensor/variance)]
    (is (= "var" (:py-name row)))
    (is (= :torch (:target row)))))

;; ---------------------------------------------------------------------------
;; 3. Table-driven coverage of every mythjure.torch.tensor defop row.
;;    Each entry: qualified op symbol → thunk returning truthy on success.
;;    Inputs are small integers-as-doubles so results are exact where the op
;;    is exact; tolerance elsewhere.
;; ---------------------------------------------------------------------------

(defn- X  [] (->t64 [[1.0 -2.0] [3.0 -4.0]]))
(defn- V  [] (->t64 [1.0 2.0 3.0 4.0]))
(defn- M  [] (t/gt (X) (->t64 0.0)))

(def ^:private op-examples
  {'mythjure.torch.tensor/cat
   #(= [[1.0 -2.0 1.0 -2.0] [3.0 -4.0 3.0 -4.0]] (t/to-clj (t/cat [(X) (X)] :dim 1)))

   'mythjure.torch.tensor/stack
   #(= [2 2 2] (t/shape (t/stack [(X) (X)])))

   'mythjure.torch.tensor/split
   #(= [[1.0] [2.0 3.0 4.0]] (mapv t/to-clj (t/split (V) [1 3])))

   'mythjure.torch.tensor/chunk
   #(= [[1.0 2.0] [3.0 4.0]] (mapv t/to-clj (t/chunk (V) 2)))

   'mythjure.torch.tensor/unbind
   #(= [[1.0 -2.0] [3.0 -4.0]] (mapv t/to-clj (t/unbind (X))))

   'mythjure.torch.tensor/permute
   #(= [4 2 3] (t/shape (t/permute (t/zeros [2 3 4] :dtype :float64) [2 0 1])))

   'mythjure.torch.tensor/flatten
   #(= [1.0 -2.0 3.0 -4.0] (t/to-clj (t/flatten (X))))

   'mythjure.torch.tensor/broadcast-to
   #(= [[1.0 2.0 3.0 4.0] [1.0 2.0 3.0 4.0]] (t/to-clj (t/broadcast-to (V) [2 4])))

   'mythjure.torch.tensor/mean
   #(= 2.5 (t/item (t/mean (V))))

   'mythjure.torch.tensor/prod
   #(= 24.0 (t/item (t/prod (V))))

   'mythjure.torch.tensor/cumsum
   #(= [1.0 3.0 6.0 10.0] (t/to-clj (t/cumsum (V) 0)))

   'mythjure.torch.tensor/argmax
   #(and (= 2 (t/item (t/argmax (X))))          ; flat index of 3.0
         (= [0 0] (t/to-clj (t/argmax (X) :dim 1))))

   'mythjure.torch.tensor/argmin
   #(= 3 (t/item (t/argmin (X))))               ; flat index of -4.0

   'mythjure.torch.tensor/topk
   #(= [[4.0 3.0] [3 2]] (mapv t/to-clj (t/topk (V) 2)))

   'mythjure.torch.tensor/variance
   #(and (< (Math/abs (- (t/item (t/variance (V))) (/ 5.0 3.0))) TOL)
         (= 1.25 (t/item (t/variance (V) :correction 0))))

   'mythjure.torch.tensor/std
   #(< (Math/abs (- (t/item (t/std (V))) (Math/sqrt (/ 5.0 3.0)))) TOL)

   'mythjure.torch.tensor/abs
   #(= [[1.0 2.0] [3.0 4.0]] (t/to-clj (t/abs (X))))

   'mythjure.torch.tensor/pow
   #(= [1.0 4.0 9.0 16.0] (t/to-clj (t/pow (V) 2.0)))

   'mythjure.torch.tensor/clamp
   #(= [[1.0 -1.0] [2.0 -1.0]] (t/to-clj (t/clamp (X) :min -1.0 :max 2.0)))

   'mythjure.torch.tensor/where
   #(= [[1.0 0.0] [3.0 0.0]] (t/to-clj (t/where (M) (X) (t/zeros-like (X)))))

   'mythjure.torch.tensor/floor
   #(= 1.0 (t/item (t/floor (->t64 1.7))))

   'mythjure.torch.tensor/ceil
   #(= 2.0 (t/item (t/ceil (->t64 1.2))))

   'mythjure.torch.tensor/round
   #(and (= 2.0 (t/item (t/round (->t64 2.5))))  ; half-to-even, pinned
         (= 4.0 (t/item (t/round (->t64 3.5)))))

   'mythjure.torch.tensor/sign
   #(= [[1.0 -1.0] [1.0 -1.0]] (t/to-clj (t/sign (X))))

   'mythjure.torch.tensor/log1p
   #(= 0.0 (t/item (t/log1p (->t64 0.0))))

   'mythjure.torch.tensor/expm1
   #(= 0.0 (t/item (t/expm1 (->t64 0.0))))

   'mythjure.torch.tensor/erf
   #(and (= 0.0 (t/item (t/erf (->t64 0.0))))
         (< (Math/abs (- (t/item (t/erf (->t64 1.0))) 0.8427007929497149)) TOL))

   'mythjure.torch.tensor/maximum
   #(= [[1.0 0.0] [3.0 0.0]] (t/to-clj (t/maximum (X) (t/zeros-like (X)))))

   'mythjure.torch.tensor/minimum
   #(= [[0.0 -2.0] [0.0 -4.0]] (t/to-clj (t/minimum (X) (t/zeros-like (X)))))

   'mythjure.torch.tensor/eq
   #(= [true true true true] (t/to-clj (t/eq (V) (V))))

   'mythjure.torch.tensor/ne
   #(= [false false false false] (t/to-clj (t/ne (V) (V))))

   'mythjure.torch.tensor/ge
   #(= [[true false] [true false]] (t/to-clj (t/ge (X) (t/zeros-like (X)))))

   'mythjure.torch.tensor/le
   #(= [[false true] [false true]] (t/to-clj (t/le (X) (t/zeros-like (X)))))

   'mythjure.torch.tensor/logical-and
   #(= [[true false] [true false]] (t/to-clj (t/logical-and (M) (M))))

   'mythjure.torch.tensor/logical-not
   #(= [[false true] [false true]] (t/to-clj (t/logical-not (M))))

   'mythjure.torch.tensor/masked-select
   #(= [1.0 3.0] (t/to-clj (t/masked-select (X) (M))))

   'mythjure.torch.tensor/allclose
   #(and (true? (t/allclose (V) (V)))
         (false? (t/allclose (V) (t/zeros-like (V)))))})

(deftest every-tensor-op-has-a-passing-example
  (let [registered (->> (keys @op/op-registry)
                        (filter #(= "mythjure.torch.tensor" (namespace %)))
                        set)
        covered    (set (keys op-examples))]
    (is (empty? (set/difference registered covered))
        (str "defop rows without a test entry: "
             (set/difference registered covered)))
    (is (empty? (set/difference covered registered))
        (str "test entries without a defop row: "
             (set/difference covered registered)))
    (doseq [[sym example] (sort op-examples)]
      (testing (str sym)
        (is (example))))))

;; einsum is hand-written (variadic), not a defop row — tested separately.
(deftest einsum-variadic
  (is (< (max-abs-diff (t/to-clj (t/einsum "ij,jk->ik" (t/eye 2 :dtype :float64) (X)))
                       (t/to-clj (X)))
         TOL))
  (is (= 11.0 (t/item (t/einsum "i,i->" (->t64 [1.0 2.0]) (->t64 [3.0 4.0]))))))
