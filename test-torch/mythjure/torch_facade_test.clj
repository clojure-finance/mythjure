(ns mythjure.torch-facade-test
  "Interop-hardening suite for the mythjure.torch.* façade (direction doc §1.3).

  Two jobs:
   1. ORACLE AGREEMENT — every façade op that overlaps mythjure.linalg must
      match it at float64 to ~machine precision (the linalg oracle is itself
      finite-diff verified).
   2. INTEROP PATTERNS — pin the libpython-clj call forms that torch's C++
      layer actually accepts (tuple-of-ints shapes, auto-converted returns,
      the pointer-safety guard), so regressions surface as test failures
      instead of JVM segfaults.

  Runs under the :test-torch alias:  clojure -M:test-torch
  Needs MYTHJURE_PYTHON / MYTHJURE_LIBPYTHON pointing at a Python with torch
  (see mythjure.torch.core docstring for the pyenv trap)."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [mythjure.linalg :as la]
            [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]
            [mythjure.torch.nn :as nn]
            [mythjure.torch.optim :as opt]
            [mythjure.torch.inspect :as insp]))

(use-fixtures :once (fn [f] (core/initialize!) (f)))

(defn- max-abs-diff [a b]
  (if (number? a)
    (Math/abs (- (double a) (double b)))
    (reduce max 0.0 (map max-abs-diff a b))))

(def ^:private TOL 1e-12)

(defn- ->t64 [x] (t/from-clj x :dtype :float64))

;; ---------------------------------------------------------------------------
;; 1. Oracle agreement (float64)
;; ---------------------------------------------------------------------------

(deftest matmul-agrees-with-oracle
  (let [A (la/rand-matrix 42 7 5 0.5)
        B (la/rand-matrix 43 5 9 0.5)]
    (is (< (max-abs-diff (la/matmul A B)
                         (t/to-clj (t/matmul (->t64 A) (->t64 B))))
           TOL))))

(deftest gelu-is-tanh-approximation
  ;; linalg/gelu is the GPT tanh approximation; torch defaults to erf.
  ;; The façade must pass approximate="tanh" — this test fails loudly if not.
  (let [A (la/rand-matrix 42 6 5 1.5)]
    (is (< (max-abs-diff (la/gelu-mat A)
                         (t/to-clj (nn/gelu (->t64 A))))
           TOL))))

(deftest layer-norm-agrees-with-oracle
  (let [M     (la/rand-matrix 44 6 5 1.5)
        gamma (mapv #(+ 0.5 (* 0.1 %)) (range 5))
        beta  (mapv #(- (* 0.05 %) 0.1) (range 5))]
    (is (< (max-abs-diff (la/layernorm M gamma beta)
                         (t/to-clj (nn/layer-norm (->t64 M) [5]
                                                  :weight (->t64 gamma)
                                                  :bias (->t64 beta))))
           TOL))))

(deftest softmax-agrees-on-causally-masked-scores
  ;; -inf entries (causal mask) must round-trip through the bridge and
  ;; produce exact zeros under softmax, matching the oracle row-wise.
  (let [S (la/apply-causal-mask (la/rand-matrix 45 6 6 1.0))]
    (is (< (max-abs-diff (la/softmax-rows S)
                         (t/to-clj (nn/softmax (->t64 S))))
           TOL))))

(deftest transpose-exp-norm-agree
  (let [A (la/rand-matrix 46 4 7 0.8)]
    (is (< (max-abs-diff (la/transpose A) (t/to-clj (t/transpose (->t64 A)))) TOL))
    (is (< (max-abs-diff (mapv #(mapv (fn [v] (Math/exp v)) %) A)
                         (t/to-clj (t/exp (->t64 A))))
           TOL))
    (is (< (Math/abs (- (Math/sqrt (reduce + (map #(* % %) (flatten A))))
                        (t/norm (->t64 A))))
           TOL))))

(deftest cross-entropy-agrees-with-hand-rolled
  (let [logits [[2.0 1.0 0.1] [0.5 0.3 3.0]]
        manual (let [rows (mapv la/softmax logits)]
                 (/ (- (+ (Math/log (get-in rows [0 0]))
                          (Math/log (get-in rows [1 2]))))
                    2.0))]
    (is (< (Math/abs (- manual
                        (t/item (nn/cross-entropy (->t64 logits)
                                                  (t/from-clj [0 2] :dtype :int64)))))
           TOL))))

;; ---------------------------------------------------------------------------
;; 2. Interop patterns
;; ---------------------------------------------------------------------------

(deftest constructors-take-clojure-shape-vectors
  ;; torch's C++ layer type-checks tuple-of-ints; the façade converts.
  (let [z (t/randn [3 4])]
    (is (= [3 4] (t/shape z)))
    (is (= "torch.float32" (str (core/attr z "dtype"))))
    (is (= "cpu" (str (core/attr z "device")))))
  (is (= [[0.0 0.0] [0.0 0.0]] (t/to-clj (t/zeros [2 2] :dtype :float64))))
  (is (= [1.0 1.0 1.0] (t/to-clj (t/ones [3] :dtype :float64)))))

(deftest reshape-and-arange
  (let [A (la/rand-matrix 47 7 5 0.5)]
    (is (= (t/to-clj (t/reshape (->t64 A) [5 7]))
           (mapv vec (partition 7 (apply concat A))))))
  (is (= [0.0 1.0 2.0 3.0 4.0] (t/to-clj (t/arange 5 :dtype :float64)))))

(deftest to-clj-round-trips
  (let [data [[1.0 2.0] [3.0 4.0]]
        x (->t64 data)]
    (is (= data (t/to-clj x)))
    (is (= 10.0 (t/item (t/sum x))))))

(deftest item-is-idempotent-on-auto-converted-numbers
  ;; libpython-clj auto-converts Python ints/floats on return; item must not
  ;; feed them back into the FFI (that was a literal JVM segfault).
  (is (= 5 (t/item 5)))
  (is (= 2.5 (t/item 2.5))))

(deftest pointer-safety-guard-throws-not-segfaults
  ;; A raw JVM value where a Python object is required must be a friendly
  ;; exception, never a native crash.
  (is (thrown? clojure.lang.ExceptionInfo (core/call 5 "item")))
  (is (thrown? clojure.lang.ExceptionInfo (core/attr "hello" "shape")))
  (is (thrown? clojure.lang.ExceptionInfo (core/call-kw [1 2] "reshape" [] {}))))

;; ---------------------------------------------------------------------------
;; 3. Optimizer plumbing (tied parameters)
;; ---------------------------------------------------------------------------

(defn- tied-fixture []
  (let [emb (t/from-clj [[1.0 2.0] [3.0 4.0]] :dtype :float64)
        w   (t/from-clj [[0.5 0.5] [0.5 0.5]] :dtype :float64)]
    {:params {:prelude {:emb emb} :block {:w w} :coda {:unemb emb}}
     :emb emb :w w
     :grads {:prelude {:emb (t/ones [2 2] :dtype :float64)}
             :block   {:w (t/ones [2 2] :dtype :float64)}
             :coda    {:unemb (t/mul (t/ones [2 2] :dtype :float64)
                                     (t/from-clj 10.0 :dtype :float64))}}}))

(deftest tied-params-identity-deduplicate
  (let [{:keys [params]} (tied-fixture)]
    (is (= 2 (count (opt/all-param-tensors params)))
        "emb reachable via two paths must appear once, or Adam double-steps it")))

(deftest set-grads-adds-on-tied-paths
  (let [{:keys [params emb grads]} (tied-fixture)]
    (opt/set-grads! params grads)
    (is (= [[11.0 11.0] [11.0 11.0]] (t/to-clj (core/attr emb "grad")))
        "1.0 via :emb + 10.0 via :unemb must SUM, not overwrite")))

(deftest adam-steps-manually-set-grads
  (let [{:keys [params w grads]} (tied-fixture)
        before (t/to-clj w)
        o (opt/adam params :lr 0.1)]
    (opt/set-grads! params grads)
    (opt/step! o)
    (is (not= before (t/to-clj w)))
    (opt/zero-grad! o)))

(deftest clip-grad-norm-returns-preclip-total
  (let [{:keys [params grads]} (tied-fixture)]
    (opt/set-grads! params grads)
    ;; tied grad 11s (4 entries) + w grad 1s (4 entries): √(4·121 + 4·1) = √488
    (is (< (Math/abs (- (Math/sqrt 488.0) (opt/clip-grad-norm! params 1.0)))
           1e-9))))

;; ---------------------------------------------------------------------------
;; 4. Inspection (clojure-mcp observability)
;; ---------------------------------------------------------------------------

(deftest tensor-summary-returns-plain-clojure-data
  (let [s (insp/tensor-summary (t/from-clj [[1.0 2.0] [3.0 4.0]] :dtype :float64))]
    (is (= [2 2] (:shape s)))
    (is (= "torch.float64" (:dtype s)))
    (is (= 1.0 (:min s)))
    (is (= 4.0 (:max s)))
    (is (= 2.5 (:mean s)))
    (is (false? (:has-nan? s)))))

(deftest tensor-head-and-param-report
  (let [x (t/from-clj [[1.0 2.0] [3.0 4.0]] :dtype :float64)]
    (is (= [1.0 2.0 3.0] (insp/tensor-head x :n 3)))
    (is (= [2.0] (insp/tensor-head (t/from-clj [2.0] :dtype :float64) :n 10))
        "n larger than numel must clamp, not crash")
    (is (= [{:path [:a] :shape [2 2] :norm (Math/sqrt 30.0)}]
           (map #(update % :norm double) (insp/param-report {:a x}))))))
