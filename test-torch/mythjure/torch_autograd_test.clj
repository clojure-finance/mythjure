(ns mythjure.torch-autograd-test
  "Validation for mythjure.torch.autograd — torch's reverse-mode autograd as
  the façade's general-purpose gradient path.

  Two jobs, same shape as the rest of the torch suite:
   1. SEMANTICS — pin torch's autograd behaviors we rely on (accumulation,
      graph freeing, no-grad scoping incl. exception safety, tied-parameter
      summing), so a torch upgrade that changes them fails loudly.
   2. ORACLE AGREEMENT — autograd gradients vs the hand-derived VJPs
      (mythjure.backprop-torch), op-level first, then the full model across
      every mode. The VJPs are themselves pinned leaf-for-leaf to the
      finite-diff-verified pure-Clojure oracle, so agreement here chains all
      the way down.

  Runs under the :test-torch alias:  clojure -M:test-torch"
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [mythjure.model :as model]
            [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]
            [mythjure.torch.nn :as nn]
            [mythjure.torch.autograd :as ag]
            [mythjure.backprop-torch :as bpt]
            [mythjure.model-torch :as mt]))

(use-fixtures :once (fn [f] (core/initialize!) (f)))

(def ^:private TOL 1e-12)

(defn- tensor-mad
  "Max abs elementwise difference of two tensors, as a Clojure double."
  [a b]
  (t/item (t/tmax (core/call (t/sub a b) "abs"))))

(defn- ->t64 [x] (t/from-clj x :dtype :float64))

;; ---------------------------------------------------------------------------
;; 1. Flag mechanics and gradient access
;; ---------------------------------------------------------------------------

(deftest requires-grad-roundtrip
  (let [x (->t64 [[1.0 2.0] [3.0 4.0]])]
    (is (false? (ag/requires-grad? x)))
    (is (true? (ag/requires-grad? (ag/requires-grad! x))))
    (is (false? (ag/requires-grad? (ag/requires-grad! x false))))))

(deftest grad-is-nil-before-backward
  (is (nil? (ag/grad (ag/requires-grad! (->t64 [1.0 2.0]))))))

(deftest backward-computes-exact-gradient
  ;; d(Σ x²)/dx = 2x, exactly representable — compare with =, not tolerance
  (let [x (ag/requires-grad! (->t64 [[1.0 2.0] [3.0 4.0]]))]
    (ag/backward! (t/sum (t/mul x x)))
    (is (= [[2.0 4.0] [6.0 8.0]] (t/to-clj (ag/grad x))))))

(deftest grads-accumulate-and-clear
  (let [x (ag/requires-grad! (->t64 [1.0 2.0]))]
    (ag/backward! (t/sum (t/mul x x)))
    (ag/backward! (t/sum (t/mul x x)))
    (is (= [4.0 8.0] (t/to-clj (ag/grad x)))
        "second backward! must ADD, not overwrite")
    (ag/clear-grad! x)
    (is (nil? (ag/grad x)))))

(deftest vector-backward-takes-explicit-vjp-and-retain-graph
  (let [x (ag/requires-grad! (->t64 [1.0 2.0 3.0]))
        y (t/mul x x)]
    (ag/backward! y :gradient (->t64 [1.0 10.0 100.0]) :retain-graph true)
    (is (= [2.0 40.0 600.0] (t/to-clj (ag/grad x))))
    ;; :retain-graph kept the graph alive for a second pass
    (ag/backward! y :gradient (t/ones-like y))
    (is (= [4.0 44.0 606.0] (t/to-clj (ag/grad x))))))

;; ---------------------------------------------------------------------------
;; 2. Grad-mode scoping and graph surgery
;; ---------------------------------------------------------------------------

(deftest no-grad-suspends-taping-and-restores
  (let [x (ag/requires-grad! (->t64 [1.0 2.0]))]
    (is (nil? (ag/no-grad (ag/grad-fn (t/mul x x)))))
    (is (some? (ag/grad-fn (t/mul x x))) "taping must resume after the block")))

(deftest no-grad-restores-after-exception
  (let [x (ag/requires-grad! (->t64 [1.0 2.0]))]
    (is (thrown? clojure.lang.ExceptionInfo
                 (ag/no-grad (throw (ex-info "boom" {})))))
    (is (some? (ag/grad-fn (t/mul x x)))
        "an exception inside no-grad must not leave grad mode off")))

(deftest enable-grad-reenables-inside-no-grad
  (let [x (ag/requires-grad! (->t64 [1.0 2.0]))]
    (is (= [nil "MulBackward0" nil]
           (ag/no-grad
            [(ag/grad-fn (t/mul x x))
             (ag/enable-grad (ag/grad-fn (t/mul x x)))
             (ag/grad-fn (t/mul x x))])))))

(deftest detach-cuts-the-graph
  (let [x (ag/requires-grad! (->t64 [1.0 2.0]))
        y (t/mul x x)
        d (ag/detach y)]
    (is (some? (ag/grad-fn y)))
    (is (nil? (ag/grad-fn d)))
    (is (false? (ag/requires-grad? d)))
    (is (= (t/to-clj y) (t/to-clj d)) "detach shares values")))

;; ---------------------------------------------------------------------------
;; 3. Op-level oracle agreement (autograd vs hand-derived VJP)
;; ---------------------------------------------------------------------------

(deftest gelu-autograd-matches-manual-derivative
  (let [data [[-2.0 -0.5 0.0] [0.3 1.0 2.5]]
        x (ag/requires-grad! (->t64 data))
        y (nn/gelu x)]
    (ag/backward! y :gradient (t/ones-like y))
    (is (< (tensor-mad (ag/grad x) (bpt/gelu' (->t64 data))) TOL))))

(deftest layer-norm-autograd-matches-manual-vjp
  (let [data [[1.0 -2.0 0.5 3.0] [0.1 0.2 -0.4 1.5] [2.0 2.0 -1.0 0.0]]
        G    [[1.0 0.5 -0.5 2.0] [0.0 1.0 1.0 -1.0] [0.3 -0.3 0.7 0.1]]
        gam  [1.1 0.9 1.0 1.2]
        bet  [0.0 0.1 -0.1 0.2]
        x     (ag/requires-grad! (->t64 data))
        gamma (ag/requires-grad! (->t64 gam))
        beta  (ag/requires-grad! (->t64 bet))
        y (nn/layer-norm x [4] :weight gamma :bias beta)
        manual (bpt/ln-backward (->t64 G) (->t64 data) (->t64 gam) 1e-5)]
    (ag/backward! y :gradient (->t64 G))
    (is (< (tensor-mad (ag/grad x) (:dX manual)) TOL))
    (is (< (tensor-mad (ag/grad gamma) (:dgamma manual)) TOL))
    (is (< (tensor-mad (ag/grad beta) (:dbeta manual)) TOL))))

;; ---------------------------------------------------------------------------
;; 4. Param-map helpers and tied parameters
;; ---------------------------------------------------------------------------

(deftest tied-params-accumulate-automatically
  ;; One tensor reachable via two paths: loss = Σw + Σ3w ⇒ dw = 4 everywhere.
  ;; Autograd sums the two contributions with no identity-dedup on our side.
  (let [w (ag/requires-grad! (->t64 [[1.0 2.0] [3.0 4.0]]))
        params {:prelude {:emb w} :coda {:unemb w} :cfg {:d-model 2}}
        loss (t/add (t/sum w) (t/sum (t/mul w (->t64 3.0))))]
    (ag/backward! loss)
    (let [g (ag/param-grads params)]
      (is (= [[4.0 4.0] [4.0 4.0]] (t/to-clj (get-in g [:prelude :emb]))))
      (is (= (t/to-clj (get-in g [:prelude :emb]))
             (t/to-clj (get-in g [:coda :unemb]))))
      (is (nil? (:cfg g)) "non-tensor entries drop out of the grads tree"))
    (ag/clear-grads! params)
    (is (nil? (ag/param-grads params)))))

;; ---------------------------------------------------------------------------
;; 5. Full-model agreement: autograd vs manual BPTT, every leaf, every mode
;; ---------------------------------------------------------------------------

(def ^:private cfg-base
  {:vocab-size 11 :d-model 8 :n-heads 2 :seq-len 6 :T 3 :seed 5})

(def ^:private inputs  [3 1 4 1 5 9])
(def ^:private targets [1 4 1 5 9 2])
(def ^:private wts     [0.0 1.0 0.5 1.0 0.0 2.0])

(defn- grad-tree-mad
  "Max abs diff between the manual grads tree and the autograd grads tree.
  A manual leaf missing on the autograd side errors — intentional."
  [man auto]
  (cond (map? man) (reduce max 0.0 (map (fn [[k v]] (grad-tree-mad v (get auto k))) man))
        (nil? man) 0.0
        :else (tensor-mad man auto)))

(defn- autograd-vs-manual [cfg]
  (let [m    (update (model/init cfg) :params mt/params->torch)
        _    (ag/requires-grad-params! (:params m))
        ft   (mt/forward m inputs targets wts)
        _    (ag/backward! (:loss ft))
        auto (ag/param-grads (:params m))
        man  (:grads (mt/backward m (:cache ft)))]
    (grad-tree-mad man auto)))

(defn- assert-mode [cfg]
  (is (< (autograd-vs-manual cfg) TOL)))

(deftest model-autograd-agrees-standard (assert-mode cfg-base))
(deftest model-autograd-agrees-windowed (assert-mode (assoc cfg-base :window 1)))
(deftest model-autograd-agrees-ablated  (assert-mode (assoc cfg-base :ablate? true)))
(deftest model-autograd-agrees-faithful (assert-mode (assoc cfg-base :faithful? true)))
(deftest model-autograd-agrees-free     (assert-mode (assoc cfg-base :carry-mode :free :log-A0 0.7)))
