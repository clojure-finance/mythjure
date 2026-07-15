(ns mythjure.torch-module-test
  "Validation for mythjure.torch.module — Python nn.Module interop.

  Same two jobs as the rest of the torch suite:
   1. ORACLE AGREEMENT — a module's forward must equal the same math done
      with façade ops on its extracted weights (nn.Linear vs matmul+add,
      nn.LayerNorm vs nn/layer-norm, which is itself oracle-pinned).
   2. INTEROP PATTERNS — state_dict key materialization, dotted-key ↔ nested
      param-map round-trips, load_state_dict from Clojure maps (strict and
      not), keep_vars semantics, torch.save/load file round-trip.

  Flagship: a Python-defined nn.Sequential trained entirely by the Clojure
  toolchain — module/params → optim/adam → autograd/backward! — loss must
  collapse. That is the general-purpose-library story end-to-end.

  Runs under the :test-torch alias:  clojure -M:test-torch"
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]
            [mythjure.torch.nn :as nn]
            [mythjure.torch.autograd :as ag]
            [mythjure.torch.optim :as opt]
            [mythjure.torch.module :as mod]))

(use-fixtures :once (fn [f] (core/initialize!) (f)))

(def ^:private TOL 1e-12)

(defn- tensor-mad [a b]
  (t/item (t/tmax (core/call (t/sub a b) "abs"))))

(defn- ->t64 [x] (t/from-clj x :dtype :float64))

;; ---------------------------------------------------------------------------
;; 1. Module forward vs façade math on extracted weights
;; ---------------------------------------------------------------------------

(deftest linear-module-matches-facade-math
  (let [lin (mod/to-dtype (mod/make "Linear" [5 3]) :float64)
        x   (t/randn [4 5] :dtype :float64)
        {:keys [weight bias]} (mod/params lin)]
    (is (< (tensor-mad (mod/call lin x)
                       (t/add (t/matmul x (t/transpose weight)) bias))
           TOL))))

(deftest layer-norm-module-matches-facade-layer-norm
  ;; load non-trivial gamma/beta from a nested Clojure map, then the module
  ;; must equal the (oracle-pinned) functional façade op with the same weights
  (let [ln  (mod/to-dtype (mod/make "LayerNorm" [4]) :float64)
        g   [1.1 0.9 1.3 0.7]
        b   [0.2 -0.1 0.0 0.4]
        x   (t/randn [6 4] :dtype :float64)]
    (mod/load-state-dict! ln {:weight (->t64 g) :bias (->t64 b)})
    (is (< (tensor-mad (mod/call ln x)
                       (nn/layer-norm x [4] :weight (->t64 g) :bias (->t64 b)))
           TOL))))

;; ---------------------------------------------------------------------------
;; 2. state_dict ↔ nested param map
;; ---------------------------------------------------------------------------

(defn- mlp []
  (mod/to-dtype (mod/sequential [(mod/make "Linear" [3 8])
                                 (mod/make "GELU" [] {:approximate "tanh"})
                                 (mod/make "Linear" [8 1])])
                :float64))

(deftest state-dict-nests-numeric-segments
  (let [sd (mod/state-dict (mlp))
        p  (mod/state-dict->params sd)]
    (is (= #{"0.weight" "0.bias" "2.weight" "2.bias"} (set (keys sd))))
    (is (= [8 3] (t/shape (get-in p [0 :weight]))))
    (is (= [1] (t/shape (get-in p [2 :bias]))))))

(deftest params-state-dict-round-trip
  (let [sd (mod/state-dict (mlp))
        rt (mod/params->state-dict (mod/state-dict->params sd))]
    (is (= (set (keys sd)) (set (keys rt))))
    (is (every? #(zero? (tensor-mad (sd %) (rt %))) (keys sd)))))

(deftest load-state-dict-transfers-weights
  (let [a (mod/to-dtype (mod/make "Linear" [5 3]) :float64)
        b (mod/to-dtype (mod/make "Linear" [5 3]) :float64)
        x (t/randn [4 5] :dtype :float64)]
    (mod/load-state-dict! b (mod/params a))
    (is (zero? (tensor-mad (mod/call a x) (mod/call b x))))))

(deftest strict-load-raises-on-missing-keys
  (let [b (mod/to-dtype (mod/make "Linear" [5 3]) :float64)]
    (is (thrown? Exception
                 (mod/load-state-dict! b {:weight (t/randn [3 5] :dtype :float64)})))
    (testing ":strict false permits the partial load"
      (is (some? (mod/load-state-dict! b {:weight (t/randn [3 5] :dtype :float64)}
                                       :strict false))))))

(deftest keep-vars-controls-liveness
  (let [lin (mod/to-dtype (mod/make "Linear" [5 3]) :float64)]
    (is (false? (ag/requires-grad? (get (mod/state-dict lin) "weight")))
        "default state_dict = detached copies, for serialization")
    (is (true? (ag/requires-grad? (get-in (mod/params lin) [:weight])))
        "params = the live parameter tensors, for optimization")))

;; ---------------------------------------------------------------------------
;; 3. Serialization round-trip
;; ---------------------------------------------------------------------------

(deftest save-load-round-trips-through-a-file
  (let [f  (java.io.File/createTempFile "mythjure-sd" ".pt")
        p  (.getAbsolutePath f)
        m  (mlp)
        x  (t/randn [4 3] :dtype :float64)
        y0 (mod/call m x)
        m2 (mlp)]
    (try
      (mod/save! m p)
      (mod/load-state-dict! m2 (mod/load-params p))
      (is (zero? (tensor-mad y0 (mod/call m2 x))))
      (finally (.delete f)))))

;; ---------------------------------------------------------------------------
;; 4. Flagship: Python module, Clojure toolchain, end to end
;; ---------------------------------------------------------------------------

(deftest python-module-trains-under-the-clojure-toolchain
  (core/call core/torch "manual_seed" 42)
  (let [m  (mlp)
        X  (t/randn [32 3] :dtype :float64)
        y  (t/matmul X (->t64 [[1.0] [-2.0] [0.5]]))
        p  (mod/params m)
        o  (opt/adam p :lr 0.02)
        loss-of (fn []
                  (let [e (t/sub (mod/call m X) y)]
                    (t/div (t/sum (t/mul e e)) 32.0)))
        l0 (t/item (loss-of))]
    (dotimes [_ 150]
      (opt/zero-grad! o)
      (ag/backward! (loss-of))
      (opt/step! o))
    (let [l1 (t/item (loss-of))]
      (is (< l1 (/ l0 20.0))
          (str "loss must collapse under Clojure-driven training: " l0 " → " l1))
      (testing "gradients were visible through the param-map the whole time"
        (is (some? (get-in (ag/param-grads p) [0 :weight])))))))
