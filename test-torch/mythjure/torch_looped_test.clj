(ns mythjure.torch-looped-test
  "Validation for mythjure.torch.looped — the autograd-native looped encoder.

   1. GRADIENTS — autograd through the reused looped core vs central finite
      differences over EVERY element of EVERY tensor leaf (the house style of
      mythjure.backprop-test, h=1e-6, tol 1e-6), for every supported mode:
      :parcae / :free / :faithful?, :last / :mean pooling, windowed
      attention, no positional term, and S < seq-len.
   2. DETERMINISM — two identical train! runs under tensor/manual-seed
      produce identical losses and identical weights (exact =).
   3. Carry diagnostics usable from this entry point (train-torch's monitors
      unchanged + abar-grad), input contracts, and the module/save! →
      load-params → restore checkpoint round trip.

  Runs under the :test-torch alias:  clojure -M:test-torch"
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]
            [mythjure.torch.op :as op]
            [mythjure.torch.autograd :as ag]
            [mythjure.torch.optim :as optim]
            [mythjure.torch.module :as module]
            [mythjure.torch.looped :as lp]
            [mythjure.train-torch :as tt]))

(use-fixtures :once (fn [f] (core/initialize!) (f)))

;; Tiny but fully general config: d-model 4 split over 2 heads, S below
;; seq-len so the narrow-to-S paths (wpe, mask) are exercised everywhere.
(def ^:private cfg
  {:d-in 2 :d-model 4 :n-heads 2 :d-ff 8 :seq-len 4 :T 2 :d-embed 3 :seed 5})

(def ^:private X3 [[0.3 -0.2] [0.1 0.4] [-0.5 0.2]])          ; S=3 < seq-len
(def ^:private target [0.7 -0.3 0.2])

;; ---------------------------------------------------------------------------
;; 1. Finite-difference gradient validation, every leaf, every element
;; ---------------------------------------------------------------------------

(def ^:private FD-H 1e-6)
(def ^:private FD-TOL 1e-6)

(defn- nudge!
  "Add dv to flat element i of a leaf tensor, in place (no-grad, so torch
  allows the in-place write on a requires-grad leaf — the optimizer pattern)."
  [leaf i dv]
  (ag/no-grad
   (let [v (op/method leaf "view" [-1])]
     (op/method v "__setitem__" [i (+ (t/item (op/method v "__getitem__" [i])) dv)]))))

(defn- fd-grad-el
  "Central finite difference of (loss-num) wrt flat element i of leaf."
  [loss-num leaf i]
  (nudge! leaf i FD-H)
  (let [l+ (loss-num)]
    (nudge! leaf i (* -2.0 FD-H))
    (let [l- (loss-num)]
      (nudge! leaf i FD-H)
      (/ (- l+ l-) (* 2.0 FD-H)))))

(defn- fd-vs-autograd
  "Max |autograd − finite-diff| over every element of every param leaf of a
  fresh model under the given config, with loss = ‖embedding − target‖²."
  [config X]
  (let [m (lp/init config)
        loss-t #(let [d (t/sub (:embedding (lp/forward m X)) (t/from-clj target :dtype :float64))]
                  (t/sum (t/mul d d)))
        loss-num #(t/item (ag/no-grad (loss-t)))
        _ (ag/backward! (loss-t))
        leaves (optim/all-param-tensors (:params m))]
    (reduce max 0.0
            (for [leaf leaves
                  :let [g (t/to-clj (t/reshape (ag/grad leaf) [-1]))]
                  i (range (count g))]
              (Math/abs (- (double (nth g i)) (fd-grad-el loss-num leaf i)))))))

(deftest gradients-parcae-last-pool
  (is (< (fd-vs-autograd cfg X3) FD-TOL)))

(deftest gradients-mean-pool
  (is (< (fd-vs-autograd (assoc cfg :pool :mean) X3) FD-TOL)))

(deftest gradients-free-carry
  (is (< (fd-vs-autograd (assoc cfg :carry-mode :free :log-A0 0.7) X3) FD-TOL)))

(deftest gradients-faithful-carry
  (is (< (fd-vs-autograd (assoc cfg :faithful? true) X3) FD-TOL)))

(deftest gradients-windowed-attention
  (is (< (fd-vs-autograd (assoc cfg :window 1) X3) FD-TOL)))

(deftest gradients-no-positional-term
  (is (< (fd-vs-autograd (assoc cfg :positional? false) X3) FD-TOL)))

;; ---------------------------------------------------------------------------
;; 2. Determinism under the curated seed
;; ---------------------------------------------------------------------------

(deftest identical-runs-under-curated-seed
  (let [run (fn []
              (t/manual-seed 123)                       ; curated op (R4)
              (let [X   (t/randn [3 2] :dtype :float64) ; torch-RNG data
                    tgt (t/randn [3] :dtype :float64)
                    m (lp/init cfg)
                    loss-fn (fn [mm]
                              (let [d (t/sub (:embedding (lp/forward mm X)) tgt)]
                                (t/sum (t/mul d d))))
                    {:keys [model history]}
                    (lp/train! m loss-fn :steps 5 :lr 1e-2 :log-every 1 :verbose? false)]
                {:losses (mapv :loss history)
                 :weights (mapv t/to-clj (optim/all-param-tensors (:params model)))}))
        a (run)
        b (run)]
    (is (= (:losses a) (:losses b)) "identical losses, exact =")
    (is (= (:weights a) (:weights b)) "identical weights, exact =")))

;; ---------------------------------------------------------------------------
;; 3. Diagnostics from the autograd entry point
;; ---------------------------------------------------------------------------

(deftest train-torch-monitors-apply-unchanged
  (let [m (lp/init cfg)
        {:keys [cache]} (lp/forward m X3)]
    (is (= 4 (count (tt/effective-abar m))))
    (is (= 4 (count (tt/channel-convergence cache))))))

(deftest abar-grad-and-diagnostics
  (let [m (lp/init cfg)
        {:keys [embedding cache]} (lp/forward m X3)]
    (is (nil? (lp/abar-grad m)) "no dL/dā before backward!")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"backward"
                          (lp/diagnostics m cache)))
    (ag/backward! (t/sum (t/mul embedding embedding)))
    (is (= [4] (t/shape (lp/abar-grad m))))
    (let [diag (lp/diagnostics m cache)]
      (is (= #{:a-bar :dL/dā :converge :worst-channel} (set (keys diag))))
      (is (= 4 (+ (get-in diag [:dL/dā :n-pos]) (get-in diag [:dL/dā :n-neg])))))
    (ag/clear-grads! (:params m))
    (is (nil? (lp/abar-grad m)) "cleared grads ⇒ no dL/dā again")))

(deftest abar-grad-matches-direct-autograd-on-abar
  ;; Independent check of the leaf-grad recovery: retain_grad on the ā
  ;; intermediate and compare torch's own dL/dā against abar-grad.
  (let [m (lp/init cfg)
        {:keys [embedding cache]} (lp/forward m X3)
        _ (core/call (:ab cache) "retain_grad")
        _ (ag/backward! (t/sum (t/mul embedding embedding)))
        direct (ag/grad (:ab cache))
        recovered (lp/abar-grad m)]
    (is (< (t/item (t/tmax (t/abs (t/sub direct recovered)))) 1e-12))))

;; ---------------------------------------------------------------------------
;; Input contracts: variable S, pooling, misuse errors
;; ---------------------------------------------------------------------------

(deftest variable-sequence-length
  (let [m (lp/init cfg)]
    (doseq [S [1 2 4]]
      (is (= [3] (t/shape (:embedding (lp/forward m (vec (repeat S [0.1 -0.1]))))))
          (str "S=" S)))))

(deftest pooling-modes-differ
  (let [emb (fn [pool] (t/to-clj (:embedding (lp/forward (lp/init (assoc cfg :pool pool)) X3))))]
    (is (not= (emb :last) (emb :mean)))))

(deftest no-positional-term-lifts-the-seq-len-cap
  (let [m (lp/init (assoc cfg :positional? false))]
    (is (= [3] (t/shape (:embedding (lp/forward m (vec (repeat 6 [0.1 -0.1])))))))))

(deftest misuse-throws-readably
  (let [m (lp/init cfg)]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"seq-len"
                          (lp/forward m (vec (repeat 5 [0.1 -0.1])))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"d-in"
                          (lp/forward m [[0.1 0.2 0.3]])))))

;; ---------------------------------------------------------------------------
;; Checkpoint round trip: save! → load-params → restore, exact embedding
;; ---------------------------------------------------------------------------

(deftest checkpoint-round-trip
  (let [m (lp/init cfg)
        e1 (t/to-clj (:embedding (lp/forward m X3)))
        path (str (System/getProperty "java.io.tmpdir") "/mythjure-looped-test.pt")]
    (try
      (module/save! (:params m) path)
      (let [m2 (lp/restore (lp/init (assoc cfg :seed 99)) (module/load-params path))]
        (is (= e1 (t/to-clj (:embedding (lp/forward m2 X3))))
            "restored model reproduces embeddings exactly (init seed irrelevant)")
        (is (number? (get-in m2 [:params :block :n-heads]))
            "non-tensor block config survives restore"))
      (finally (.delete (java.io.File. path))))))
