(ns mythjure.backprop-test
  "Gradient checks: every hand-derived VJP / BPTT gradient must match a central
  finite-difference estimate. Tolerance 1e-6 (central diff with h=1e-6 is good to
  ~1e-9 here; 1e-6 leaves margin)."
  (:require [clojure.test :refer [deftest is testing]]
            [mythjure.linalg :as la]
            [mythjure.block :as blk]
            [mythjure.backprop :as bp]))

(def ^:private tol 1e-6)
(defn- close? [analytic numeric] (< (bp/max-abs-diff analytic numeric) tol))
(defn- close-vec? [a n] (close? [a] [n]))

(deftest attention-backward
  (let [X  (la/rand-matrix 20 4 8 1.0)
        wq (la/rand-matrix 21 8 8 0.4) wk (la/rand-matrix 22 8 8 0.4)
        wv (la/rand-matrix 23 8 8 0.4) wo (la/rand-matrix 24 8 8 0.4)
        R  (la/rand-matrix 25 4 8 1.0)
        L  (fn [X* q k v o] (bp/frob-dot (:attn (bp/attn-forward X* q k v o 2)) R))
        bw (bp/attn-backward R (:cache (bp/attn-forward X wq wk wv wo 2)))]
    (is (close? (:dX  bw) (bp/fd-grad #(L % wq wk wv wo) X)))
    (is (close? (:dwq bw) (bp/fd-grad #(L X % wk wv wo) wq)))
    (is (close? (:dwk bw) (bp/fd-grad #(L X wq % wv wo) wk)))
    (is (close? (:dwv bw) (bp/fd-grad #(L X wq wk % wo) wv)))
    (is (close? (:dwo bw) (bp/fd-grad #(L X wq wk wv %) wo)))))

(deftest block-backward
  (let [p (blk/init-params {:d-model 8 :n-heads 2 :d-ff 16 :seed 3})
        X (la/rand-matrix 30 4 8 1.0)
        R (la/rand-matrix 31 4 8 1.0)
        L (fn [pp XX] (bp/frob-dot (:Y (bp/block-forward pp XX)) R))
        {:keys [dX grads]} (bp/block-backward p R (:cache (bp/block-forward p X)))]
    (is (close? dX (bp/fd-grad #(L p %) X)) "dX")
    (testing "weight matrices"
      (doseq [path [[:attn :wq] [:attn :wk] [:attn :wv] [:attn :wo] [:mlp :w1] [:mlp :w2]]]
        (is (close? (get-in grads path)
                    (bp/fd-grad #(L (assoc-in p path %) X) (get-in p path)))
            (str path))))
    (testing "layernorm gamma/beta"
      (doseq [path [[:ln1 :gamma] [:ln1 :beta] [:ln2 :gamma] [:ln2 :beta]]]
        (is (close-vec? (get-in grads path)
                        (first (bp/fd-grad #(L (assoc-in p path (first %)) X)
                                           [(get-in p path)])))
            (str path))))))

(deftest bptt
  (testing "shared weights, per-channel carry, input injection, initial state"
    (let [p  (blk/init-params {:d-model 8 :n-heads 2 :d-ff 16 :seed 5})
          E  (la/rand-matrix 40 4 8 1.0)
          H0 (la/rand-matrix 41 4 8 0.5)
          Ab (mapv #(+ 0.5 (* 0.4 (/ % 7.0))) (range 8))   ; per-channel carry
          R  (la/rand-matrix 42 4 8 1.0)
          T  3
          Bb (mapv #(+ 0.3 (* 0.2 (/ % 7.0))) (range 8))  ; per-channel B̄ injection
          L  (fn [pp EE HH AA BB] (bp/frob-dot (:HT (bp/recur-forward pp EE HH AA BB T)) R))
          {:keys [caches]} (bp/recur-forward p E H0 Ab Bb T)
          {:keys [grads dA-bar dB-bar dE dH0]} (bp/recur-backward p E Ab Bb R caches)]
      (testing "block weights summed over loop iterations"
        (doseq [path [[:attn :wq] [:attn :wo] [:mlp :w1] [:mlp :w2]]]
          (is (close? (get-in grads path)
                      (bp/fd-grad #(L (assoc-in p path %) E H0 Ab Bb) (get-in p path)))
              (str path))))
      (is (close-vec? dA-bar (first (bp/fd-grad #(L p E H0 (first %) Bb) [Ab]))) "dA-bar")
      (is (close-vec? dB-bar (first (bp/fd-grad #(L p E H0 Ab (first %)) [Bb]))) "dB-bar (input injection)")
      (is (close? dE  (bp/fd-grad #(L p % H0 Ab Bb) E))  "dE")
      (is (close? dH0 (bp/fd-grad #(L p E % Ab Bb) H0)) "dH0 (initial state)"))))
