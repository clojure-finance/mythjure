(ns mythjure.linalg
  "Minimal dense linear algebra for the transformer block — pure clojure.core.

  Conventions:
   - a VECTOR is a Clojure vector of doubles.
   - a MATRIX is a vector of row vectors. A length-m vector of length-n rows is
     an m×n matrix. A 'sequence of token activations' is a [seq_len × d_model]
     matrix: one row per position.
   - everything is immutable; ops return fresh data. This is slow relative to
     BLAS but trivially fast at toy scale, and it keeps the upcoming manual
     backprop completely explicit (no autograd, no hidden state).

  Written for clarity over speed. When training perf bites, the same API can be
  re-backed by a native BLAS/LAPACK backend without changing callers."
  (:require [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; Vector ops
;; ---------------------------------------------------------------------------

(defn dot [a b] (reduce + (map * a b)))
(defn v+ [a b] (mapv + a b))
(defn v- [a b] (mapv - a b))
(defn scale [s v] (mapv #(* s %) v))
(defn hadamard "Element-wise product." [a b] (mapv * a b))
(defn norm [v] (Math/sqrt (reduce + (map #(* % %) v))))
(defn mean [v] (/ (reduce + v) (double (count v))))

;; ---------------------------------------------------------------------------
;; Matrix ops
;; ---------------------------------------------------------------------------

(defn shape [M] [(count M) (count (first M))])

(defn transpose [M]
  (apply mapv vector M))

(defn matmul
  "A[m×k] · B[k×n] -> [m×n]. Transposes B once so each output entry is a dot of
  an A-row with a B-column."
  [A B]
  (let [Bt (transpose B)]
    (mapv (fn [arow] (mapv (fn [bcol] (dot arow bcol)) Bt)) A)))

(defn mat+ [A B] (mapv v+ A B))
(defn mat- [A B] (mapv v- A B))
(defn scale-mat "Multiply every entry of a matrix by scalar s." [s M] (mapv #(scale s %) M))

(defn add-row-vec
  "Add a length-n bias vector to every row of an m×n matrix (broadcast)."
  [M b]
  (mapv #(v+ % b) M))

(defn linear
  "Affine map of each row: X[m×d_in] · W[d_in×d_out] (+ b). Standard NN layer."
  ([X W] (matmul X W))
  ([X W b] (add-row-vec (matmul X W) b)))

;; ---------------------------------------------------------------------------
;; Activations & normalizations (row-wise over the feature axis)
;; ---------------------------------------------------------------------------

(defn softmax
  "Numerically-stable softmax of a single vector."
  [v]
  (let [m (apply max v)
        ex (mapv #(Math/exp (- % m)) v)
        s (reduce + ex)]
    (mapv #(/ % s) ex)))

(defn softmax-rows [M] (mapv softmax M))

(defn gelu
  "Gaussian Error Linear Unit (tanh approximation, as used in GPT)."
  [x]
  (* 0.5 x (+ 1.0 (Math/tanh (* (Math/sqrt (/ 2.0 Math/PI))
                                (+ x (* 0.044715 x x x)))))))

(defn gelu-row [v] (mapv gelu v))
(defn gelu-mat [M] (mapv gelu-row M))

(defn layernorm-row
  "LayerNorm over a single feature vector: (x-μ)/√(σ²+ε) ⊙ γ + β.
  γ (gamma) and β (beta) are per-feature learnable vectors."
  [x gamma beta eps]
  (let [mu (mean x)
        centered (mapv #(- % mu) x)
        var (mean (mapv #(* % %) centered))
        inv (/ 1.0 (Math/sqrt (+ var eps)))]
    (v+ (hadamard (scale inv centered) gamma) beta)))

(defn layernorm
  "Row-wise LayerNorm of a matrix."
  ([M gamma beta] (layernorm M gamma beta 1e-5))
  ([M gamma beta eps] (mapv #(layernorm-row % gamma beta eps) M)))

;; ---------------------------------------------------------------------------
;; Construction / init
;; ---------------------------------------------------------------------------

(defn rand-matrix
  "Deterministic Gaussian matrix (fixed seed ⇒ reproducible), entries ~
  N(0, scale²)."
  [seed rows cols scale]
  (let [r (java.util.Random. seed)]
    (vec (for [_ (range rows)]
           (vec (for [_ (range cols)] (* scale (.nextGaussian r))))))))

(defn zeros [n] (vec (repeat n 0.0)))
(defn ones [n] (vec (repeat n 1.0)))

;; ---------------------------------------------------------------------------
;; Causal (autoregressive) attention mask
;; ---------------------------------------------------------------------------

(defn causal-mask-row
  "Given a score row of length seq_len and the query position i, set scores at
  key positions j > i to -inf so softmax assigns them ~0 (a token can't attend
  to its future)."
  [scores i]
  (vec (map-indexed (fn [j s] (if (> j i) Double/NEGATIVE_INFINITY s)) scores)))

(defn apply-causal-mask
  "Mask a full [seq_len × seq_len] score matrix so row i ignores columns > i."
  [scores]
  (vec (map-indexed (fn [i row] (causal-mask-row row i)) scores)))

(defn windowed-causal-mask-row
  "Keep only key positions j in [i-w, i] for query i; mask the future (j>i) AND
  anything farther back than the window (j < i-w). With w=1 a position sees only
  itself and its immediate predecessor — so information moves at most one position
  per attention application, hence one position per loop iteration."
  [scores i w]
  (vec (map-indexed (fn [j s] (if (or (> j i) (< j (- i w))) Double/NEGATIVE_INFINITY s))
                    scores)))

(defn apply-windowed-mask
  "Banded causal mask of half-width w over a [seq_len × seq_len] score matrix."
  [scores w]
  (vec (map-indexed (fn [i row] (windowed-causal-mask-row row i w)) scores)))

;; ---------------------------------------------------------------------------
;; REPL pretty-printer
;; ---------------------------------------------------------------------------

(defn pp
  "Compact fixed-precision print of a matrix for REPL inspection."
  ([M] (pp M 3))
  ([M digits]
   (let [fmt (str "%." digits "f")]
     (str/join "\n" (map (fn [row] (str/join " " (map #(format fmt (double %)) row))) M)))))
