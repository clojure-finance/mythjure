(ns mythjure.analysis-test
  "Base-suite pins for mythjure.analysis — the estimator layer every §5.3
  spectral number rests on, previously exercised only under :test-torch.
  `spectral-radius` is checked against maps with analytically known spectral
  radii — the same validation cases its docstring cites: a pure scaling map
  (ρ = the scale factor), a diagonal-carry map (ρ = the largest channel
  carry), and an r·R(θ) rotation-scaling map (ρ = r with a complex dominant
  eigenpair — the spiral case the Gelfand schedule exists for). Plus the
  cumulative-:iters contract of `classify-settle`."
  (:require [clojure.test :refer [deftest is testing]]
            [mythjure.analysis :as an]
            [mythjure.linalg :as la]))

(def ^:private H0 (la/rand-matrix 7 3 4 1.0))
(def ^:private tol 2e-3)   ; :rho-spec is rounded to 3 decimals

(deftest spectral-radius-analytic-cases
  (testing "pure scaling map: ρ = the scale factor"
    (let [G (fn [H] (la/scale-mat 0.85 H))]
      (is (< (Math/abs (- 0.85 (:rho-spec (an/spectral-radius G H0)))) tol))))
  (testing "diagonal-carry map: ρ = the largest channel carry"
    (let [abar [0.2 0.5 0.9 0.7]
          G    (fn [H] (mapv (fn [row] (mapv * abar row)) H))]
      (is (< (Math/abs (- 0.9 (:rho-spec (an/spectral-radius G H0)))) tol))))
  (testing "r·R(θ) rotation-scaling: ρ = r (complex eigenpair, the spiral case)"
    (let [r 0.8 th 0.7 c (Math/cos th) s (Math/sin th)
          H2 (la/rand-matrix 11 3 2 1.0)
          G  (fn [H] (mapv (fn [[x y]]
                             [(* r (- (* x c) (* y s)))
                              (* r (+ (* x s) (* y c)))])
                           H))]
      (is (< (Math/abs (- 0.8 (:rho-spec (an/spectral-radius G H2)))) tol)))))

(deftest classify-settle-cumulative-iters
  (testing ":iters counts both legs when the probe fallback fires"
    (let [C  (la/rand-matrix 5 3 4 1.0)
          ;; affine contraction, fixed point 2C: resid halves every iteration,
          ;; so cap 3 forces the probe leg and the probe leg converges.
          G  (fn [H] (la/mat+ (la/scale-mat 0.5 H) C))
          st (an/classify-settle G H0 :tol 1e-9 :cap 3 :probe 100)]
      (is (= :converged (:status st)))
      (is (> (:iters st) 3)))))
