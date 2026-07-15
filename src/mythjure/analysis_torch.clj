(ns mythjure.analysis-torch
  "Torch-accelerated drivers for mythjure.analysis (the spectral experiments).

  The expensive part of every spectral experiment is the tens of thousands of
  evaluations of the looped update G(H) = ā⊙H + (block(H+E) − (H+E)) inside
  settle, the JVP power iterations, and the O(N) dense Jacobian. This
  namespace mirrors exactly that hot core with tensors END-TO-END (settle,
  jvp, spectral-radius, jacobian-matrix, operator-norm, classify-settle) —
  same iteration schedules, same seeded starting vectors (generated with the
  oracle's own RNG helpers, converted once), float64 throughout. The cheap
  one-shot finite-difference probes (lipschitz, lipschitz-robust) run through
  a Clojure-matrix adapter G so mythjure.lipschitz is reused untouched.

  Entry tables (spectral-analysis, injection-ratio, cf-analysis,
  spectral-seed-sweep) mirror their mythjure.analysis bodies key-for-key —
  outputs are drop-in comparable to the committed .edn results."
  (:require [mythjure.linalg :as la]
            [mythjure.block :as blk]
            [mythjure.lipschitz :as lip]
            [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]
            [mythjure.block-torch :as blkt]
            [mythjure.model-torch :as mt]))

(defn- r3 [x] (Double/parseDouble (format "%.3f" x)))

;; ---------------------------------------------------------------------------
;; Torch-backed G: tensor→tensor for the hot loops, plus a Clojure-matrix
;; adapter for the (cheap, one-shot) lipschitz probes.
;; ---------------------------------------------------------------------------

(defn gated-update-t
  "H-tensor ↦ G(H)-tensor. params/E/a-bar are the oracle's Clojure values."
  [params E a-bar]
  (let [pt (mt/params->torch params)
        Et (t/from-clj E :dtype :float64)
        d  (count (first E))
        av (if (number? a-bar) (vec (repeat d a-bar)) a-bar)
        at (t/from-clj av :dtype :float64)]
    (fn [Ht]
      (let [Z (t/add Ht Et)]
        (t/add (t/mul Ht at) (t/sub (blkt/forward pt Z) Z))))))

(defn increment-t
  "h-tensor ↦ increment(h+E)-tensor = block(h+E) − (h+E)."
  [params E]
  (let [pt (mt/params->torch params)
        Et (t/from-clj E :dtype :float64)]
    (fn [ht]
      (let [Z (t/add ht Et)]
        (t/sub (blkt/forward pt Z) Z)))))

(defn ->clj-fn
  "Wrap a tensor→tensor map as a Clojure-matrix→Clojure-matrix map (for the
  mythjure.lipschitz probes; ~1e-15 identical to the oracle's map)."
  [f-t]
  (fn [H] (t/to-clj (f-t (t/from-clj H :dtype :float64)))))

;; ---------------------------------------------------------------------------
;; Hot core, tensors end-to-end (mirrors mythjure.analysis one-to-one)
;; ---------------------------------------------------------------------------

(defn settle-t
  "analysis/settle on tensors. Returns {:H <tensor> :resid :iters :converged?}."
  [G-t H0-t & {:keys [tol cap] :or {tol 1e-9 cap 20000}}]
  (loop [k 0, H H0-t]
    (let [GH (G-t H)
          r  (/ (t/norm (t/sub GH H)) (t/norm H))]
      (cond
        (< r tol)  {:H H :resid r :iters k :converged? true}
        (>= k cap) {:H H :resid r :iters k :converged? false}
        :else      (recur (inc k) GH)))))

(defn classify-settle-t
  "analysis/classify-settle on tensors (same statuses, same trend fallback)."
  [G-t H0-t & {:keys [tol cap probe] :or {tol 1e-9 cap 20000 probe 2000}}]
  (let [st (settle-t G-t H0-t :tol tol :cap cap)]
    (if (:converged? st)
      (assoc st :status :converged)
      (let [st2 (settle-t G-t (:H st) :tol tol :cap probe)]
        (cond
          (:converged? st2)                     (assoc st2 :status :converged)
          (< (:resid st2) (* 0.75 (:resid st))) (assoc st2 :status :slow)
          :else                                 (assoc st2 :status :non-convergent))))))

(defn jvp-t
  "Central-difference J_G·V on tensors (analysis/jvp)."
  [G-t Ht Vt & {:keys [eps] :or {eps 1e-4}}]
  (let [step (* eps (max 1.0 (t/norm Ht)))
        dV   (t/mul Vt step)]
    (t/mul (t/sub (G-t (t/add Ht dV)) (G-t (t/sub Ht dV)))
           (/ 1.0 (* 2.0 step)))))

(defn spectral-radius-t
  "analysis/spectral-radius on tensors — identical Gelfand schedule and the
  oracle's own seeded starting vector."
  [G-t Ht & {:keys [iters warmup eps seed] :or {iters 160 warmup 70 eps 1e-4 seed 1}}]
  (let [H-clj (t/to-clj Ht)
        v0    (t/from-clj (lip/scale-to (lip/rand-like seed H-clj) 1.0) :dtype :float64)]
    (loop [k 0, v v0, logs []]
      (if (= k iters)
        (let [late (drop warmup logs)]
          {:rho-spec    (r3 (Math/exp (/ (reduce + late) (double (count late)))))
           :growth-max  (r3 (Math/exp (apply max logs)))
           :growth-late (r3 (Math/exp (last logs)))})
        (let [w  (jvp-t G-t Ht v :eps eps)
              nw (t/norm w)]
          (recur (inc k) (t/mul w (/ 1.0 nw)) (conj logs (Math/log nw))))))))

(defn jacobian-matrix-t
  "Densely materialize J_G(H) as an [N N] tensor, one JVP per basis direction
  (analysis/jacobian-matrix; column i = flattened JVP with basis matrix e_i)."
  [G-t Ht & {:keys [eps] :or {eps 1e-4}}]
  (let [[S d] (t/shape Ht)
        N (* S d)
        I (t/eye N :dtype :float64)
        col (fn [i] (t/reshape (jvp-t G-t Ht (t/reshape (t/narrow I 0 i 1) [S d]) :eps eps)
                               [N]))]
    ;; stack columns then transpose ⇒ J[r][c] = ∂G_r/∂H_c, matching the oracle
    (t/transpose (core/call core/torch "stack" (mapv col (range N))) 0 1)))

(defn operator-norm-t
  "analysis/operator-norm on an [N N] tensor (power iteration on JᵀJ, the
  oracle's seeded start)."
  [Jt & {:keys [iters seed] :or {iters 300 seed 1}}]
  (let [N (last (t/shape Jt))
        rng (java.util.Random. (long seed))
        v0 (vec (repeatedly N #(.nextGaussian rng)))
        Jtt (t/transpose Jt 0 1)]
    (loop [k 0, v (t/mul (t/from-clj v0 :dtype :float64) (/ 1.0 (la/norm v0)))]
      (if (= k iters)
        (t/norm (t/matmul Jt v))
        (let [w (t/matmul Jtt (t/matmul Jt v))]
          (recur (inc k) (t/mul w (/ 1.0 (t/norm w)))))))))

(defn spectral-radius-dense-t
  "analysis/spectral-radius-dense on an [N N] tensor."
  [Jt & {:keys [iters warmup seed] :or {iters 300 warmup 150 seed 3}}]
  (let [N (last (t/shape Jt))
        rng (java.util.Random. (long seed))
        v0 (vec (repeatedly N #(.nextGaussian rng)))]
    (loop [k 0, v (t/mul (t/from-clj v0 :dtype :float64) (/ 1.0 (la/norm v0))), logs []]
      (if (= k iters)
        (let [late (drop warmup logs)]
          (r3 (Math/exp (/ (reduce + late) (double (count late))))))
        (let [w (t/matmul Jt v) nw (t/norm w)]
          (recur (inc k) (t/mul w (/ 1.0 nw)) (conj logs (Math/log nw))))))))

;; ---------------------------------------------------------------------------
;; Entry tables (mirror mythjure.analysis key-for-key)
;; ---------------------------------------------------------------------------

(defn spectral-analysis
  [params E & {:keys [rhos k dense? tol] :or {rhos [0.3 0.5 0.7 0.9 0.95] k 50 dense? true tol 1e-9}}]
  (let [E-t (t/from-clj E :dtype :float64)]
    (into (sorted-map)
          (for [rho rhos]
            (let [G-t    (gated-update-t params E rho)
                  st     (settle-t G-t E-t :tol tol)
                  H*t    (:H st)
                  H*     (t/to-clj H*t)
                  mf     (spectral-radius-t G-t H*t)
                  probed (lip/lipschitz-robust (->clj-fn G-t) H* :k k)
                  base   {:H*-norm      (r3 (t/norm H*t))
                          :rho-spec     (:rho-spec mf)
                          :growth-max   (:growth-max mf)
                          :probed-Lip   (r3 probed)
                          :settle-resid (:resid st)
                          :converged?   (:converged? st)}]
              [rho (if dense?
                     (let [J  (jacobian-matrix-t G-t H*t)
                           op (operator-norm-t J)]
                       (assoc base
                              :op-norm        (r3 op)
                              :rho-spec-dense (spectral-radius-dense-t J)
                              :nonnormality   (r3 (- op (:rho-spec mf)))))
                     base)])))))

(defn injection-ratio
  [params E & {:keys [rhos tol] :or {rhos [0.3 0.5 0.7 0.9 0.95] tol 1e-9}}]
  (let [E-t (t/from-clj E :dtype :float64)
        enorm (t/norm E-t)]
    (into (sorted-map)
          (for [rho rhos]
            (let [st (settle-t (gated-update-t params E rho) E-t :tol tol)
                  H*t (:H st)
                  hn (t/norm H*t)]
              [rho {:H*-norm    (r3 hn)
                    :E-norm     (r3 enorm)
                    :ratio      (r3 (/ (t/norm (t/add H*t E-t)) hn))
                    :converged? (:converged? st)}])))))

(defn cf-analysis
  [params E & {:keys [rhos tol] :or {rhos [0.3 0.5 0.7 0.9 0.95] tol 1e-9}}]
  (let [E-t (t/from-clj E :dtype :float64)]
    (into (sorted-map)
          (for [rho rhos]
            (let [st   (settle-t (gated-update-t params E rho) E-t :tol tol)
                  H*t  (:H st)
                  f-t  (increment-t params E)
                  lipf (operator-norm-t (jacobian-matrix-t f-t H*t))
                  hn   (t/norm H*t)
                  F    (t/norm (f-t H*t))]
              [rho {:lipf-true  (r3 lipf)
                    :lipf-probe (r3 (:max (lip/lipschitz (->clj-fn f-t) (t/to-clj H*t) :k 50)))
                    :F          (r3 F)
                    :balance    (r3 (/ F (* (- 1.0 rho) hn)))
                    :cf-true    (r3 (/ lipf (- 1.0 rho)))
                    :converged? (:converged? st)}])))))

(defn spectral-seed-sweep
  [& {:keys [seeds rhos d-model n-heads seq-len tol cap]
      :or {seeds (range 1 11) rhos [0.3 0.5 0.7 0.9 0.95]
           d-model 32 n-heads 4 seq-len 16 tol 1e-9 cap 20000}}]
  (let [E   (la/rand-matrix 99 seq-len d-model 1.0)
        E-t (t/from-clj E :dtype :float64)
        r1e (fn [x] (Double/parseDouble (format "%.1e" x)))]
    (into (sorted-map)
          (for [rho rhos]
            (let [runs (vec (for [s seeds]
                              (let [p   (blk/init-params {:d-model d-model :n-heads n-heads :seed s})
                                    G-t (gated-update-t p E rho)
                                    st  (classify-settle-t G-t E-t :tol tol :cap cap)]
                                {:seed s
                                 :status (:status st)
                                 :resid (:resid st)
                                 :rho-spec (when (not= :non-convergent (:status st))
                                             (:rho-spec (spectral-radius-t G-t (:H st))))})))
                  fixed (filterv #(not= :non-convergent (:status %)) runs)
                  vals (mapv :rho-spec fixed)
                  n    (count vals)
                  mu   (when (pos? n) (/ (reduce + vals) n))
                  sd   (when (pos? n)
                         (Math/sqrt (/ (reduce + (map #(let [x (- % mu)] (* x x)) vals)) n)))]
              [rho {:rho-spec-mean (some-> mu r3)
                    :rho-spec-std  (some-> sd r3)
                    :margin-mean   (some->> mu (- 1) r3)
                    :margin-std    (some-> sd r3)
                    :rho-spec-min  (when (pos? n) (r3 (apply min vals)))
                    :rho-spec-max  (when (pos? n) (r3 (apply max vals)))
                    :n-above-1     (count (filter #(> % 1.0) vals))
                    :n-with-fixed-point n
                    :slow          (vec (for [r fixed :when (= :slow (:status r))]
                                          {:seed (:seed r) :resid (r1e (:resid r))
                                           :rho-spec (:rho-spec r)}))
                    :non-convergent (vec (for [r runs :when (= :non-convergent (:status r))]
                                           {:seed (:seed r) :resid (r1e (:resid r))}))
                    :n             (count runs)}])))))
