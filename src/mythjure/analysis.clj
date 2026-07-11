(ns mythjure.analysis
  "Stability diagnostics for the looped block, built on the empirical Lipschitz
  estimator. These double as TRAINING-TIME MONITORS: re-run them as weights
  change to see whether the model is drifting out of the self-stabilizing regime.

  Findings these tools established (toy: d_model 32, 4 heads, seq 16, random
  weights — re-verify after training):

   1. Lip(block) is NOT scale-invariant. LayerNorm gives Lip(increment) ∝ 1/‖H‖.
      At the well-behaved operating scale (rms≈1, ‖H‖≈22) Lip(block)≈1.7; as ‖H‖
      shrinks toward 0 it blows up (≈17 at ‖H‖≈2). Small activations are the
      danger zone — that's where training-time gradient explosion will originate.

   2. Looping a pre-LN block grows the residual stream LINEARLY (not
      exponentially): the per-step increment has ~constant norm (LN-capped) while
      the multiplicative amplification dies off as ‖H‖ grows.

   3. The Parcae carry A_bar=ρ makes the looped update G a contraction
      (Lip(G)<1). Two coupled laws govern it, both reproductions of the linear
      toy's 1/(1−a) behaviour around a REAL block:
        (A) fixed point obeys (I−A_bar)H* = increment(H*+E), and since
            ‖increment‖ is ~scale-invariant ⇒ ‖H*‖ ≈ ‖increment‖/(1−ρ).
        (B) LayerNorm ⇒ Lip(increment) ≈ (1−ρ) at that fixed point.
      Hence the sub-additive bound Lip(G) ≤ ρ + Lip(increment) ≈ ρ + (1−ρ) = 1
      sits exactly at the stability edge for every ρ; actual Lip(G) sits BELOW it
      by the directional-misalignment margin, which shrinks as ρ→1 (≈0.15–0.17 at
      ρ≤0.7 down to 0.032 at ρ=0.95). NOTE: these probed Lip values are finite-diff
      LOWER bounds; the true stability budget is the spectral margin 1−ρ_spec
      (see spectral-analysis / spectral-seed-sweep below)."
  (:require [mythjure.linalg :as la]
            [mythjure.block :as blk]
            [mythjure.looped :as lp]
            [mythjure.lipschitz :as lip]))

(defn- r3 [x] (Double/parseDouble (format "%.3f" x)))

;; ---------------------------------------------------------------------------
;; (1) How does the block's Lipschitz vary with activation scale?
;; ---------------------------------------------------------------------------

(defn scale-profile
  "Estimate Lip(block) and Lip(increment) at H = c·H0 for each c in `cs`.
  Reveals whether LayerNorm delivers a scale-invariant Lipschitz (it does not —
  Lip(increment) ∝ 1/‖H‖)."
  [params H0 & {:keys [cs k] :or {cs [0.1 0.25 0.5 1.0 2.0 5.0 10.0] k 50}}]
  (let [fwd (fn [H] (blk/forward params H))
        inc (fn [H] (lp/increment params H))]
    (into (sorted-map)
          (for [c cs]
            (let [H (la/scale-mat c H0)]
              [c {:H-norm    (r3 (lip/fro-norm H))
                  :Lip-block (r3 (:max (lip/lipschitz fwd H :k k)))
                  :Lip-incr  (r3 (:max (lip/lipschitz inc H :k k)))}])))))

;; ---------------------------------------------------------------------------
;; (2) The gated update map and its contraction analysis
;; ---------------------------------------------------------------------------

(defn gated-update-fn
  "The looped update as a pure function of the state, for a per-channel carry
  vector `a-bar` (a scalar ρ is broadcast to all channels):
     G(H) = A_bar ⊙ H + increment(H + E).
  Its Jacobian is diag(A_bar) + J_increment(H+E); Lip(G) < 1 ⇒ stable loop. The
  carry term is AXIS-ALIGNED, so the worst direction is the highest-carry
  channel — estimate Lip(G) with `lipschitz/lipschitz-robust`, not random probes."
  [params E a-bar]
  (let [d (count (first E))
        av (if (number? a-bar) (vec (repeat d a-bar)) a-bar)]
    (fn [H] (la/mat+ (mapv #(la/hadamard % av) H)
                     (lp/increment params (la/mat+ H E))))))

(defn settle
  "Iterate H ← G(H) until the relative residual ‖G(H)−H‖/‖H‖ < tol, up to `cap`
  iterations. Returns {:H :resid :iters :converged?}.

  Replaces fixed-iteration settling (the old max(150, 12/(1−ρ)) schedule): the
  loop converges at rate ρ_spec(J_G) — which sits near 1 at high carry — not at
  the carry rate ρ, so an iteration count calibrated on ρ can stop far from the
  fixed point and hand the Jacobian analysis a transient. Worse, a minority of
  random blocks never reach a fixed point at all (bounded, non-settling
  trajectories): callers must check :converged? and report those separately
  rather than assign them a ρ_spec."
  [G H0 & {:keys [tol cap] :or {tol 1e-9 cap 20000}}]
  (loop [t 0 H H0]
    (let [GH (G H)
          r  (/ (lip/fro-norm (la/mat- GH H)) (lip/fro-norm H))]
      (cond
        (< r tol)  {:H H :resid r :iters t :converged? true}
        (>= t cap) {:H H :resid r :iters t :converged? false}
        :else      (recur (inc t) GH)))))

(defn gated-contraction
  "For each SCALAR carry ρ: settle to the fixed point H*, then report the
  contraction analysis there. Confirms Lip(G)<1 and the two laws (A:
  (1−ρ)‖H*‖≈‖increment‖, B: Lip(incr)≈(1−ρ)). `margin = 1 − Lip(G)` is the
  stability headroom. Uses lipschitz-robust so the estimate is reliable even
  though the carry term is axis-aligned."
  [params E & {:keys [rhos k tol] :or {rhos [0.3 0.5 0.7 0.9 0.95] k 50 tol 1e-9}}]
  (let [d (count (first E))
        inc (fn [H] (lp/increment params H))]
    (into (sorted-map)
          (for [rho rhos]
            (let [G     (gated-update-fn params E rho)
                  st    (settle G E :tol tol)
                  H*    (:H st)
                  fro*  (lip/fro-norm H*)
                  incn  (lip/fro-norm (lp/increment params (la/mat+ H* E)))
                  lipI  (:max (lip/lipschitz inc (la/mat+ H* E) :k k))
                  lipG  (lip/lipschitz-robust G H* :k k)]
              [rho {:H*-norm       (r3 fro*)
                    :lawA-lhs      (r3 (* (- 1 rho) fro*)) ; (1−ρ)‖H*‖
                    :lawA-rhs      (r3 incn)               ; ‖increment‖  (should match lhs)
                    :lawB-LipI/dρ  (r3 (/ lipI (- 1 rho))) ; ≈ const ≈ 1
                    :Lip-G         (r3 lipG)
                    :bound-ρ+LipI  (r3 (+ rho lipI))       ; ≈ 1 (sits at the edge)
                    :margin        (r3 (- 1 lipG))
                    :settle-resid  (:resid st)
                    :converged?    (:converged? st)}])))))

(defn per-channel-contraction
  "Contraction analysis for an arbitrary per-channel carry vector `a-bar`. The
  headline finding: Lip(G) is governed by the MAX channel carry (the slowest
  channel), NOT the mean — per-channel log_A buys decoupled memory timescales,
  not a worst-case stability discount. Reports the robust (axis-aware) Lip(G),
  what random-only probing would have (mis)reported, and the per-channel
  amplifications so you can see which channel dominates."
  [params E a-bar & {:keys [k tol] :or {k 40 tol 1e-9}}]
  (let [G   (gated-update-fn params E a-bar)
        st  (settle G E :tol tol)
        H*  (:H st)
        rnd (:max (lip/lipschitz G H* :k k))
        ch  (lip/channel-lipschitz G H*)]
    {:carry-mean   (r3 (/ (reduce + a-bar) (count a-bar)))
     :carry-max    (r3 (apply max a-bar))
     :Lip-G-robust (r3 (max rnd (:max ch)))   ; the true estimate
     :Lip-G-random (r3 rnd)                    ; what random-only would report (too low)
     :margin       (r3 (- 1 (max rnd (:max ch))))
     :worst-channel (apply max-key (:per-channel ch) (range (count a-bar)))
     :H*-rms       (r3 (lp/rms H*))
     :settle-resid (:resid st)
     :converged?   (:converged? st)}))

;; ---------------------------------------------------------------------------
;; (3) Spectral radius vs. operator norm at the fixed point.
;;
;; The Lipschitz probes above estimate the OPERATOR NORM ‖J_G‖₂ — the worst-case
;; one-step amplification. The ASYMPTOTIC per-iteration convergence rate of the
;; loop (the r_eff behind Parcae's test-time scaling law) is instead the SPECTRAL
;; RADIUS ρ_spec(J_G), which ‖J_G‖₂ only upper-bounds. Their gap
;; ‖J_G‖₂ − ρ_spec is the non-normality of J_G = diag(A_bar) + J_increment — the
;; "inner gap" of the hierarchy ρ_spec ≤ ‖J_G‖₂ ≤ ρ + ‖J_incr‖ that the paper
;; left unmeasured. We measure both:
;;
;;   J_G·v   — Jacobian–vector products by central finite difference of the
;;             forward map G (matrix-free; scales to any dimension).
;;   ρ_spec  — Gelfand power iteration: geometric mean of per-step growth factors
;;             over a late window. ‖J^k v‖^{1/k} → ρ_spec even when the dominant
;;             eigenvalue is a COMPLEX pair (the spiral case), where single-step
;;             growth merely oscillates. Validated to exact on scaling, diagonal-
;;             carry, and r·R(θ) rotation-scaling maps.
;;   ‖J_G‖₂  — TRUE top singular value via power iteration on JᵀJ of the densely
;;             materialized J_G. Unlike the finite-difference probe (a lower
;;             bound), this does not under-estimate — and here it exceeds the
;;             probe, so the honest non-normality gap needs it.
;; ---------------------------------------------------------------------------

(defn jvp
  "Jacobian–vector product J_G·V at H by central finite difference of the map G
  (matrix-free — only forward evaluations, so it scales to any dimension). V is a
  unit-Frobenius matrix the shape of H; `eps` is relative to ‖H‖."
  [G H V & {:keys [eps] :or {eps 1e-4}}]
  (let [step (* eps (max 1.0 (lip/fro-norm H)))
        dV   (la/scale-mat step V)]
    (la/scale-mat (/ 1.0 (* 2.0 step))
                  (la/mat- (G (la/mat+ H dV)) (G (la/mat- H dV))))))

(defn spectral-radius
  "Estimate ρ_spec(J_G(H)) by Gelfand power iteration on matrix-free JVPs: iterate
  v ← J_G·v/‖J_G·v‖ and take the geometric mean of the per-step growth factors
  over the last (iters−warmup) steps. Robust to complex/rotational eigenstructure.
  Also returns :growth-max, the worst single-step amplification observed — itself
  a matrix-free lower bound on ‖J_G‖₂."
  [G H & {:keys [iters warmup eps seed] :or {iters 160 warmup 70 eps 1e-4 seed 1}}]
  (loop [k 0, v (lip/scale-to (lip/rand-like seed H) 1.0), logs []]
    (if (= k iters)
      (let [late (drop warmup logs)]
        {:rho-spec    (r3 (Math/exp (/ (reduce + late) (double (count late)))))
         :growth-max  (r3 (Math/exp (apply max logs)))
         :growth-late (r3 (Math/exp (last logs)))})
      (let [w  (jvp G H v :eps eps)
            nw (lip/fro-norm w)]
        (recur (inc k) (la/scale-mat (/ 1.0 nw) w) (conj logs (Math/log nw)))))))

(defn- flatten-mat [H] (vec (apply concat H)))
(defn- unflatten-mat [v d] (mapv vec (partition d v)))

(defn jacobian-matrix
  "Densely materialize J_G(H) as an N×N matrix (N = seq_len·d_model), one JVP per
  canonical basis direction. O(N) forward pairs — cheap at toy scale; prefer the
  matrix-free `spectral-radius` when N is large."
  [G H & {:keys [eps] :or {eps 1e-4}}]
  (let [seq-len (count H) d (count (first H)) N (* seq-len d)
        col (fn [i] (flatten-mat
                     (jvp G H (unflatten-mat (assoc (vec (repeat N 0.0)) i 1.0) d) :eps eps)))]
    (la/transpose (mapv col (range N)))))

(defn operator-norm
  "True operator 2-norm ‖J‖₂ (top singular value) of a materialized Jacobian via
  power iteration on JᵀJ. Unlike the finite-difference probes in `lipschitz`,
  this does not under-estimate the worst-case amplification."
  [J & {:keys [iters seed] :or {iters 300 seed 1}}]
  (let [Jt (la/transpose J) N (count (first J))
        rng (java.util.Random. seed)
        v0 (vec (repeatedly N #(.nextGaussian rng)))
        mv (fn [M v] (mapv #(la/dot % v) M))]
    (loop [k 0, v (la/scale (/ 1.0 (la/norm v0)) v0)]
      (if (= k iters)
        (la/norm (mv J v))
        (let [w (mv Jt (mv J v))]
          (recur (inc k) (la/scale (/ 1.0 (la/norm w)) w)))))))

(defn spectral-radius-dense
  "ρ_spec from a materialized Jacobian J by Gelfand power iteration (matvec).
  A cross-check on the matrix-free `spectral-radius`."
  [J & {:keys [iters warmup seed] :or {iters 300 warmup 150 seed 3}}]
  (let [N (count (first J)) rng (java.util.Random. seed)
        v0 (vec (repeatedly N #(.nextGaussian rng)))
        mv (fn [v] (mapv #(la/dot % v) J))]
    (loop [k 0, v (la/scale (/ 1.0 (la/norm v0)) v0), logs []]
      (if (= k iters)
        (let [late (drop warmup logs)]
          (r3 (Math/exp (/ (reduce + late) (double (count late))))))
        (let [w (mv v) nw (la/norm w)]
          (recur (inc k) (la/scale (/ 1.0 nw) w) (conj logs (Math/log nw))))))))

(defn spectral-analysis
  "For each scalar carry ρ: settle to the fixed point H*, then at J_G(H*) report
    :rho-spec       asymptotic per-iteration convergence rate (matrix-free)
    :rho-spec-dense cross-check from the materialized Jacobian
    :op-norm        TRUE operator norm ‖J_G‖₂ (worst-case one-step amplification)
    :probed-Lip     finite-difference lower bound (what §5.3 Table 3 reports)
    :growth-max     worst single-step growth seen during power iteration
    :nonnormality   ‖J_G‖₂ − ρ_spec  (the hierarchy's inner gap)
  Upgrades the §5.3 / Limitations 'leave to future work' to a measurement. With
  :dense? false, skips the O(N) Jacobian materialization (no op-norm/gap)."
  [params E & {:keys [rhos k dense? tol] :or {rhos [0.3 0.5 0.7 0.9 0.95] k 50 dense? true tol 1e-9}}]
  (into (sorted-map)
        (for [rho rhos]
          (let [G      (gated-update-fn params E rho)
                st     (settle G E :tol tol)
                H*     (:H st)
                mf     (spectral-radius G H*)
                probed (lip/lipschitz-robust G H* :k k)
                base   {:H*-norm      (r3 (lip/fro-norm H*))
                        :rho-spec     (:rho-spec mf)
                        :growth-max   (:growth-max mf)
                        :probed-Lip   (r3 probed)
                        :settle-resid (:resid st)
                        :converged?   (:converged? st)}]
            [rho (if dense?
                   (let [J  (jacobian-matrix G H*)
                         op (operator-norm J)]
                     (assoc base
                            :op-norm        (r3 op)
                            :rho-spec-dense (spectral-radius-dense J)
                            :nonnormality   (r3 (- op (:rho-spec mf)))))
                   base)]))))

;; ---------------------------------------------------------------------------
;; (4) Input-injection scale ratio ‖H*+E‖/‖H*‖ across ρ  (Appendix A test).
;;
;; The self-stabilization derivation pins ‖H*‖, but the scale variable entering
;; LayerNorm is s(H*+E). At high carry ‖H*‖ ≫ ‖E‖ and the two coincide; at low
;; carry the injection E contributes materially, biasing the effective C/F
;; constant. A ratio above 1 at low ρ is consistent in sign with the low-carry
;; softening of C/F (2.33 at ρ=0.3 vs peak 2.78; the probe-table dip to 0.87 is
;; additionally confounded by the probe's own ~2.6× underestimate — App. A).
;; ---------------------------------------------------------------------------

(defn injection-ratio
  "For each scalar carry ρ: settle to H*, report ‖H*+E‖/‖H*‖ (Frobenius) together
  with ‖H*‖ and ‖E‖. Tests the Appendix-A claim that input injection inflates the
  normalizer's scale variable at low carry."
  [params E & {:keys [rhos tol] :or {rhos [0.3 0.5 0.7 0.9 0.95] tol 1e-9}}]
  (let [enorm (lip/fro-norm E)]
    (into (sorted-map)
          (for [rho rhos]
            (let [st (settle (gated-update-fn params E rho) E :tol tol)
                  H* (:H st)
                  hn (lip/fro-norm H*)]
              [rho {:H*-norm    (r3 hn)
                    :E-norm     (r3 enorm)
                    :ratio      (r3 (/ (lip/fro-norm (la/mat+ H* E)) hn))
                    :converged? (:converged? st)}])))))

;; ---------------------------------------------------------------------------
;; (5) The TRUE self-stabilization constant C/F.
;;
;; §5's affine form Lip(f)|h* = (C/F)(1−ρ) [scalar carry] rests on an empirically
;; calibrated C/F. \cref{tab:directional-margin} implied C/F ≈ 1 — but it used the
;; finite-difference PROBED Lip(f), which under-estimates the operator norm. Here
;; C/F = ‖J_f‖₂/(1−ρ) with ‖J_f‖₂ the TRUE operator norm of the increment's
;; Jacobian ∂f/∂h at h* (power iteration on the materialized Jacobian). The
;; fixed-point balance identity F = (1−ρ)‖h*‖ (reported as :balance ≈ 1) makes
;; C/F = ‖J_f‖₂·‖h*‖/F reduce to ‖J_f‖₂/(1−ρ). Finding: the probe under-estimates
;; ‖J_f‖₂ by ≈2.6× at every ρ, so the true C/F is ≈2.5 (roughly flat across ρ),
;; NOT ≈1: the "flat near 1" reading of ρ+Lip(f) is a probe artifact. C/F IS a
;; constant (the derivation's structural claim), but its value is ≈2.5.
;; ---------------------------------------------------------------------------

(defn cf-analysis
  "For each scalar carry ρ: settle to h*, then report the TRUE self-stabilization
  constant C/F = ‖J_f‖₂/(1−ρ), from the operator norm of the increment's Jacobian
  (dense, power-iterated) rather than the finite-difference probe. Also returns the
  probed Lip(f), F=‖f(h*,e)‖, the balance check F/((1−ρ)‖h*‖) (≈1 at a fixed point),
  and cf-true. See \\cref{tab:directional-margin}: probe ⇒ implied C/F≈1 is an
  artifact; true C/F≈2.5."
  [params E & {:keys [rhos tol] :or {rhos [0.3 0.5 0.7 0.9 0.95] tol 1e-9}}]
  (into (sorted-map)
        (for [rho rhos]
          (let [st   (settle (gated-update-fn params E rho) E :tol tol)
                H*   (:H st)
                finc (fn [h] (lp/increment params (la/mat+ h E)))
                lipf (operator-norm (jacobian-matrix finc H*))
                hn   (lip/fro-norm H*)
                F    (lip/fro-norm (finc H*))]
            [rho {:lipf-true  (r3 lipf)
                  :lipf-probe (r3 (:max (lip/lipschitz finc H* :k 50)))
                  :F          (r3 F)
                  :balance    (r3 (/ F (* (- 1.0 rho) hn)))
                  :cf-true    (r3 (/ lipf (- 1.0 rho)))
                  :converged? (:converged? st)}]))))

;; ---------------------------------------------------------------------------
;; (6) Seed robustness of the spectral-margin shape.
;;
;; The single-instance (seed-7) ρ_spec sweep shows a non-monotonic margin bump
;; near ρ≈0.5. Averaging over block-weight seeds (matrix-free, cheap) shows that
;; bump is instance noise: across seeds the margin is widest at LOW carry and
;; collapses monotonically as ρ→1 (0.159 → 0.012). The sweep also surfaces the
;; sharper finding: 1–2 of 10 seeds per carry never converge to a fixed point at
;; all (bounded non-settling orbits) — every seed that does converge is
;; contractive there (ρ_spec < 1 without exception).
;; ---------------------------------------------------------------------------

(defn classify-settle
  "Settle G from H0 with a residual-trend fallback. Returns
    {:status :converged | :slow | :non-convergent, :H, :resid, :iters}.
  :converged — relative residual < tol within `cap` iterations.
  :slow      — residual still above tol at the cap but steadily decreasing
               (drops by ≥25% over `probe` further iterations): a fixed point
               exists and is being approached, just at a rate too close to 1
               to reach tol in budget. :H is the best (last) iterate.
  :non-convergent — residual plateaus or grows: a bounded, non-settling orbit.
               No fixed point is reached; Jacobian analysis 'at h*' is
               undefined for these."
  [G H0 & {:keys [tol cap probe] :or {tol 1e-9 cap 20000 probe 2000}}]
  (let [st (settle G H0 :tol tol :cap cap)]
    (if (:converged? st)
      (assoc st :status :converged)
      (let [st2 (settle G (:H st) :tol tol :cap probe)]
        (cond
          (:converged? st2)                     (assoc st2 :status :converged)
          (< (:resid st2) (* 0.75 (:resid st))) (assoc st2 :status :slow)
          :else                                 (assoc st2 :status :non-convergent))))))

(defn spectral-seed-sweep
  "ρ_spec(J_G) across carries and block-weight seeds (matrix-free). Fixed input
  injection E; only the weights vary. Each seed is settled to a residual-verified
  fixed point first (see `classify-settle`); seeds whose trajectory never
  converges (bounded, non-settling orbits — they exist at moderate/high carry)
  are reported separately under :non-convergent rather than assigned a ρ_spec
  they don't have. Statistics (mean/std/min/max of ρ_spec, margin 1−ρ_spec) are
  over the seeds with a fixed point (:converged plus :slow, the latter listed
  under :slow with their residuals)."
  [& {:keys [seeds rhos d-model n-heads seq-len tol cap]
      :or {seeds (range 1 11) rhos [0.3 0.5 0.7 0.9 0.95]
           d-model 32 n-heads 4 seq-len 16 tol 1e-9 cap 20000}}]
  (let [E   (la/rand-matrix 99 seq-len d-model 1.0)
        r1e (fn [x] (Double/parseDouble (format "%.1e" x)))]
    (into (sorted-map)
          (for [rho rhos]
            (let [runs (vec (for [s seeds]
                              (let [p  (blk/init-params {:d-model d-model :n-heads n-heads :seed s})
                                    G  (gated-update-fn p E rho)
                                    st (classify-settle G E :tol tol :cap cap)]
                                {:seed s
                                 :status (:status st)
                                 :resid (:resid st)
                                 :rho-spec (when (not= :non-convergent (:status st))
                                             (:rho-spec (spectral-radius G (:H st))))})))
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

(comment
  (def p (blk/init-params {:d-model 32 :n-heads 4 :seed 7}))
  (def E (la/rand-matrix 99 16 32 1.0))
  (scale-profile p E)        ; Lip ∝ 1/‖H‖ — small activations are dangerous
  (gated-contraction p E))   ; Lip(G)<1, the two laws, shrinking margin as ρ→1
