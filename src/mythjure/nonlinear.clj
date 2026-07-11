(ns mythjure.nonlinear
  "Step 1: put a NONLINEAR term R(h, e) inside the stable loop and watch the
  Parcae 'spiral geometry' emerge — still pure clojure.core, no NN machinery yet.

  In `mythjure.dynamics` the loop's only dynamics were the linear carry
  A_bar * h. That converges straight to a fixed point along a fixed direction —
  boring geometry. The real architecture's interesting behavior lives in the
  nonlinear transformer block R. Here we stub R with the smallest thing that has
  a direction-mixing weight matrix:

      R(h, e) = tanh(W h + e)

  so the full loop is

      h_{t+1} = A_bar (*) h_t + B_bar (*) e + tanh(W h_t + e)

  Three things become visible, all in the REPL on a d=8 toy:

  1. STABILITY IS THE CARRY'S JOB. tanh is bounded, so R can never rescue an
     unconstrained carry: with A_bar >= 1 the loop still explodes (see
     `demo-carry-is-load-bearing`). Parcae's negative-diagonal constraint on the
     LINEAR path is what holds; the bounded nonlinearity is not a safety net.

  2. THERE IS A CONTRACTION THRESHOLD. The loop converges to a fixed point only
     when R's gain (the scale of W) is small enough. Above it (~0.25–0.35 for
     this init) the map stops contracting and the state wanders instead of
     settling. `contraction-sweep` finds the edge.

  3. THE SPIRAL. When it does converge, the update Δh_t = h_{t+1} − h_t shrinks
     (coarse early, fine late) AND its DIRECTION rotates: the cosine of Δh_t
     against the first update sweeps 1 → 0 (orthogonal) → negative (reversed) →
     back. Shrinking + rotating = a spiral. This is the 'updates rapidly decay
     in step-size while becoming increasingly orthogonal' finding from Parcae,
     reproduced on a laptop. See `spiral-metrics`.

  Caveat: W here is RANDOM (untrained). The spiral is a property of the
  contractive dynamics, not of learned weights — training will shape *where* the
  fixed point is and *what* it computes, not whether the geometry is spiral."
  (:require [mythjure.dynamics :as dyn]))

;; ---------------------------------------------------------------------------
;; Small linear-algebra helpers (dense d×d matrix as a vector of row-vectors)
;; ---------------------------------------------------------------------------

(defn matvec
  "Matrix-vector product. W is a vector of row vectors."
  [W v]
  (mapv (fn [row] (reduce + (map * row v))) W))

(defn vtanh [v] (mapv #(Math/tanh %) v))

(defn rand-matrix
  "Deterministic Gaussian matrix (fixed seed => reproducible), entries ~
  N(0, scale^2). `scale` is the gain knob; 1/sqrt(d) is a standard init."
  [seed rows cols scale]
  (let [r (java.util.Random. seed)]
    (vec (for [_ (range rows)]
           (vec (for [_ (range cols)] (* scale (.nextGaussian r))))))))

(defn block
  "The toy transformer-block stand-in: R(h, e) = tanh(W h + e)."
  [W]
  (fn [h e] (vtanh (dyn/v+ (matvec W h) e))))

;; ---------------------------------------------------------------------------
;; Geometry diagnostics
;; ---------------------------------------------------------------------------

(defn dot [a b] (reduce + (map * a b)))

(defn cos-sim [a b]
  (/ (dot a b) (max 1e-12 (* (dyn/norm a) (dyn/norm b)))))

(defn deltas
  "Per-iteration update vectors Δh_t = h_{t+1} − h_t."
  [states]
  (mapv (fn [a b] (mapv - b a)) states (rest states)))

(defn step-sizes
  "‖Δh_t‖ at each iteration — should decay as the loop refines."
  [states]
  (mapv dyn/norm (deltas states)))

(defn cos-successive
  "cos angle between consecutive updates Δh_t and Δh_{t+1}. Stays near 1 for a
  slow rotation (each step turns only a little)."
  [states]
  (let [ds (deltas states)]
    (mapv cos-sim ds (rest ds))))

(defn cos-vs-first
  "cos angle between each update and the FIRST update. Sweeping 1 → 0 → negative
  is the rotation signature of the spiral (direction turns away from where it
  started while the step shrinks)."
  [states]
  (let [ds (deltas states)
        d0 (first ds)]
    (mapv #(cos-sim d0 %) ds)))

;; ---------------------------------------------------------------------------
;; Defaults + demos (d = 8 toy)
;; ---------------------------------------------------------------------------

(def ^:private D 8)
(def ^:private e1 (vec (repeat D 1.0)))
(def ^:private b1 (vec (repeat D 0.1)))
(def ^:private h0 (vec (repeat D 0.0)))
;; Parcae-constrained carry: log_A = 0, dt = 1  =>  A_bar = exp(-1) ≈ 0.368.
(def ^:private a-bar (dyn/a-bar-constrained (vec (repeat D 0.0)) 1.0))

(defn run
  "Run the nonlinear loop and return the trajectory of states.
  `gain` scales W; `carry` overrides the diagonal carry (defaults to the stable
  Parcae A_bar)."
  [{:keys [gain carry seed T] :or {gain 0.25 seed 42 T 40}}]
  (dyn/run-loop {:a-bar (if carry (vec (repeat D carry)) a-bar)
                 :b-bar b1 :e e1 :h0 h0 :T T
                 :R (block (rand-matrix seed D D gain))}))

(defn demo-carry-is-load-bearing
  "Point 1: bounded R cannot save an unconstrained carry. Returns final ‖h‖ for
  the stable carry vs. carries > 1 — the latter explode regardless of R."
  []
  (let [final (fn [carry] (last (dyn/trajectory-norms (run {:carry carry :T 40}))))]
    {:constrained-0.368 (final (Math/exp -1.0))
     :unconstrained-1.1 (final 1.1)
     :unconstrained-1.3 (final 1.3)}))

(defn contraction-sweep
  "Point 2: find the gain threshold where the loop stops converging. Returns,
  per gain, the final step size and whether it converged."
  ([] (contraction-sweep [0.05 0.10 0.15 0.20 0.25 0.30 0.354]))
  ([gains]
   (mapv (fn [g]
           (let [last-step (last (step-sizes (run {:gain g :T 60})))]
             {:gain g
              :final-step last-step
              :converges? (< last-step 1e-3)}))
         gains)))

(defn spiral-metrics
  "Point 3: the spiral. Returns step-size decay (sparkline + values) and the
  rotation signature (cos of each update vs the first). Run at a gain inside the
  contractive regime so convergence is slow enough to watch."
  [& {:keys [gain T] :or {gain 0.25 T 40}}]
  (let [st (run {:gain gain :T T})
        ss (step-sizes st)]
    {:step-size-sparkline (dyn/sparkline ss)
     :step-sizes (mapv #(Double/parseDouble (format "%.4f" %)) (take 16 ss))
     :cos-vs-first (mapv #(Double/parseDouble (format "%.3f" %))
                         (take 16 (cos-vs-first st)))}))

(comment
  ;; 1. The carry, not the nonlinearity, holds stability:
  (demo-carry-is-load-bearing)
  ;; => {:constrained-0.368 ~2.9, :unconstrained-1.1 ~763, :unconstrained-1.3 ~195691}

  ;; 2. Where does contraction break? (~between 0.25 and 0.354 for this init)
  (contraction-sweep)

  ;; 3. The spiral: step size shrinks while direction rotates through orthogonal:
  (spiral-metrics)
  ;; :step-size-sparkline "█▃▂▂▁▁ ..."  (coarse -> fine)
  ;; :cos-vs-first        [1.0 0.44 -0.29 -0.57 ... 0.0 ...]  (rotation)

  ;; Above the threshold it wanders instead of settling (no spiral, no fixed pt):
  (:step-size-sparkline (spiral-metrics :gain 0.354)))
