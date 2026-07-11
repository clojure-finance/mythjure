(ns mythjure.dynamics
  "Step 0 of the looped-transformer build: the discretization stability math,
  in isolation, before any neural-network machinery.

  The looped (recurrent-depth) transformer refines a hidden state h across T
  loop iterations. Stripping away the nonlinear transformer block, the residual
  stream is a linear time-invariant recurrence:

      h_{t+1} = A_bar * h_t + B_bar * e + R(h_t, e)

  where e is the (re-injected) Prelude encoding and R is the nonlinear
  transformer computation. The LINEAR carry path `A_bar * h_t` is what makes the
  loop stable or explosive. Classical control theory: the recurrence diverges
  when the spectral radius of A_bar is >= 1.

  Parcae's fix (borrowed from S4/Mamba) is a *parameterization* that makes
  divergence structurally impossible on the linear path:

      A      = -exp(log_A)        ; continuous, forced strictly negative
      A_bar  = exp(dt * A)        ; discretized, forced into (0, 1)

  Because exp(negative) is always in (0,1), every eigenvalue of the (diagonal)
  A_bar has magnitude < 1 by construction. No hyperparameter tuning, no Post-Norm
  hacks. This namespace lets you *watch* that happen in the REPL: build an
  unconstrained loop, see it explode; apply the Parcae parameterization, see it
  stay bounded.

  Everything here is plain clojure.core on vectors of doubles — diagonal A means
  all the linear algebra is element-wise, and the eigenvalues of a diagonal
  matrix are just its diagonal. No native libs, no BLAS. Pure dynamics."
  (:require [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; Tiny vector helpers (diagonal world => everything is element-wise)
;; ---------------------------------------------------------------------------

(defn v+ [a b] (mapv + a b))
(defn v* "Element-wise (Hadamard) product." [a b] (mapv * a b))
(defn scale [s v] (mapv #(* s %) v))
(defn vexp [v] (mapv #(Math/exp %) v))
(defn norm "Euclidean (L2) norm." [v] (Math/sqrt (reduce + (map #(* % %) v))))

;; ---------------------------------------------------------------------------
;; The Parcae parameterization of the diagonal carry matrix A_bar
;; ---------------------------------------------------------------------------

(defn a-bar-constrained
  "Parcae's stable discretization of the diagonal carry term.

    log-A : learnable real vector (the only stored parameter)
    dt    : positive discretization step size

  Returns A_bar = exp(dt * (-exp(log-A))), guaranteed element-wise in (0, 1).
  This is the whole trick. Note exp(log-A) > 0  =>  A < 0  =>  dt*A < 0  =>
  exp(dt*A) in (0,1). Stability is not learned or tuned; it is structural."
  [log-A dt]
  (vexp (scale dt (mapv #(- (Math/exp %)) log-A))))

(defn spectral-radius
  "Spectral radius of a DIAGONAL matrix given by its diagonal vector: the largest
  eigenvalue magnitude. For diagonal matrices the eigenvalues ARE the diagonal,
  so this is just max |entry|. The linear recurrence is stable iff this is < 1."
  [a-bar-diag]
  (apply max (map #(Math/abs (double %)) a-bar-diag)))

;; ---------------------------------------------------------------------------
;; The looped recurrence
;; ---------------------------------------------------------------------------

(defn step
  "One loop iteration of the linearized recurrence (R defaults to a no-op so we
  study the bare linear carry path first):

      h_{t+1} = a-bar (*) h_t + b-bar (*) e + R(h_t, e)"
  ([a-bar b-bar e h] (step a-bar b-bar e h (fn [_h _e] (repeat (count h) 0.0))))
  ([a-bar b-bar e h R]
   (v+ (v+ (v* a-bar h) (v* b-bar e)) (R h e))))

(defn run-loop
  "Iterate `step` T times from h0, returning the sequence of states
  [h0 h1 ... hT] so you can inspect the trajectory (norms, decay, geometry)."
  [{:keys [a-bar b-bar e h0 T R]
    :or   {R (fn [_h _e] (repeat (count h0) 0.0))}}]
  (take (inc T) (iterate #(step a-bar b-bar e % R) h0)))

(defn trajectory-norms
  "L2 norm of the hidden state at each iteration — the single most diagnostic
  number. Explosion = norms grow without bound; stability = norms settle."
  [states]
  (mapv norm states))

;; ---------------------------------------------------------------------------
;; Demonstrations you can eval directly in the REPL
;; ---------------------------------------------------------------------------

(def ^:private d 8)            ; toy hidden dimension
(def ^:private e8 (vec (repeat d 1.0)))
(def ^:private h0 (vec (repeat d 1.0)))
(def ^:private b8 (vec (repeat d 0.1)))

(defn demo-explosion
  "UNCONSTRAINED carry: pick A_bar diagonal entries > 1 (what prior looped
  architectures allowed A to learn). Watch the norms blow up across loops.
  Returns {:spectral-radius .. :norms [..]}."
  ([] (demo-explosion 1.3 20))
  ([carry T]
   (let [a-bar (vec (repeat d carry))
         states (run-loop {:a-bar a-bar :b-bar b8 :e e8 :h0 h0 :T T})]
     {:spectral-radius (spectral-radius a-bar)
      :norms (trajectory-norms states)})))

(defn demo-stable
  "CONSTRAINED carry via the Parcae parameterization: choose any real log-A and
  any dt > 0; A_bar lands in (0,1) automatically and the loop stays bounded.
  Returns {:spectral-radius .. :norms [..]}."
  ([] (demo-stable (vec (repeat d 0.0)) 1.0 20))
  ([log-A dt T]
   (let [a-bar (a-bar-constrained log-A dt)
         states (run-loop {:a-bar a-bar :b-bar b8 :e e8 :h0 h0 :T T})]
     {:spectral-radius (spectral-radius a-bar)
      :norms (trajectory-norms states)})))

(defn sparkline
  "Crude REPL-friendly visualization of a norm trajectory."
  [norms]
  (let [blocks " ▁▂▃▄▅▆▇█"
        mx (apply max norms)
        idx (fn [x] (int (* (dec (count blocks)) (/ x (max mx 1e-12)))))]
    (str/join (map #(nth blocks (idx %)) norms))))

(comment
  ;; --- eval these once the nREPL is up ---

  ;; 1. Unconstrained loop diverges (spectral radius >= 1):
  (demo-explosion)
  ;; => norms march off toward infinity

  ;; 2. Parcae-constrained loop is bounded (spectral radius < 1):
  (demo-stable)

  ;; 3. See it at a glance:
  (sparkline (:norms (demo-explosion 1.3 25)))   ; ramps up
  (sparkline (:norms (demo-stable (vec (repeat 8 0.0)) 1.0 25)))  ; settles

  ;; 4. The knife's edge: carry exactly 1.0 is the stability boundary.
  (:norms (demo-explosion 1.0 25))   ; linear growth (driven by b-bar*e)
  (:norms (demo-explosion 0.99 25))  ; converges to a fixed point

  ;; 5. No matter how you set log-A and dt, you can't escape (0,1):
  (spectral-radius (a-bar-constrained [3 -3 0 10 -10 1 -1 5] 2.0))  ; < 1, always
  )
