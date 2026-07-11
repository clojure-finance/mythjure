(ns mythjure.lipschitz
  "Empirical local Lipschitz estimation for the block and the looped update.

  The local Lipschitz constant of f at a point H is the spectral norm (largest
  singular value) of its Jacobian there — the worst-case amplification of a small
  input perturbation:

      Lip(f, H) = max over directions ε of  ‖f(H+ε) − f(H)‖ / ‖ε‖     (ε → 0)

  We estimate it by finite differences along K random directions and taking the
  max. Random directions under-estimate the true spectral norm (a random vector
  rarely aligns with the top singular vector), so this is a LOWER BOUND — but a
  useful one for asking the questions that matter here:

   - Is Lip(block) > 1?  (does an ungated loop amplify, hence drift/diverge)
   - Does the Parcae carry ρ bring the looped update's Lip below 1? (contraction)
   - Is Lip(block) roughly SCALE-INVARIANT in ‖H‖?  If LayerNorm is doing its job
     it should be; where it isn't is where training-time instability will appear.

  Norms are Frobenius (flatten the [seq × d_model] matrix, take L2). Ratios are
  scale-free so Frobenius vs RMS makes no difference."
  (:require [mythjure.linalg :as la]))

(defn fro-norm [M]
  (Math/sqrt (reduce + (map (fn [row] (reduce + (map #(* % %) row))) M))))

(defn rand-like
  "Random matrix the same shape as M (entries ~ N(0,1)), seeded."
  [seed M]
  (la/rand-matrix seed (count M) (count (first M)) 1.0))

(defn scale-to
  "Rescale a matrix to a target Frobenius norm."
  [M target]
  (la/scale-mat (/ target (max 1e-12 (fro-norm M))) M))

(defn lipschitz
  "Empirical local Lipschitz of f at H via K random finite-difference probes.
  `delta` is the relative perturbation magnitude. Returns {:max .. :mean ..}:
  :max ≈ spectral-norm lower bound (the Lipschitz estimate);
  :mean ≈ RMS singular value (context on how anisotropic the Jacobian is).

  CAUTION: random directions are BLIND to sparse axis-aligned worst cases. For a
  diagonal-carry looped update G(H)=A_bar⊙H+increment, the worst direction is the
  channel with the largest carry — if only a few channels are high-carry, random
  probes in d_model·seq_len–dimensional space almost never hit it and this
  under-estimates badly. Use `channel-lipschitz` / `lipschitz-robust` there."
  [f H & {:keys [k delta seed] :or {k 40 delta 1e-3 seed 0}}]
  (let [mag (* delta (max 1.0 (fro-norm H)))
        fH  (f H)
        ratios (mapv (fn [i]
                       (let [eps (scale-to (rand-like (+ seed 1 (* 7 i)) H) mag)]
                         (/ (fro-norm (la/mat- (f (la/mat+ H eps)) fH))
                            (fro-norm eps))))
                     (range k))]
    {:max  (apply max ratios)
     :mean (/ (reduce + ratios) (double k))}))

(defn directional-ratio
  "Finite-difference amplification ‖f(H+ε)−f(H)‖/‖ε‖ of f at H along a specific
  perturbation ε. The building block under both Lipschitz estimators."
  [f H eps]
  (/ (fro-norm (la/mat- (f (la/mat+ H eps)) (f H))) (fro-norm eps)))

(defn channel-axis
  "Perturbation localized to channel c (equal in every position), Frobenius
  magnitude `mag`. This is the eigendirection of a diagonal carry's c-th
  channel — exactly the direction random probing misses."
  [seq-len d c mag]
  (vec (repeat seq-len (assoc (vec (repeat d 0.0)) c (/ mag (Math/sqrt seq-len))))))

(defn channel-lipschitz
  "Probe every channel-axis direction. For diagonal-carry looped updates this
  catches the axis-aligned worst case (≈ the largest channel carry) that random
  probing misses. Returns {:max .. :per-channel [..]}."
  [f H & {:keys [delta] :or {delta 1e-3}}]
  (let [seq-len (count H) d (count (first H))
        mag (* delta (max 1.0 (fro-norm H)))
        per (mapv #(directional-ratio f H (channel-axis seq-len d % mag)) (range d))]
    {:max (apply max per) :per-channel per}))

(defn lipschitz-robust
  "max of random-direction and channel-axis probes — the reliable estimate for
  diagonal-carry looped updates."
  [f H & {:as opts}]
  (max (:max (apply lipschitz f H (mapcat identity opts)))
       (:max (channel-lipschitz f H))))
