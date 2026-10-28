;; v2 review follow-up (2026-10-28): EXACT spectral radii on the rho=0.9 width
;; ladder (scripts/spectral_seed_expansion_torch.clj, 100 seeds per d in
;; {32,64,128,256}).  Reviews of the v2 draft objected that (i) "converged to
;; residual 1e-9 cannot be locally unstable" is not a valid inference and (ii)
;; the Gelfand power-iteration estimate's resolution (~0.008 at d=32) exceeds
;; the mean margin it reports at d=128/256 (0.006/0.003).  Both are answered by
;; the paper's own block-lower-triangular theorem: with causal attention the
;; spectrum of J_G is exactly the union of the n per-token d x d diagonal
;; blocks' spectra.  This script re-settles every seed with the expansion
;; run's exact protocol (same E draw, seeds, schedules, float64), materializes
;; J_G at every converged / still-decaying state, and records
;;   :rho-spec-exact   max |eig| over the per-token diagonal blocks
;;   :op-norm          top singular value of the full J_G (svdvals)
;;   :upper-block-max  max Frobenius norm of the strictly-upper token blocks
;;                     (must be ~0: the block-triangular check at this width)
;;   :rho-spec-full    max |eig| of the full nd x nd J_G (d <= 128 only, as a
;;                     cross-check of the block union; skipped at d=256)
;; alongside the power-iteration estimate of record for the same seed.
;; Checkpointed after every seed.  One rung per process:
;;   MYTHJURE_WIDTH_D=128 clojure -M:torch scripts/width_block_eig_torch.clj
;; Output: scripts/width_block_eig_torch_d<d>.edn
(require '[mythjure.block :as blk]
         '[mythjure.linalg :as la]
         '[mythjure.torch.core :as tc]
         '[mythjure.torch.tensor :as t]
         '[mythjure.analysis-torch :as ant]
         '[clojure.edn :as edn]
         '[clojure.pprint :as pp])

(tc/initialize!)
(tc/call tc/torch "set_num_threads" 2)
(def linalg (tc/attr tc/torch "linalg"))

(def D (Long/parseLong (or (System/getenv "MYTHJURE_WIDTH_D") "32")))
(def HEADS (/ D 8))
(def N-SEEDS (Long/parseLong (or (System/getenv "MYTHJURE_WIDTH_SEEDS") "100")))
(def RHO 0.9)
(def S 16)
(def OUT (str "scripts/width_block_eig_torch_d" D ".edn"))
(def T0 (System/nanoTime))
(defn elapsed-min [] (/ (- (System/nanoTime) T0) 6e10))
(defn r4 [x] (Double/parseDouble (format "%.4f" x)))
(defn r1e [x] (Double/parseDouble (format "%.1e" x)))

(defn eig-abs [J] (vec (t/to-clj (tc/call tc/torch "abs" (tc/call linalg "eigvals" J)))))
(defn op-svd [J] (apply max (t/to-clj (tc/call linalg "svdvals" J))))

(defn block-analysis [J]
  (let [blk-of (fn [r c] (t/narrow (t/narrow J 0 (* r D) D) 1 (* c D) D))
        diag-max (apply max (for [s (range S)] (apply max (eig-abs (blk-of s s)))))
        upper (apply max (for [r (range S) c (range S) :when (> c r)] (t/norm (blk-of r c))))]
    {:rho-spec-exact (r4 diag-max)
     :upper-block-max (r1e upper)
     :op-norm (r4 (op-svd J))
     :rho-spec-full (when (<= D 128) (r4 (apply max (eig-abs J))))}))

(def record
  (let [r (edn/read-string (slurp "scripts/spectral_seed_expansion_torch.edn"))
        rung (first (filter #(= D (:d %)) (:rungs r)))]
    (into {} (map (fn [s] [(:seed s) s]) (:seeds rung)))))

(def E (la/rand-matrix 99 S D 1.0))
(def E-t (t/from-clj E :dtype :float64))
(def results (atom []))

(defn checkpoint! [status]
  (spit OUT (with-out-str
              (pp/pprint {:config {:d D :n-heads HEADS :rho RHO :seq-len S :e-seed 99
                                   :seeds [1 N-SEEDS] :protocol "spectral_seed_expansion_torch"}
                          :status status
                          :elapsed-min (r4 (elapsed-min))
                          :seeds @results}))))

(println (format "=== exact block spectra, d=%d heads=%d rho=%.1f, %d seeds ===" D HEADS RHO N-SEEDS))
(doseq [s (range 1 (inc N-SEEDS))]
  (let [p (blk/init-params {:d-model D :n-heads HEADS :seed s})
        G (ant/gated-update-t p E RHO)
        st (ant/classify-settle-t G E-t)
        rec (get record s)
        base {:seed s :status (:status st) :resid (r1e (:resid st)) :iters (:iters st)
              :rec-status (:status rec) :rec-rho-spec (:rho-spec rec)}
        row (if (contains? #{:converged :slow} (:status st))
              (merge base (block-analysis (ant/jacobian-matrix-t G (:H st))))
              base)]
    (println (format "seed %3d %-14s resid %.1e iters %5d  exact %s  (pi %s)  op %s  upper %s  [%.1f min]"
                     s (name (:status st)) (:resid st) (:iters st)
                     (str (:rho-spec-exact row)) (str (:rec-rho-spec row))
                     (str (:op-norm row)) (str (:upper-block-max row)) (elapsed-min)))
    (flush)
    (swap! results conj row)
    (checkpoint! :in-progress)))

(let [fixed (filter :rho-spec-exact @results)
      conv (filter #(= :converged (:status %)) fixed)
      margins (map #(- 1.0 (:rho-spec-exact %)) fixed)
      mean (fn [xs] (/ (reduce + xs) (count xs)))]
  (println (format "\n=== d=%d summary: fixed %d (converged %d)  exact margin mean %.4f  min %.4f  max %.4f  exact<0 %d/%d  conv-only mean %.4f  op-norm mean %.3f  op>1 %d/%d"
                   D (count fixed) (count conv) (mean margins) (apply min margins) (apply max margins)
                   (count (filter neg? margins)) (count fixed)
                   (mean (map #(- 1.0 (:rho-spec-exact %)) conv))
                   (mean (map :op-norm fixed))
                   (count (filter #(> (:op-norm %) 1.0) fixed)) (count fixed))))
(checkpoint! :done)
(println (str "saved " OUT))
(System/exit 0)
