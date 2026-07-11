;; Seed robustness of the operator norm and a dense cross-check of ρ_spec at the
;; low- and high-carry ends, to back two claims made from the single (seed-7)
;; instance: (a) ‖J_G‖₂ > 1 at low carry; (b) whether ρ_spec < 1 is robust across
;; random-weight seeds at high carry. Each seed is settled to a residual-verified
;; fixed point (analysis/classify-settle); seeds whose trajectory never converges
;; are reported separately — a Jacobian "at h*" is undefined for them. (An earlier
;; version settled for a fixed iteration count calibrated on the carry rate ρ;
;; because the loop actually converges at ρ_spec ≈ 1 at high carry, that stopped
;; far from the fixed point and mislabeled non-convergent seeds as ρ_spec > 1.)
;; Run:  clojure -M scripts/seed_verify.clj   (~35 min; dense Jacobian per seed,
;; and non-convergent seeds burn the full residual-settle cap)
(require '[mythjure.block :as blk]
         '[mythjure.linalg :as la]
         '[mythjure.analysis :as an]
         '[clojure.pprint :as pp])

(def E (la/rand-matrix 99 16 32 1.0))
(def SEEDS (range 1 11))

(defn r3 [x] (Double/parseDouble (format "%.3f" x)))
(defn r1e [x] (Double/parseDouble (format "%.1e" x)))
(defn agg [xs]
  (let [n (count xs) mu (/ (reduce + xs) n)
        sd (Math/sqrt (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs)) n))]
    {:mean (r3 mu) :std (r3 sd) :min (r3 (apply min xs)) :max (r3 (apply max xs))}))

(defn row [rho]
  (let [runs (vec (for [s SEEDS]
                    (let [p  (blk/init-params {:d-model 32 :n-heads 4 :seed s})
                          G  (an/gated-update-fn p E rho)
                          st (an/classify-settle G E)]
                      (if (= :non-convergent (:status st))
                        {:seed s :status :non-convergent :resid (:resid st)}
                        (let [H* (:H st)
                              J  (an/jacobian-matrix G H*)]
                          {:seed s :status (:status st) :resid (:resid st)
                           :rho-spec-mf    (:rho-spec (an/spectral-radius G H*))
                           :rho-spec-dense (an/spectral-radius-dense J)
                           :op-norm        (an/operator-norm J :iters 150)})))))
        fixed (filterv #(not= :non-convergent (:status %)) runs)]
    {:rho rho
     :n-with-fixed-point (count fixed)
     :slow           (vec (for [r fixed :when (= :slow (:status r))]
                            {:seed (:seed r) :resid (r1e (:resid r))}))
     :non-convergent (vec (for [r runs :when (= :non-convergent (:status r))]
                            {:seed (:seed r) :resid (r1e (:resid r))}))
     :rho-spec-mf    (agg (map :rho-spec-mf fixed))
     :rho-spec-dense (agg (map :rho-spec-dense fixed))
     :op-norm        (agg (map :op-norm fixed))
     :n-rhospec>1    (count (filter #(> (:rho-spec-dense %) 1.0) fixed))
     :n-opnorm>1     (count (filter #(> (:op-norm %) 1.0) fixed))}))

(println "=== seed robustness (10 seeds, d=32/seq16), residual-verified fixed points, dense cross-check ===")
(def res (mapv row [0.3 0.9]))
(doseq [{:keys [rho rho-spec-mf rho-spec-dense op-norm n-rhospec>1 n-opnorm>1
                n-with-fixed-point slow non-convergent]} res]
  (println (format "ρ=%.2f  [%d/10 with a fixed point; slow: %s; non-convergent: %s]"
                   rho n-with-fixed-point (pr-str slow) (pr-str non-convergent)))
  (println (format "        ρ_spec(mf)=%s  ρ_spec(dense)=%s  ‖J_G‖₂=%s  [among those: ρ_spec>1: %d, ‖J_G‖₂>1: %d]"
                   (pr-str rho-spec-mf) (pr-str rho-spec-dense) (pr-str op-norm) n-rhospec>1 n-opnorm>1)))
(spit "scripts/seed_verify.edn" (with-out-str (pp/pprint res)))
(println "--- wrote scripts/seed_verify.edn ---")
(System/exit 0)
