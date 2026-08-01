;; Torch-backend rerun of scripts/seed_verify.clj (§5.3 seed-robustness table):
;; dense cross-check of ρ_spec + ‖J_G‖₂ across 10 seeds at ρ=0.3, 0.9 — same
;; seeds, same schedules, float64 — via mythjure.analysis-torch. Writes
;; scripts/seed_verify_torch.edn and diffs against the committed
;; scripts/seed_verify.edn — exactly, except the :resid of seeds without a
;; fixed point (:slow / :non-convergent lists), compared at 50% relative:
;; a non-settling orbit's tail has no floating-point stability under
;; environment drift (see scripts/spectral_torch.clj, 2026-08-01).
;; Run:  clojure -M:torch scripts/seed_verify_torch.clj   (Python w/ torch auto-discovered; MYTHJURE_PYTHON/MYTHJURE_LIBPYTHON override)
(require '[mythjure.block :as blk]
         '[mythjure.linalg :as la]
         '[mythjure.torch.core :as tc]
         '[mythjure.torch.tensor :as t]
         '[mythjure.analysis-torch :as ant]
         '[clojure.edn :as edn]
         '[clojure.pprint :as pp]
         '[clojure.walk :as walk])

(tc/initialize!)

(def E (la/rand-matrix 99 16 32 1.0))
(def E-t (t/from-clj E :dtype :float64))
(def SEEDS (range 1 11))

(defn r3 [x] (Double/parseDouble (format "%.3f" x)))
(defn r1e [x] (Double/parseDouble (format "%.1e" x)))
(defn agg [xs]
  (let [n (count xs) mu (/ (reduce + xs) n)
        sd (Math/sqrt (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs)) n))]
    {:mean (r3 mu) :std (r3 sd) :min (r3 (apply min xs)) :max (r3 (apply max xs))}))

(defn row [rho]
  (let [runs (vec (for [s SEEDS]
                    (let [p   (blk/init-params {:d-model 32 :n-heads 4 :seed s})
                          G-t (ant/gated-update-t p E rho)
                          st  (ant/classify-settle-t G-t E-t)]
                      (if (= :non-convergent (:status st))
                        {:seed s :status :non-convergent :resid (:resid st)}
                        (let [H*t (:H st)
                              J   (ant/jacobian-matrix-t G-t H*t)]
                          {:seed s :status (:status st) :resid (:resid st)
                           :rho-spec-mf    (:rho-spec (ant/spectral-radius-t G-t H*t))
                           :rho-spec-dense (ant/spectral-radius-dense-t J)
                           :op-norm        (ant/operator-norm-t J :iters 150)})))))
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

(println "=== seed robustness, torch backend (float64) ===")
(def t0 (System/nanoTime))
(def res (mapv row [0.3 0.9]))
(println (format "wall: %.1f s" (/ (- (System/nanoTime) t0) 1e9)))
(spit "scripts/seed_verify_torch.edn" (with-out-str (pp/pprint res)))
(println "saved scripts/seed_verify_torch.edn")

(defn deep-close? [a b]
  (cond
    (and (number? a) (number? b))
    (or (== a b)
        (< (Math/abs (- (double a) (double b)))
           (* 1e-4 (max 1.0 (Math/abs (double a))))))
    (and (map? a) (map? b))
    (and (= (set (keys a)) (set (keys b)))
         (every? (fn [[k v]] (deep-close? v (get b k))) a))
    (and (sequential? a) (sequential? b))
    (and (= (count a) (count b)) (every? true? (map deep-close? a b)))
    :else (= a b)))

(def tail-rel-tol 0.5)

(defn strip-tail-resids
  "Drop :resid from :slow/:non-convergent entries (seeds without a
  residual-verified fixed point) so deep-close? diffs everything else
  exactly; those resids are compared separately by tail-resids-close?."
  [x]
  (walk/postwalk
   (fn [m]
     (if (map? m)
       (cond-> m
         (:slow m)           (update :slow (fn [v] (mapv #(dissoc % :resid) v)))
         (:non-convergent m) (update :non-convergent (fn [v] (mapv #(dissoc % :resid) v))))
       m))
   x))

(defn tail-resids-close?
  "Walk a and b in parallel: :resid values inside :slow/:non-convergent lists
  must agree to tail-rel-tol relative. Entry alignment (same seeds, same
  order, same counts) is enforced by deep-close? on the stripped structures."
  [a b]
  (cond
    (and (map? a) (map? b))
    (every? (fn [[k va]]
              (let [vb (get b k)]
                (if (#{:slow :non-convergent} k)
                  (and (= (count va) (count vb))
                       (every? (fn [[ea eb]]
                                 (let [ra (:resid ea) rb (:resid eb)]
                                   (or (and (nil? ra) (nil? rb))
                                       (and (number? ra) (number? rb)
                                            (< (Math/abs (- (double ra) (double rb)))
                                               (* tail-rel-tol (max 1e-12 (Math/abs (double ra)))))))))
                               (map vector va vb)))
                  (tail-resids-close? va vb))))
            a)
    (and (sequential? a) (sequential? b))
    (and (= (count a) (count b)) (every? true? (map tail-resids-close? a b)))
    :else true))

(def committed (edn/read-string (slurp "scripts/seed_verify.edn")))
(def ok? (and (deep-close? (strip-tail-resids committed) (strip-tail-resids res))
              (tail-resids-close? committed res)))
(println (if ok?
           "MATCH — reproduces the committed seed_verify.edn."
           "*** MISMATCH vs committed seed_verify.edn ***"))
(System/exit (if ok? 0 1))
