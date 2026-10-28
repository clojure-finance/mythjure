;; v2 review follow-up (2026-10-28): the gain-attenuation ratio C/F of
;; tab:cf was reported for one weight instance (seed 7).  Reviews asked for
;; it across seeds.  Same protocol as scripts/spectral_eig_torch.clj (d=32,
;; 4 heads, seq 16, E seed 99, float64): for seeds 1-10 and every carry of
;; record, settle, materialize J_G, and take J_f = J_G - rho*I (the carry is
;; scalar, so the increment Jacobian is J_G minus the diagonal carry), then
;;   Lip(f) = top singular value of J_f,  F = ||f(h*, e)||,
;;   C/F = Lip(f) / (1 - rho)          (eq:marginal, scalar carry).
;; Run:  clojure -M:torch scripts/cf_seeds_torch.clj
(require '[mythjure.block :as blk]
         '[mythjure.linalg :as la]
         '[mythjure.torch.core :as tc]
         '[mythjure.torch.tensor :as t]
         '[mythjure.analysis-torch :as ant]
         '[clojure.pprint :as pp])

(tc/initialize!)
(tc/call tc/torch "set_num_threads" 2)
(def linalg (tc/attr tc/torch "linalg"))
(defn r3 [x] (Double/parseDouble (format "%.3f" x)))
(defn op-svd [J] (apply max (t/to-clj (tc/call linalg "svdvals" J))))

(def D 32) (def S 16)
(def E (la/rand-matrix 99 S D 1.0))
(def E-t (t/from-clj E :dtype :float64))
(def RHOS [0.3 0.5 0.7 0.9 0.95])
(def SEEDS (range 1 11))

(def rows
  (vec (for [s SEEDS rho RHOS]
         (let [p (blk/init-params {:d-model D :n-heads 4 :seed s})
               G (ant/gated-update-t p E rho)
               f (ant/increment-t p E)
               st (ant/classify-settle-t G E-t)]
           (if (contains? #{:converged :slow} (:status st))
             (let [J (ant/jacobian-matrix-t G (:H st))
                   I (t/eye (* S D) :dtype :float64)
                   Jf (t/sub J (t/mul I rho))
                   lip (op-svd Jf)
                   F (t/norm (f (:H st)))
                   row {:seed s :rho rho :status (:status st) :iters (:iters st)
                        :lip (r3 lip) :F (r3 F) :h-norm (r3 (t/norm (:H st)))
                        :cf (r3 (/ lip (- 1.0 rho)))}]
               (println (pr-str row)) (flush) row)
             (let [row {:seed s :rho rho :status (:status st) :resid (:resid st)}]
               (println (pr-str row)) (flush) row))))))

(def summary
  (into (sorted-map)
        (for [rho RHOS]
          (let [ok (filter #(and (= rho (:rho %)) (:cf %)) rows)
                cfs (map :cf ok)
                mean (/ (reduce + cfs) (count cfs))
                sd (Math/sqrt (/ (reduce + (map #(let [d (- % mean)] (* d d)) cfs)) (count cfs)))]
            [rho {:n (count ok) :cf-mean (r3 mean) :cf-std (r3 sd)
                  :cf-min (r3 (apply min cfs)) :cf-max (r3 (apply max cfs))
                  :lip-mean (r3 (/ (reduce + (map :lip ok)) (count ok)))
                  :F-mean (r3 (/ (reduce + (map :F ok)) (count ok)))}]))))
(println "\n=== C/F across seeds ===")
(doseq [[rho s] summary] (println (format "rho=%.2f  %s" rho (pr-str s))))
(spit "scripts/cf_seeds_torch.edn" (with-out-str (pp/pprint {:rows rows :summary summary})))
(println "saved scripts/cf_seeds_torch.edn")
(System/exit 0)
