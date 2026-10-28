;; v2-review follow-up (2026-10-04, review-2 pt 6): does the trained model's
;; accuracy survive iterating PAST its training depth T=8, toward the fixed
;; point it settles at (56-64 iterations on k=6 inputs, per the E4
;; measurement)? The T=8 operating state is locally expansive (rho_spec
;; 1.24/1.66) while the fixed point is contractive (0.72) — so does the
;; computation live in the transient, and if so, does settling destroy it?
;; Loads the trained condition-(a) params saved by carry_intervention.clj;
;; held-out accuracy protocol identical (64 examples, seeds 7000+k).
;; Run: clojure -M scripts/depth_accuracy.clj
(require '[mythjure.model :as model]
         '[mythjure.copytask :as copy]
         '[clojure.edn :as edn]
         '[clojure.pprint :as pp])

(def SEQ 12) (def NC 14) (def W 1) (def D 32)
(def ALL-K [2 3 4 5 6])
(def TS [2 4 6 8 12 16 24 32 48 64])

(defn init-model [log-A0]
  (model/init {:vocab-size 16 :d-model D :n-heads 4 :d-ff 128
               :seq-len SEQ :T 8 :seed 0 :window W :log-A0 log-A0}))

(defn accuracy [m k n seed]
  (let [rng (java.util.Random. (long seed))]
    (/ (count (filter (fn [{:keys [input qpos answer]}]
                        (= answer (apply max-key
                                         (vec (nth (:probs (:cache (model/forward m input input))) qpos))
                                         (range 16))))
                      (copy/batch k SEQ NC n rng)))
       (double n))))

(defn acc-sweep [m] (into (sorted-map) (for [k ALL-K] [k (accuracy m k 64 (+ 7000 k))])))

(def a05 (Math/log (- (Math/log 0.5))))
(def params (edn/read-string (slurp "scripts/carry_intervention_model_a.edn")))
(def m0 (assoc (init-model a05) :params params))

;; reproduce the recorded T=8 final-acc sweep as a load sanity check
(println "T=8 sanity (record: {2 0.15625, 3 0.078125, 4 0.125, 5 0.484375, 6 1.0}):")
(println "  " (pr-str (acc-sweep m0)))

(def sweep
  (into (sorted-map)
        (for [T TS]
          (let [acc (acc-sweep (assoc-in m0 [:config :T] T))]
            (println (format "T=%2d  acc %s" T (pr-str acc)))
            [T acc]))))

(spit "scripts/depth_accuracy.edn"
      (with-out-str (pp/pprint {:protocol {:n 64 :seeds "7000+k" :model "carry_intervention_model_a"}
                                :by-T sweep})))
(println "\n--- depth-accuracy done (wrote scripts/depth_accuracy.edn) ---")
(System/exit 0)
