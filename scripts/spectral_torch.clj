;; Torch-backend rerun of scripts/spectral.clj (§5.3 + App. A): the same four
;; measurements (spectral-analysis, injection-ratio, cf-analysis,
;; spectral-seed-sweep — same seeds, same schedules, float64), through
;; mythjure.analysis-torch's tensor-native hot core. Writes
;; scripts/spectral_torch.edn and DIFFS every value against the committed
;; scripts/spectral.edn (the paper's numbers): rounded values must be equal,
;; raw residuals equal to 1e-4 relative.
;; Run:  clojure -M:torch scripts/spectral_torch.clj   (Python w/ torch auto-discovered; MYTHJURE_PYTHON/MYTHJURE_LIBPYTHON override)
(require '[mythjure.block :as blk]
         '[mythjure.linalg :as la]
         '[mythjure.torch.core :as tc]
         '[mythjure.analysis-torch :as ant]
         '[clojure.edn :as edn]
         '[clojure.pprint :as pp])

(tc/initialize!)

(defn deep-close?
  "Structural equality with numeric tolerance (relative 1e-4 for doubles)."
  [a b]
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

(def p (blk/init-params {:d-model 32 :n-heads 4 :seed 7}))
(def E (la/rand-matrix 99 16 32 1.0))

(defn timed [label f]
  (let [t0 (System/nanoTime) r (f)]
    (println (format "  %-22s %6.1f s" label (/ (- (System/nanoTime) t0) 1e9)))
    r))

(println "=== spectral suite, torch backend (float64) ===")
(def spec (timed "spectral-analysis" #(ant/spectral-analysis p E)))
(def inj  (timed "injection-ratio"   #(ant/injection-ratio p E)))
(def cf   (timed "cf-analysis"       #(ant/cf-analysis p E)))
(def sweep (timed "spectral-seed-sweep" #(ant/spectral-seed-sweep)))

(def out {:spectral-analysis spec :injection-ratio inj
          :cf-analysis cf :seed-robustness sweep})
(spit "scripts/spectral_torch.edn" (with-out-str (pp/pprint out)))
(println "saved scripts/spectral_torch.edn")

(println "\n=== diff vs committed scripts/spectral.edn (paper numbers) ===")
(def committed (edn/read-string (slurp "scripts/spectral.edn")))
(def failures (atom 0))
(doseq [k [:spectral-analysis :injection-ratio :cf-analysis :seed-robustness]]
  (let [ok? (deep-close? (get committed k) (get out k))]
    (when-not ok? (swap! failures inc))
    (println (format "  %-20s %s" (name k) (if ok? "MATCH" "*** MISMATCH ***")))))
(if (zero? @failures)
  (println "ALL SECTIONS REPRODUCE THE COMMITTED RESULTS.")
  (println @failures "section(s) differ — inspect spectral_torch.edn."))
(System/exit (if (zero? @failures) 0 1))
