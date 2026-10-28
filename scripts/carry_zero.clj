;; v2-review follow-up (2026-10-04, review-2 pt 1): condition (e) of the carry
;; intervention — train with the carry frozen at EXACTLY zero. log-A = 10.0
;; gives a-bar = exp(-exp(10)), which underflows to 0.0 in doubles, so the
;; recurrence is h_{t+1} = f(h_t, e): no linear carry at all. The reviewer's
;; point: "in a normalized system a numerically small pathway is not
;; automatically functionally negligible" — 0.02 tested the floor, not zero.
;; Protocol identical to scripts/carry_intervention.clj (config, curriculum,
;; batch-stream seed, held-out accuracy protocol all unchanged).
;; Run: clojure -M scripts/carry_zero.clj
(require '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.copytask :as copy]
         '[clojure.pprint :as pp])

(def SEQ 12) (def NC 14) (def T 8) (def W 1) (def BATCH 4) (def LR 1.5e-3)
(def D 32)
(def STAGES [[2 300] [3 200] [4 200] [5 250] [6 400]])
(def ALL-K [2 3 4 5 6])

(defn init-model [log-A0]
  (model/init {:vocab-size 16 :d-model D :n-heads 4 :d-ff 128
               :seq-len SEQ :T T :seed 0 :window W :log-A0 log-A0}))

(defn accuracy [m k n seed]
  (let [rng (java.util.Random. (long seed))]
    (/ (count (filter (fn [{:keys [input qpos answer]}]
                        (= answer (apply max-key
                                         (vec (nth (:probs (:cache (model/forward m input input))) qpos))
                                         (range 16))))
                      (copy/batch k SEQ NC n rng)))
       (double n))))

(defn acc-sweep [m] (into (sorted-map) (for [k ALL-K] [k (accuracy m k 64 (+ 7000 k))])))

(defn train-condition
  [label log-A0 freeze?]
  (println (format "\n########## condition %s (log-A0 %.4f, freeze? %s) ##########" label log-A0 freeze?))
  (let [rng (java.util.Random. 0)
        t0  (System/currentTimeMillis)
        fin (reduce
             (fn [{:keys [m opt results]} [k steps]]
               (let [{:keys [m opt]}
                     (reduce (fn [{:keys [m opt]} step]
                               (let [b (copy/batch k SEQ NC BATCH rng)
                                     {:keys [grads loss]} (train/batch-grad m b)
                                     grads (if freeze? (assoc grads :log-A (vec (repeat D 0.0))) grads)
                                     opt (or opt (train/init-adam grads))
                                     [p' opt'] (train/adam-update (:params m) grads opt {:lr LR})]
                                 (when (zero? (mod step 100))
                                   (println (format "  [%s] k=%d step %3d  loss %.3f  max(a-bar) %.6f"
                                                    label k step loss
                                                    (apply max (model/effective-abar m)))))
                                 {:m (assoc m :params p') :opt opt'}))
                             {:m m :opt opt} (range 1 (inc steps)))
                     acc (accuracy m k 64 (+ 7000 k))
                     ab  (model/effective-abar m)]
                 (println (format "  [%s] k=%d DONE acc=%.2f max(a-bar)=%.6f" label k acc (apply max ab)))
                 {:m m :opt opt
                  :results (conj results {:k k :accuracy acc
                                          :max-abar (apply max ab)
                                          :mean-abar (/ (reduce + ab) (count ab))})}))
             {:m (init-model log-A0) :opt nil :results []} STAGES)
        final-acc (acc-sweep (:m fin))]
    (println (format "  [%s] final-model acc sweep %s  (%.1f min)"
                     label (pr-str final-acc) (/ (- (System/currentTimeMillis) t0) 60000.0)))
    {:m (:m fin) :per-stage (:results fin) :final-acc final-acc}))

;; sanity: a-bar really is exactly 0.0 at log-A = 10.0
(let [ab (model/effective-abar (init-model 10.0))]
  (assert (every? #(== 0.0 %) ab) (str "a-bar not exactly zero: " (take 3 ab)))
  (println "sanity: a-bar == 0.0 exactly at log-A = 10.0"))

(def cond-e (train-condition "frozen-0.0" 10.0 true))

(def out {:config {:seq SEQ :nc NC :T T :w W :batch BATCH :lr LR :d D :stages STAGES}
          :frozen-0.0 {:per-stage (:per-stage cond-e) :final-acc (:final-acc cond-e)}})
(spit "scripts/carry_zero.edn" (with-out-str (pp/pprint out)))
(println "\n--- carry-zero done (wrote scripts/carry_zero.edn) ---")
(System/exit 0)
