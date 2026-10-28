;; Extension of scripts/windowed_seed_replication.clj (2026-10-14), prompted
;; by its zero-carry result: frozen a-bar==0 solved the full curriculum in 4
;; of 5 seeds, but seed 2 failed EVERY stage (loss plateau ~1.8-2.7, chance
;; 2.77, never learns even one-hop k=2) while the learned-carry condition at
;; the same init solves everything -- a trainability failure, not a
;; memory-capacity one. Two parts, protocol identical to the replication
;; script (config, curriculum, batch-stream seeding, held-out eval 64 @
;; rng 7000+k):
;;   1. frozen-0.0 seeds 5-9 -> failure fraction on n=10 instead of n=5.
;;   2. seed-2 extended-budget probe: k=2 ALONE for 3x the stage budget
;;      (900 steps), accuracy logged every 100 steps -- distinguishes
;;      "stuck on a plateau" from "merely slow".
;; Writes its own .edn (does NOT touch windowed_seed_replication.edn, the
;; completed run of record); checkpointed after every completed run.
;; Smoke test: MYTHJURE_EXT_SMOKE=1 (10-step stages, 100-step probe);
;; MYTHJURE_EXT_SEEDS=n runs only the first n of the five seeds.
;; Run: clojure -M scripts/windowed_seed_ext.clj
(require '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.copytask :as copy]
         '[clojure.pprint :as pp])

(def SEQ 12) (def NC 14) (def T 8) (def W 1) (def BATCH 4) (def LR 1.5e-3)
(def D 32)
(def smoke? (some? (System/getenv "MYTHJURE_EXT_SMOKE")))
(def STAGES (if smoke?
              [[2 10] [3 10] [4 10] [5 10] [6 10]]
              [[2 300] [3 200] [4 200] [5 250] [6 400]]))
(def PROBE-STEPS (if smoke? 100 900))
(def ALL-K [2 3 4 5 6])
(def SEEDS (let [n (some-> (System/getenv "MYTHJURE_EXT_SEEDS") Long/parseLong)]
             (vec (take (or n 5) [5 6 7 8 9]))))
(def OUT "scripts/windowed_seed_ext.edn")

(defn init-model [seed log-A0]
  (model/init {:vocab-size 16 :d-model D :n-heads 4 :d-ff 128
               :seq-len SEQ :T T :seed seed :window W :log-A0 log-A0}))

(defn accuracy [m k n seed]
  (let [rng (java.util.Random. (long seed))]
    (/ (count (filter (fn [{:keys [input qpos answer]}]
                        (= answer (apply max-key
                                         (vec (nth (:probs (:cache (model/forward m input input))) qpos))
                                         (range 16))))
                      (copy/batch k SEQ NC n rng)))
       (double n))))

(defn acc-sweep [m] (into (sorted-map) (for [k ALL-K] [k (accuracy m k 64 (+ 7000 k))])))

;; sanity: a-bar really is exactly 0.0 at log-A = 10.0
(let [ab (model/effective-abar (init-model 0 10.0))]
  (assert (every? #(== 0.0 %) ab) (str "a-bar not exactly zero: " (take 3 ab)))
  (println "sanity: a-bar == 0.0 exactly at log-A = 10.0"))

(def results (atom {:config {:seq SEQ :nc NC :T T :w W :batch BATCH :lr LR :d D
                             :stages STAGES :seeds SEEDS :probe-steps PROBE-STEPS
                             :smoke smoke?}}))
(defn checkpoint! [] (spit OUT (with-out-str (pp/pprint @results))))

;; --- Part 1: frozen-0.0, seeds 5-9 -------------------------------------------
(defn frozen-run [seed]
  (println (format "\n########## frozen-0.0 seed %d ##########" seed))
  (let [rng (java.util.Random. (long seed))
        t0  (System/currentTimeMillis)
        fin (reduce
             (fn [{:keys [m opt results]} [k steps]]
               (let [{:keys [m opt]}
                     (reduce (fn [{:keys [m opt]} step]
                               (let [b (copy/batch k SEQ NC BATCH rng)
                                     {:keys [grads loss]} (train/batch-grad m b)
                                     grads (assoc grads :log-A (vec (repeat D 0.0)))
                                     opt (or opt (train/init-adam grads))
                                     [p' opt'] (train/adam-update (:params m) grads opt {:lr LR})]
                                 (when (zero? (mod step 100))
                                   (println (format "  [s%d] k=%d step %3d  loss %.3f" seed k step loss)))
                                 {:m (assoc m :params p') :opt opt'}))
                             {:m m :opt opt} (range 1 (inc steps)))
                     acc (accuracy m k 64 (+ 7000 k))]
                 (println (format "  [s%d] k=%d DONE acc=%.2f" seed k acc))
                 {:m m :opt opt :results (conj results {:k k :accuracy acc})}))
             {:m (init-model seed 10.0) :opt nil :results []} STAGES)
        final-acc (acc-sweep (:m fin))
        mins (/ (- (System/currentTimeMillis) t0) 60000.0)]
    (println (format "  [s%d] final-model acc sweep %s  (%.1f min)" seed (pr-str final-acc) mins))
    {:per-stage (:results fin) :final-acc final-acc :minutes mins}))

(doseq [seed SEEDS]
  (swap! results assoc-in [:frozen-0.0 seed] (frozen-run seed))
  (checkpoint!))

;; --- Part 2: seed-2 extended-budget probe, k=2 alone, 3x budget --------------
(println (format "\n########## seed-2 probe: frozen-0.0, k=2 alone, %d steps ##########" PROBE-STEPS))
(let [rng (java.util.Random. 2)
      t0  (System/currentTimeMillis)
      fin (reduce (fn [{:keys [m opt curve]} step]
                    (let [b (copy/batch 2 SEQ NC BATCH rng)
                          {:keys [grads loss]} (train/batch-grad m b)
                          grads (assoc grads :log-A (vec (repeat D 0.0)))
                          opt (or opt (train/init-adam grads))
                          [p' opt'] (train/adam-update (:params m) grads opt {:lr LR})
                          m' (assoc m :params p')]
                      (if (zero? (mod step 100))
                        (let [acc (accuracy m' 2 64 7002)]
                          (println (format "  [probe s2] step %4d  loss %.3f  acc %.3f" step loss acc))
                          {:m m' :opt opt' :curve (conj curve {:step step :loss loss :accuracy acc})})
                        {:m m' :opt opt' :curve curve})))
                  {:m (init-model 2 10.0) :opt nil :curve []}
                  (range 1 (inc PROBE-STEPS)))
      mins (/ (- (System/currentTimeMillis) t0) 60000.0)]
  (println (format "  [probe s2] done (%.1f min): %s" mins
                   (if (> (:accuracy (last (:curve fin)) 0.0) 0.7) "LEARNS WITH MORE BUDGET (slow)" "still at plateau (stuck)")))
  (swap! results assoc :seed2-probe {:curve (:curve fin) :minutes mins})
  (checkpoint!))

;; --- summary ------------------------------------------------------------------
(let [runs (get @results :frozen-0.0)
      solved? (fn [r] (every? #(>= (:accuracy %) 0.7) (:per-stage r)))
      n-solved (count (filter solved? (vals runs)))]
  (swap! results assoc :summary
         {:n-run (count runs) :n-all-stages-solved n-solved
          :per-seed-min-acc (into (sorted-map)
                                  (for [[s r] runs] [s (apply min (map :accuracy (:per-stage r)))]))})
  (checkpoint!)
  (println (format "\n=== ext summary: %d/%d new frozen-0.0 seeds solve all stages (>=0.7 per stage) ==="
                   n-solved (count runs)))
  (pp/pprint (:summary @results)))

(println (format "\n--- windowed-seed-ext done (wrote %s) ---" OUT))
(System/exit 0)
