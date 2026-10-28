;; Seed-2 rescue probe (2026-10-14). Background: in the zero-carry
;; (a-bar == 0) replication, seed 2 is the single failure in 10 -- stuck on a
;; loss plateau from stage one (k=2, one hop, minimal memory demand), still at
;; chance after 3x budget (windowed_seed_ext.clj probe), while the LEARNED
;; carry at the same init solves everything. Hypothesis: optimization failure
;; (the carry acts as a depth-axis skip connection easing descent), so
;; standard remedies may rescue it. Two probes, k=2 only (the failing stage),
;; zero carry (log-A = 10 => a-bar == 0.0 exactly), eval 64 held-out examples
;; (rng 7002) every 100 steps:
;;   1. DIAGNOSTIC at the record lr 1.5e-3, stage budget (300 steps): our
;;      "seed" sets model init AND batch stream together; cross the factors
;;      (init 2 x stream 102, init 102 x stream 2) to see which one carries
;;      the failure. (init 2 x stream 2 = known stuck, not rerun; init 102 is
;;      a fresh draw outside the tested seeds.)
;;   2. LR GRID at init 2 x stream 2: lr in {3e-4, 7.5e-4, 3e-3}, double
;;      budget (600 steps) so slower rates get a fair shot.
;; Verdict per run: RESCUED if any eval reaches acc >= 0.7 (solve threshold
;; of the record); else still stuck. Checkpointed after every run.
;; Smoke test: MYTHJURE_RESCUE_SMOKE=1 (20-step runs).
;; Run: clojure -M scripts/seed2_rescue.clj
(require '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.copytask :as copy]
         '[clojure.pprint :as pp])

(def SEQ 12) (def NC 14) (def T 8) (def W 1) (def BATCH 4)
(def D 32)
(def smoke? (some? (System/getenv "MYTHJURE_RESCUE_SMOKE")))
(def DIAG-STEPS (if smoke? 20 300))
(def GRID-STEPS (if smoke? 20 600))
(def OUT "scripts/seed2_rescue.edn")

(defn init-model [seed]
  (model/init {:vocab-size 16 :d-model D :n-heads 4 :d-ff 128
               :seq-len SEQ :T T :seed seed :window W :log-A0 10.0}))

(defn accuracy [m]
  (let [rng (java.util.Random. 7002)]
    (/ (count (filter (fn [{:keys [input qpos answer]}]
                        (= answer (apply max-key
                                         (vec (nth (:probs (:cache (model/forward m input input))) qpos))
                                         (range 16))))
                      (copy/batch 2 SEQ NC 64 rng)))
       64.0)))

(let [ab (model/effective-abar (init-model 0))]
  (assert (every? #(== 0.0 %) ab) "a-bar not exactly zero at log-A = 10.0"))
(println "sanity: a-bar == 0.0 exactly")

(defn k2-run [label init-seed stream-seed lr steps]
  (println (format "\n##### %s (init %d, stream %d, lr %.1e, %d steps) #####"
                   label init-seed stream-seed lr steps))
  (let [rng (java.util.Random. (long stream-seed))
        t0  (System/currentTimeMillis)
        fin (reduce (fn [{:keys [m opt curve]} step]
                      (let [b (copy/batch 2 SEQ NC BATCH rng)
                            {:keys [grads loss]} (train/batch-grad m b)
                            grads (assoc grads :log-A (vec (repeat D 0.0)))
                            opt (or opt (train/init-adam grads))
                            [p' opt'] (train/adam-update (:params m) grads opt {:lr lr})
                            m' (assoc m :params p')]
                        (if (zero? (mod step 100))
                          (let [acc (accuracy m')]
                            (println (format "  [%s] step %4d  loss %.3f  acc %.3f" label step loss acc))
                            {:m m' :opt opt' :curve (conj curve {:step step :loss loss :accuracy acc})})
                          {:m m' :opt opt' :curve curve})))
                    {:m (init-model init-seed) :opt nil :curve []}
                    (range 1 (inc steps)))
        curve (:curve fin)
        best (if (seq curve) (apply max (map :accuracy curve)) (accuracy (:m fin)))
        rescued? (>= best 0.7)
        mins (/ (- (System/currentTimeMillis) t0) 60000.0)]
    (println (format "  [%s] done (%.1f min): best acc %.3f  %s"
                     label mins best (if rescued? "RESCUED" "still stuck")))
    {:init init-seed :stream stream-seed :lr lr :steps steps
     :curve curve :best-acc best :rescued? rescued? :minutes mins}))

(def results (atom {:config {:seq SEQ :nc NC :T T :w W :batch BATCH :d D
                             :diag-steps DIAG-STEPS :grid-steps GRID-STEPS
                             :smoke smoke?
                             :baseline-note "init 2 x stream 2 at lr 1.5e-3 = known stuck (windowed_seed_replication + ext probe)"}}))
(defn checkpoint! [] (spit OUT (with-out-str (pp/pprint @results))))

;; Part 1: factor diagnostic at the record lr
(doseq [[label is ss] [["diag-init2-stream102" 2 102]
                       ["diag-init102-stream2" 102 2]]]
  (swap! results assoc-in [:diagnostic label] (k2-run label is ss 1.5e-3 DIAG-STEPS))
  (checkpoint!))

;; Part 2: lr grid on the failing combination
(doseq [lr [3e-4 7.5e-4 3e-3]]
  (swap! results assoc-in [:lr-grid lr] (k2-run (format "grid-lr%.1e" lr) 2 2 lr GRID-STEPS))
  (checkpoint!))

(let [diag (get @results :diagnostic)
      grid (get @results :lr-grid)]
  (println "\n=== seed-2 rescue summary ===")
  (doseq [[l r] diag] (println (format "  %s: best %.3f %s" l (:best-acc r) (if (:rescued? r) "RESCUED" "stuck"))))
  (doseq [[lr r] (sort grid)] (println (format "  lr %.1e: best %.3f %s" lr (:best-acc r) (if (:rescued? r) "RESCUED" "stuck")))))

(println (format "\n--- seed2-rescue done (wrote %s) ---" OUT))
(System/exit 0)
