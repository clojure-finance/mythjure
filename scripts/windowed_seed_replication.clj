;; §H-2 of attic/paper-v2-plan.md (venue-review item 2): the training
;; experiments are single-seed; replicate the windowed-copy run of record
;; across training seeds. Two conditions, the ones carrying seed-sensitive
;; claims:
;;   :learned     — the task solves k=2–6 AND training moves the carry DOWN,
;;                  not toward the 1−1/k "carry recruitment" prediction
;;                  (max ā tracked per stage; mean dL/dā measured on a
;;                  held-aside batch);
;;   :frozen-0.0  — the headline condition (log-A=10 ⇒ ā underflows to
;;                  exactly 0.0, no linear carry at all) solves every stage —
;;                  "no necessary depth memory".
;; Seeds 0–4. The seed varies BOTH the model init and the training batch
;; stream; the held-out eval protocol (64 examples, rng seed 7000+k) is held
;; fixed across seeds for comparability. Seed 0 doubles as the
;; reproduce-the-record control: :learned against carry_intervention.edn
;; :learned, :frozen-0.0 against carry_zero.edn :frozen-0.0 (compared
;; automatically when those files are present; pure-Clojure determinism ⇒
;; match expected, compared at 1e-6). The dL/dā measurement uses its own rng
;; so the training stream is untouched and the control stays exact.
;; Checkpointed: the .edn is rewritten after every completed
;; (condition, seed) run — a killed run loses at most one.
;; Smoke test: MYTHJURE_REPL_SMOKE=1 shrinks every stage to 10 steps;
;; MYTHJURE_REPL_SEEDS=n runs only the first n seeds.
;; Run: clojure -M scripts/windowed_seed_replication.clj
(require '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.copytask :as copy]
         '[clojure.pprint :as pp]
         '[clojure.edn :as edn]
         '[clojure.java.io :as io])

(def SEQ 12) (def NC 14) (def T 8) (def W 1) (def BATCH 4) (def LR 1.5e-3)
(def D 32)
(def smoke? (some? (System/getenv "MYTHJURE_REPL_SMOKE")))
(def STAGES (if smoke?
              [[2 10] [3 10] [4 10] [5 10] [6 10]]
              [[2 300] [3 200] [4 200] [5 250] [6 400]]))
(def ALL-K [2 3 4 5 6])
(def SEEDS (let [n (some-> (System/getenv "MYTHJURE_REPL_SEEDS") Long/parseLong)]
             (vec (take (or n 5) [0 1 2 3 4]))))
(def OUT "scripts/windowed_seed_replication.edn")

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

(defn train-run
  "One full curriculum at `seed`. freeze? ⇒ zero the :log-A grad every step.
  measure-grad? ⇒ record mean dL/dā at each stage end on a held-aside batch
  (own rng, training stream untouched)."
  [label seed log-A0 freeze? measure-grad?]
  (println (format "\n########## %s seed %d (log-A0 %.4f, freeze? %s) ##########"
                   label seed log-A0 freeze?))
  (let [rng (java.util.Random. (long seed))
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
                                   (println (format "  [%s s%d] k=%d step %3d  loss %.3f  max(ā) %.6f"
                                                    label seed k step loss
                                                    (apply max (model/effective-abar m)))))
                                 {:m (assoc m :params p') :opt opt'}))
                             {:m m :opt opt} (range 1 (inc steps)))
                     acc (accuracy m k 64 (+ 7000 k))
                     ab  (model/effective-abar m)
                     md  (when measure-grad?
                           (let [{:keys [dA-bar]} (train/batch-grad
                                                   m (copy/batch k SEQ NC BATCH
                                                                 (java.util.Random. (+ 9000 k))))]
                             (/ (reduce + dA-bar) (count dA-bar))))]
                 (println (format "  [%s s%d] k=%d DONE acc=%.2f max(ā)=%.6f%s"
                                  label seed k acc (apply max ab)
                                  (if md (format "  mean dL/dā %+.4f" md) "")))
                 {:m m :opt opt
                  :results (conj results
                                 (cond-> {:k k :accuracy acc
                                          :max-abar (apply max ab)
                                          :mean-abar (/ (reduce + ab) (count ab))}
                                   md (assoc :dL-dabar-mean md)))}))
             {:m (init-model seed log-A0) :opt nil :results []} STAGES)
        final-acc (acc-sweep (:m fin))
        mins (/ (- (System/currentTimeMillis) t0) 60000.0)]
    (println (format "  [%s s%d] final-model acc sweep %s  (%.1f min)"
                     label seed (pr-str final-acc) mins))
    {:per-stage (:results fin) :final-acc final-acc :minutes mins}))

;; sanity: ā really is exactly 0.0 at log-A = 10.0
(let [ab (model/effective-abar (init-model 0 10.0))]
  (assert (every? #(== 0.0 %) ab) (str "ā not exactly zero: " (take 3 ab)))
  (println "sanity: ā == 0.0 exactly at log-A = 10.0"))

(def a05 (Math/log (- (Math/log 0.5))))     ; ā = 0.5 (the record's init)

(def results (atom {:config {:seq SEQ :nc NC :T T :w W :batch BATCH :lr LR
                             :d D :stages STAGES :seeds SEEDS :smoke smoke?}}))
(defn checkpoint! [] (spit OUT (with-out-str (pp/pprint @results))))

(doseq [seed SEEDS
        [cond-key label log-A0 freeze? grad?] [[:learned "learned" a05 false true]
                                               [:frozen-0.0 "frozen-0.0" 10.0 true false]]]
  (swap! results assoc-in [cond-key seed]
         (dissoc (train-run label seed log-A0 freeze? grad?) :m))
  (checkpoint!))

;; --- seed-0 control vs the records (skip under smoke: stages differ) ---------
(defn control-check [record-file record-path got-rows]
  (if (and (not smoke?) (some #{0} SEEDS) (.exists (io/file record-file)))
    (let [rec-rows (get-in (edn/read-string (slurp record-file)) record-path)
          close? (fn [a b] (< (Math/abs (- (double a) (double b))) 1e-6))
          row-ok? (fn [r g] (every? #(close? (% r) (% g)) [:accuracy :max-abar :mean-abar]))
          ok? (and (= (count rec-rows) (count got-rows))
                   (every? identity (map row-ok? rec-rows got-rows)))]
      {:file record-file :match (if ok? :MATCH :MISMATCH)
       :record (vec rec-rows)
       :got (mapv #(select-keys % [:k :accuracy :max-abar :mean-abar]) got-rows)})
    {:file record-file :match :skipped}))

(let [checks {:learned    (control-check "scripts/carry_intervention.edn" [:learned :per-stage]
                                         (get-in @results [:learned 0 :per-stage]))
              :frozen-0.0 (control-check "scripts/carry_zero.edn" [:frozen-0.0 :per-stage]
                                         (get-in @results [:frozen-0.0 0 :per-stage]))}]
  (swap! results assoc :control-check checks)
  (doseq [[k v] checks] (println (format "control %s: %s (%s)" (name k) (:match v) (:file v)))))

;; --- cross-seed summary -------------------------------------------------------
(defn summarize [cond-key]
  (let [runs (vals (get @results cond-key))
        stage-accs (fn [k] (mapv (fn [r] (:accuracy (first (filter #(= k (:k %)) (:per-stage r))))) runs))
        max-abars (mapv (fn [r] (apply max (map :max-abar (:per-stage r)))) runs)]
    {:per-stage-acc (into (sorted-map) (for [k ALL-K] [k (stage-accs k)]))
     :min-stage-acc (apply min (for [k ALL-K, a (stage-accs k)] a))
     :max-abar-range [(apply min max-abars) (apply max max-abars)]}))

(let [summary {:learned (summarize :learned) :frozen-0.0 (summarize :frozen-0.0)}]
  (swap! results assoc :summary summary)
  (checkpoint!)
  (println "\n=== cross-seed summary ===")
  (pp/pprint summary))

(println (format "\n--- windowed-seed-replication done (wrote %s) ---" OUT))
(System/exit 0)
