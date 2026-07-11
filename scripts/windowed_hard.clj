;; Remove the k≥6 learnability confound from §7.4: solve the SAME windowed (w=1)
;; copy task at k=6 via a FINER curriculum (step by 1, for transfer), a bigger
;; model (d=32), and more steps. If k=6 solves, we can read max(ā)/dL/dā in the
;; high-predicted-carry regime (k=6 ⇒ 1−1/k=0.833) unconfounded: does ā stay flat
;; + dL/dā anti-cliff (finding EXTENDS past k=4), or does ā climb (cliff APPEARS)?
;; Run:  clojure -M scripts/windowed_hard.clj
(require '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.copytask :as copy]
         '[clojure.pprint :as pp])

(def SEQ 12) (def NC 14) (def T 8) (def W 1) (def BATCH 4) (def LR 1.5e-3)
(def STAGES [[2 300] [3 200] [4 200] [5 250] [6 400]])   ; finer, step by 1

(defn accuracy [m k n rng]
  (/ (count (filter (fn [{:keys [input qpos answer]}]
                      (= answer (apply max-key (vec (nth (:probs (:cache (model/forward m input input))) qpos)) (range 16))))
                    (copy/batch k SEQ NC n rng)))
     (double n)))

(println "windowed(w=1) copy, FINER curriculum, d=32 — removing the k≥6 confound")
(def m0 (model/init {:vocab-size 16 :d-model 32 :n-heads 4 :d-ff 128
                     :seq-len SEQ :T T :seed 0 :window W
                     :log-A0 (Math/log (- (Math/log 0.5)))}))
(def rng (java.util.Random. 0)) (def erng (java.util.Random. 7))

(def final
  (reduce
   (fn [{:keys [m opt results]} [k steps]]
     (println (format "\n=== k=%d  predicted(if carry=mechanism) max(ā)=%.3f ===" k (- 1.0 (/ 1.0 k))))
     (let [{:keys [m opt]}
           (reduce (fn [{:keys [m opt]} step]
                     (let [b (copy/batch k SEQ NC BATCH rng)
                           {:keys [loss grads dA-bar]} (train/batch-grad m b)
                           opt (or opt (train/init-adam grads))
                           [p' opt'] (train/adam-update (:params m) grads opt {:lr LR})
                           m' (assoc m :params p')]
                       (when (zero? (mod step 100))
                         (println (format "  k=%d step %3d  loss %.3f  acc %.2f  max(ā) %.3f  mean dL/dā %+.4f"
                                          k step loss (accuracy m' k 32 erng)
                                          (apply max (model/effective-abar m'))
                                          (/ (reduce + dA-bar) (count dA-bar)))))
                       {:m m' :opt opt'}))
                   {:m m :opt opt} (range 1 (inc steps)))
           acc (accuracy m k 64 erng)
           ab (model/effective-abar m)
           {:keys [dA-bar]} (train/batch-grad m (copy/batch k SEQ NC BATCH rng))
           md (/ (reduce + dA-bar) (count dA-bar))]
       (println (format "k=%d DONE  acc=%.2f  max(ā)=%.3f  pred=%.3f  mean dL/dā=%+.4f  %s%s"
                        k acc (apply max ab) (- 1.0 (/ 1.0 k)) md
                        (if (pos? md) "ANTI-CLIFF" "toward-cliff")
                        (if (> acc 0.7) "  [SOLVED — unconfounded]" "  [unsolved]")))
       {:m m :opt opt
        :results (conj results {:k k
                                :predicted (- 1.0 (/ 1.0 k))
                                :max-abar (apply max ab)
                                :mean-abar (/ (reduce + ab) (count ab))
                                :accuracy acc
                                :dL-dabar-mean md})}))
   {:m m0 :opt nil :results []} STAGES))

(spit "scripts/windowed_hard.edn" (with-out-str (pp/pprint (:results final))))
(println "\n--- windowed-hard done (wrote scripts/windowed_hard.edn) ---")
(System/exit 0)
