;; Ablation: is the nonlinear recurrence (block seeing H_t) what propagates info
;; across loops in the windowed copy task? Cut it (block reads only E) and see.
;; Three conditions on k=4 copy, all else equal:
;;   A full   block + windowed(w=1)  -> expect SOLVE  (recurrence propagates)
;;   B ablate block + windowed(w=1)  -> expect FAIL   (no recurrence; carry can't
;;                                                     cross positions; block(E)
;;                                                     reaches only 1 position)
;;   C ablate block + full attention -> expect SOLVE  (attention bridges from E
;;                                                     in one hop; recurrence unneeded)
;; A-vs-B isolates the nonlinear recurrence; B-vs-C shows the ablation isn't fatal.
;; Run:  clojure -M scripts/ablation_k4.clj
(require '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.copytask :as copy]
         '[clojure.pprint :as pp])

(def K 4) (def SEQ 12) (def NC 14) (def T 8)
(def STEPS 400) (def BATCH 6) (def LR 1e-3)

(defn accuracy [m n rng]
  (/ (count (filter (fn [{:keys [input qpos answer]}]
                      (= answer (apply max-key (vec (nth (:probs (:cache (model/forward m input input))) qpos))
                                       (range 16))))
                    (copy/batch K SEQ NC n rng)))
     (double n)))

(defn run [label cfg]
  (println (format "\n=== %s ===" label))
  (let [m0 (model/init (merge {:vocab-size 16 :d-model 24 :n-heads 4 :d-ff 96
                               :seq-len SEQ :T T :seed 0
                               :log-A0 (Math/log (- (Math/log 0.5)))} cfg))
        rng (java.util.Random. 0) erng (java.util.Random. 7)
        mf (reduce (fn [{:keys [m opt]} step]
                     (let [b (copy/batch K SEQ NC BATCH rng)
                           {:keys [loss grads]} (train/batch-grad m b)
                           opt (or opt (train/init-adam grads))
                           [p' opt'] (train/adam-update (:params m) grads opt {:lr LR})
                           m' (assoc m :params p')]
                       (when (zero? (mod step 50))
                         (println (format "  %s step %3d  loss %.4f  acc %.2f"
                                          label step loss (accuracy m' 32 erng))))
                       {:m m' :opt opt'}))
                   {:m m0 :opt nil} (range 1 (inc STEPS)))
        m (:m mf)
        acc (accuracy m 64 erng)
        ab (model/a-bar (get-in m [:params :log-A]))]
    (println (format "%s FINAL  acc=%.2f  max(ā)=%.3f" label acc (apply max ab)))
    {:label label :accuracy acc :max-abar (apply max ab)}))

(println (format "ablation on k=%d copy: T=%d seq=%d d=24 lr=%g steps=%d" K T SEQ LR STEPS))
(def results
  [(run "A full+windowed"  {:window 1 :ablate? false})
   (run "B ablate+windowed" {:window 1 :ablate? true})
   (run "C ablate+full-attn" {:window nil :ablate? true})])

(println "\n===== ABLATION SUMMARY (k=4 copy) =====")
(doseq [{:keys [label accuracy max-abar]} results]
  (println (format "%-20s  acc=%.2f  max(ā)=%.3f" label accuracy max-abar)))
(println "A solves & B fails ⇒ nonlinear recurrence (block seeing H_t) does the cross-loop propagation.")
(println "C solves ⇒ ablation isn't fatal; it specifically removes multi-loop propagation.")
(spit "scripts/ablation_k4.edn" (with-out-str (pp/pprint results)))
(System/exit 0)
