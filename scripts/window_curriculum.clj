;; Curriculum sweep with WINDOWED (w=1) attention — the real cliff/equilibrium
;; test. One model, warm-started through k = 2,4,6,8,10. w=1 makes the loop carry
;; the ONLY mechanism that can bridge the gap (attention moves info 1 pos/loop),
;; so the learned max(ā) is a direct readout of the memory–stability tradeoff
;; under gradient pressure. T=8 ⇒ gaps up to ~7 are solvable; k≥8 needs τ>T
;; (cliff). Prediction to check: max(ā) ≈ 1−1/k (rising with k), and at the cliff
;; either dā flips (self-correcting) or ā is driven toward 1 (cliff).
;; Run:  clojure -M scripts/window_curriculum.clj
(require '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.copytask :as copy]
         '[clojure.pprint :as pp])

(def SEQ 14) (def NC 14) (def T 8) (def W 1)
(def BATCH 6) (def LR 1e-3)
(def STAGES [[2 350] [4 250] [6 250] [8 250] [10 250]])  ; [k steps]

(defn accuracy [m k n rng]
  (/ (count (filter (fn [{:keys [input qpos answer]}]
                      (= answer (apply max-key (vec (nth (:probs (:cache (model/forward m input input))) qpos))
                                       (range 16))))
                    (copy/batch k SEQ NC n rng)))
     (double n)))

(defn dabar-mean [m k rng]   ; sign/mag of dL/dā averaged over a batch + channels
  (let [b (copy/batch k SEQ NC BATCH rng)
        d (:dA-bar (train/batch-grad m b))]
    (/ (reduce + d) (count d))))

(println (format "windowed curriculum: w=%d T=%d seq=%d d=24 lr=%g batch=%d" W T SEQ LR BATCH))
(def m0 (model/init {:vocab-size 16 :d-model 24 :n-heads 4 :d-ff 96
                     :seq-len SEQ :T T :seed 0 :window W
                     :log-A0 (Math/log (- (Math/log 0.5)))}))
(def rng (java.util.Random. 0))
(def erng (java.util.Random. 999))

(def final
  (reduce
   (fn [{:keys [m opt results]} [k steps]]
     (let [pred (- 1.0 (/ 1.0 k))]
       (println (format "\n=== stage k=%d  steps=%d  predicted max(ā)=%.3f %s ==="
                        k steps pred (if (>= k 8) "[τ≥T: cliff region]" "")))
       (let [{:keys [m opt]}
             (reduce (fn [{:keys [m opt]} step]
                       (let [b (copy/batch k SEQ NC BATCH rng)
                             {:keys [loss grads]} (train/batch-grad m b)
                             opt (or opt (train/init-adam grads))
                             [p' opt'] (train/adam-update (:params m) grads opt {:lr LR})
                             m' (assoc m :params p')]
                         (when (zero? (mod step 50))
                           (let [ab (model/a-bar (get-in m' [:params :log-A]))]
                             (println (format "  k=%d step %3d  loss %.4f  acc %.2f  ā[max %.3f mean %.3f]"
                                              k step loss (accuracy m' k 32 erng)
                                              (apply max ab) (/ (reduce + ab) (count ab))))))
                         {:m m' :opt opt'}))
                     {:m m :opt opt} (range 1 (inc steps)))
             ab (model/a-bar (get-in m [:params :log-A]))
             acc (accuracy m k 64 erng)
             dabar (dabar-mean m k rng)
             res {:k k :predicted pred :max-abar (apply max ab)
                  :mean-abar (/ (reduce + ab) (count ab))
                  :top4 (vec (take 4 (sort > ab))) :accuracy acc :dL-dabar-mean dabar}]
         (println (format "k=%d DONE  acc=%.2f  max(ā)=%.3f  predicted=%.3f  Δ=%+.3f  dL/dā(mean)=%+.4f"
                          k acc (apply max ab) pred (- (apply max ab) pred) dabar))
         {:m m :opt opt :results (conj results res)})))
   {:m m0 :opt nil :results []} STAGES))

(println "\n========= SUMMARY: windowed (w=1) max(ā) vs prediction 1−1/k =========")
(doseq [{:keys [k predicted max-abar accuracy dL-dabar-mean]} (:results final)]
  (println (format "k=%2d  predicted %.3f   learned %.3f   Δ %+.3f   acc %.2f   dL/dā %+.4f"
                   k predicted max-abar (- max-abar predicted) accuracy dL-dabar-mean)))
(spit "scripts/window_curriculum.edn" (with-out-str (pp/pprint (:results final))))
(println "\n--- saved scripts/window_curriculum.edn ---")
(System/exit 0)
