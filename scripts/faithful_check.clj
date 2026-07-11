;; Robustness check: does the paper's CENTRAL finding (§7.4 — under windowed
;; attention the carry stays flat and dL/dā is anti-cliff, i.e. the nonlinear
;; block, not the linear carry, provides depth memory) hold under the FAITHFUL
;; Parcae architecture (learnable Δ + B̄e injection + prelude LN)? One warm-started
;; model through k=2→4 (as in §7.4). Reference (simple mode): k2 ā=0.533 dL/dā=+7e-4;
;; k4 ā=0.532 dL/dā=+3e-3.
;; Run:  clojure -M scripts/faithful_check.clj
(require '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.copytask :as copy])

(def SEQ 10) (def NC 14) (def T 8) (def W 1) (def BATCH 6) (def LR 1e-3)
(def STAGES [[2 300] [4 300]])

(defn accuracy [m k n rng]
  (/ (count (filter (fn [{:keys [input qpos answer]}]
                      (= answer (apply max-key (vec (nth (:probs (:cache (model/forward m input input))) qpos)) (range 16))))
                    (copy/batch k SEQ NC n rng)))
     (double n)))

(println "FAITHFUL (learnable Δ + B̄ + prelude LN) + windowed(w=1) copy curriculum")
(def m0 (model/init {:vocab-size 16 :d-model 24 :n-heads 4 :d-ff 96
                     :seq-len SEQ :T T :seed 0 :window W :faithful? true
                     :log-A0 (Math/log (- (Math/log 0.5)))}))   ; ā₀=0.5
(def rng (java.util.Random. 0)) (def erng (java.util.Random. 7))

(reduce
 (fn [{:keys [m opt]} [k steps]]
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
         ab (model/effective-abar m)
         {:keys [dA-bar]} (train/batch-grad m (copy/batch k SEQ NC BATCH rng))]
     (println (format "k=%d DONE  acc=%.2f  max(ā)=%.3f  mean dL/dā=%+.4f  n(dL/dā>0)=%d/%d  %s"
                      k (accuracy m k 64 erng) (apply max ab)
                      (/ (reduce + dA-bar) (count dA-bar))
                      (count (filter pos? dA-bar)) (count dA-bar)
                      (if (pos? (/ (reduce + dA-bar) (count dA-bar))) "ANTI-CLIFF ✓" "toward-cliff")))
     {:m m :opt opt}))
 {:m m0 :opt nil} STAGES)

(println "\n--- faithful robustness check done ---")
(System/exit 0)
