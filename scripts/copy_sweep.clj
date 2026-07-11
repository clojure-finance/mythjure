;; Sweep the copy-task gap k ∈ {2,6,8,12} at T=8 and watch whether the optimizer
;; drives max(ā) toward the prediction 1−1/k. k=12 (τ>T) probes the cliff.
;; Run from project root:  clojure -M scripts/copy_sweep.clj
(require '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.copytask :as copy]
         '[clojure.pprint :as pp])

(def SEQ-LEN 16)
(def N-CONTENT 14)        ; content tokens 2..15, vocab 16
(def T 8)
(def STEPS 120)
(def BATCH 6)
(def LR 1e-2)

(defn accuracy [m k rng]
  (let [b (copy/batch k SEQ-LEN N-CONTENT BATCH rng)]
    (/ (count (filter (fn [{:keys [input qpos answer]}]
                        (let [probs (:probs (:cache (model/forward m input input)))
                              pred (apply max-key (vec (nth probs qpos)) (range 16))]
                          (= pred answer)))
                      b))
       (double (count b)))))

(defn top-abars [m n]
  (->> (model/a-bar (get-in m [:params :log-A])) (sort >) (take n) vec))

(defn run-k [k]
  (let [rng (java.util.Random. (+ 100 k))
        ;; neutral start ā₀=0.5 (log-A₀ = ln(−ln 0.5)); k=2 predicts staying, k>2 climbing
        m0 (model/init {:vocab-size 16 :d-model 32 :n-heads 4 :d-ff 128
                        :seq-len SEQ-LEN :T T :seed k :log-A0 (Math/log (- (Math/log 0.5)))})
        predicted (- 1.0 (/ 1.0 k))]
    (println (format "\n=== k=%d   τ_needed=%d   predicted max(ā)=%.3f%s ===\n"
                     k k predicted (if (> k T) "   [UNSOLVABLE at T=8: τ>T → cliff probe]" "")))
    (loop [step 1, m m0, opt nil]
      (if (> step STEPS)
        (let [acc (accuracy m k rng)
              tops (top-abars m 4)]
          (println (format "k=%d DONE  acc=%.2f  max(ā)=%.3f  top4=%s  predicted=%.3f"
                           k acc (first tops) (mapv #(format "%.3f" %) tops) predicted))
          {:k k :predicted predicted :max-abar (first tops) :top4 tops
           :accuracy acc :final-loss nil})
        (let [b (copy/batch k SEQ-LEN N-CONTENT BATCH rng)
              {:keys [loss grads]} (train/batch-grad m b)
              opt (or opt (train/init-adam grads))
              [p' opt'] (train/adam-update (:params m) grads opt {:lr LR})
              m' (assoc m :params p')]
          (when (or (= step 1) (zero? (mod step 20)))
            (println (format "  step %3d  loss %.4f  acc %.2f  max(ā) %.3f"
                             step loss (accuracy m k rng) (first (top-abars m 1)))))
          (recur (inc step) m' opt'))))))

(println "copy-task sweep: T=8, seq-len" SEQ-LEN ", vocab 16, lr" LR ", steps" STEPS)
(def results (mapv run-k [2 6 8 12]))

(println "\n================ SUMMARY: learned max(ā) vs prediction 1−1/k ================")
(doseq [{:keys [k predicted max-abar accuracy]} results]
  (println (format "k=%2d  predicted %.3f   learned %.3f   Δ %+.3f   acc %.2f"
                   k predicted max-abar (- max-abar predicted) accuracy)))
(spit "scripts/copy_sweep.edn" (with-out-str (pp/pprint results)))
(println "\n--- saved scripts/copy_sweep.edn ---")
(System/exit 0)
