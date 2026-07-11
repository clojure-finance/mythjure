;; Feasibility gate: can a WINDOWED (w=1) looped model learn even the k=2 copy?
;; w=1 ⇒ info moves 1 position/loop, so bridging the gap is a learned shift
;; register (carry holds the token between hops), not a one-hop attention lookup.
;; If this won't learn, the curriculum is moot. Watch acc and max(ā).
;; Run:  clojure -M scripts/window_feasibility.clj
(require '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.copytask :as copy])

(def SEQ 10) (def NC 14) (def T 8) (def W 1) (def K 2)
(def STEPS 500) (def BATCH 6) (def LR 1e-3)

(defn accuracy [m k n rng]
  (let [b (copy/batch k SEQ NC n rng)]
    (/ (count (filter (fn [{:keys [input qpos answer]}]
                        (= answer (apply max-key (vec (nth (:probs (:cache (model/forward m input input))) qpos))
                                         (range 16))))
                      b))
       (double n))))

(println (format "windowed feasibility: w=%d k=%d T=%d seq=%d d=24 lr=%g batch=%d" W K T SEQ LR BATCH))
(def m0 (model/init {:vocab-size 16 :d-model 24 :n-heads 4 :d-ff 96
                     :seq-len SEQ :T T :seed 0 :window W
                     :log-A0 (Math/log (- (Math/log 0.5)))}))   ; ā₀=0.5
(def rng (java.util.Random. 0))
(def eval-rng (java.util.Random. 999))

(reduce (fn [{:keys [m opt]} step]
          (let [b (copy/batch K SEQ NC BATCH rng)
                {:keys [loss grads]} (train/batch-grad m b)
                opt (or opt (train/init-adam grads))
                [p' opt'] (train/adam-update (:params m) grads opt {:lr LR})
                m' (assoc m :params p')]
            (when (or (= step 1) (zero? (mod step 30)))
              (let [ab (model/a-bar (get-in m' [:params :log-A]))]
                (println (format "step %3d  loss %.4f  acc %.2f  ā[max %.3f mean %.3f]"
                                 step loss (accuracy m' K 32 eval-rng)
                                 (apply max ab) (/ (reduce + ab) (count ab))))))
            {:m m' :opt opt'}))
        {:m m0 :opt nil} (range 1 (inc STEPS)))
(println "--- feasibility run done ---")
(System/exit 0)
