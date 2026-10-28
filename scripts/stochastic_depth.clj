;; v2 review follow-up (2026-10-27): does the carry-dispensability result
;; survive STOCHASTIC DEPTH training, the regime Parcae and Geiping et al.
;; actually use? The fixed-T=8 runs of record (carry_intervention.clj,
;; carry_zero.clj) produce a model whose accuracy lives only at T=8
;; (depth_accuracy.clj); the reviews ask whether that is a protocol artifact
;; and whether zero-carry still learns when T is sampled per batch.
;;
;; Protocol: identical to carry_intervention.clj (d=32, w=1, seq 12, NC 14,
;; vocab 16, batch 4, lr 1.5e-3, curriculum k=2..6 with the same stage
;; budgets, init seed 0, batch-stream rng seed 0, held-out eval seeds
;; 7000+k) EXCEPT that the loop depth T is drawn per batch, uniformly from
;; T-RANGE = {7,...,16}, from a dedicated rng (seed 1) so both conditions see
;; the same depth sequence. 7 is the minimum depth at which k=6 is solvable
;; at w=1 (the symbol needs k+1 hops), so every stage is solvable at every
;; sampled depth. Per-stage held-out accuracy is reported at T=8 (comparable
;; to the record) and at T in {7,12,16}; the final model is swept over
;; T in {4,...,64} x k in {2,...,6}, the depth_accuracy.clj protocol.
;;
;; Conditions (one per process, so they run in parallel):
;;   learned  — carry trainable, init a-bar = 0.5
;;   zero     — carry frozen at exactly 0 (log-A = 10.0, underflows to 0.0)
;;   frozen   — carry frozen at a-bar = 0.5 (condition (b) of the record):
;;              separates "nonzero carry" from "trainable carry" when zero fails
;;   frozen002 — carry frozen at a-bar = 0.02 (condition (c)): separates "any
;;              nonzero carry" from "a substantial carry" (added 2026-10-28)
;; Optional 2nd/3rd args: training seed (model init AND batch stream; default
;; 0) and learning rate (default 1.5e-3) — the rescue-probe axes of
;; seed2_rescue.clj. Outputs are suffixed with them when non-default.
;; Run:  clojure -M scripts/stochastic_depth.clj learned
;;       clojure -M scripts/stochastic_depth.clj zero
;;       clojure -M scripts/stochastic_depth.clj frozen
;;       clojure -M scripts/stochastic_depth.clj zero 1        (seed 1)
;;       clojure -M scripts/stochastic_depth.clj zero 0 3e-4   (seed 0, lr 3e-4)
(require '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.copytask :as copy]
         '[clojure.pprint :as pp])

(def COND (or (first *command-line-args*) "learned"))
(assert (#{"learned" "zero" "frozen" "frozen002"} COND) (str "unknown condition " COND))
(def SEED (Long/parseLong (or (second *command-line-args*) "0")))
(def LR (Double/parseDouble (or (nth *command-line-args* 2 nil) "1.5e-3")))
(def TAG (str COND (when-not (zero? SEED) (str "_seed" SEED)) (when-not (== LR 1.5e-3) (str "_lr" LR))))

(def SEQ 12) (def NC 14) (def W 1) (def BATCH 4)
(def D 32)
(def T-RANGE (vec (range 7 17)))                 ; 7..16 inclusive
(def T-EVAL 8)                                   ; per-stage reference depth
(def T-STAGE-EXTRA [7 12 16])
(def T-SWEEP [4 6 7 8 9 10 12 16 24 32 48 64])
(def STAGES [[2 300] [3 200] [4 200] [5 250] [6 400]])
(def ALL-K [2 3 4 5 6])

(defn init-model [log-A0]
  (model/init {:vocab-size 16 :d-model D :n-heads 4 :d-ff 128
               :seq-len SEQ :T T-EVAL :seed SEED :window W :log-A0 log-A0}))

(defn at-depth [m t] (assoc-in m [:config :T] t))

(defn accuracy [m k n seed]
  (let [rng (java.util.Random. (long seed))]
    (/ (count (filter (fn [{:keys [input qpos answer]}]
                        (= answer (apply max-key
                                         (vec (nth (:probs (:cache (model/forward m input input))) qpos))
                                         (range 16))))
                      (copy/batch k SEQ NC n rng)))
       (double n))))

(defn acc-sweep [m t]
  (into (sorted-map) (for [k ALL-K] [k (accuracy (at-depth m t) k 64 (+ 7000 k))])))

(defn train-condition
  [label log-A0 freeze?]
  (println (format "\n########## condition %s (log-A0 %.4f, freeze? %s, T ~ U%s, seed %d, lr %s) ##########"
                   label log-A0 freeze? (pr-str [(first T-RANGE) (last T-RANGE)]) SEED LR))
  (let [rng  (java.util.Random. SEED)
        trng (java.util.Random. 1)
        t0   (System/currentTimeMillis)
        fin (reduce
             (fn [{:keys [m opt results]} [k steps]]
               (let [{:keys [m opt t-hist]}
                     (reduce (fn [{:keys [m opt t-hist]} step]
                               (let [t (nth T-RANGE (.nextInt trng (count T-RANGE)))
                                     b (copy/batch k SEQ NC BATCH rng)
                                     {:keys [grads loss]} (train/batch-grad (at-depth m t) b)
                                     grads (if freeze? (assoc grads :log-A (vec (repeat D 0.0))) grads)
                                     opt (or opt (train/init-adam grads))
                                     [p' opt'] (train/adam-update (:params m) grads opt {:lr LR})]
                                 (when (zero? (mod step 100))
                                   (println (format "  [%s] k=%d step %3d  T=%2d  loss %.3f  max(a-bar) %.6f"
                                                    label k step t loss
                                                    (apply max (model/effective-abar m))))
                                   (flush))
                                 {:m (assoc m :params p') :opt opt' :t-hist (conj t-hist t)}))
                             {:m m :opt opt :t-hist []} (range 1 (inc steps)))
                     acc   (accuracy (at-depth m T-EVAL) k 64 (+ 7000 k))
                     extra (into (sorted-map)
                                 (for [t T-STAGE-EXTRA] [t (accuracy (at-depth m t) k 64 (+ 7000 k))]))
                     ab    (model/effective-abar m)]
                 (println (format "  [%s] k=%d DONE acc@T=8 %.2f  acc@T=%s  max(a-bar)=%.6f  mean T %.1f"
                                  label k acc (pr-str extra) (apply max ab)
                                  (/ (reduce + t-hist) (double (count t-hist)))))
                 (flush)
                 {:m m :opt opt
                  :results (conj results {:k k :accuracy acc :accuracy-at-T extra
                                          :max-abar (apply max ab)
                                          :mean-abar (/ (reduce + ab) (count ab))
                                          :mean-T (/ (reduce + t-hist) (double (count t-hist)))})}))
             {:m (init-model log-A0) :opt nil :results []} STAGES)
        final (into (sorted-map)
                    (for [t T-SWEEP]
                      (let [acc (acc-sweep (:m fin) t)]
                        (println (format "  [%s] final model  T=%2d  acc %s" label t (pr-str acc)))
                        (flush)
                        [t acc])))]
    (println (format "  [%s] done (%.1f min)" label (/ (- (System/currentTimeMillis) t0) 60000.0)))
    {:m (:m fin) :per-stage (:results fin) :final-by-T final}))

(def a05 (Math/log (- (Math/log 0.5))))         ; a-bar = 0.5

(when (= COND "zero")
  (let [ab (model/effective-abar (init-model 10.0))]
    (assert (every? #(== 0.0 %) ab) (str "a-bar not exactly zero: " (take 3 ab)))
    (println "sanity: a-bar == 0.0 exactly at log-A = 10.0")))

(def res (case COND
           "learned" (train-condition "learned" a05 false)
           "frozen"  (train-condition "frozen" a05 true)
           "frozen002" (train-condition "frozen002" (Math/log (- (Math/log 0.02))) true)
           "zero"    (train-condition "zero" 10.0 true)))

(def out {:config {:seq SEQ :nc NC :T-range T-RANGE :T-eval T-EVAL :w W :batch BATCH
                   :lr LR :d D :stages STAGES :condition COND :seed SEED}
          :per-stage (:per-stage res)
          :final-by-T (:final-by-T res)
          :abar (vec (model/effective-abar (:m res)))})
(spit (str "scripts/stochastic_depth_" TAG ".edn") (with-out-str (pp/pprint out)))
(spit (str "scripts/stochastic_depth_" TAG "_model.edn") (pr-str (:params (:m res))))
(println (format "\n--- stochastic-depth %s done (wrote scripts/stochastic_depth_%s.edn) ---" TAG TAG))
(System/exit 0)
