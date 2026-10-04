;; E1+E4 of attic/paper-v2-plan.md §C (review-2 pt 1, "the missing
;; intervention"): does the windowed-copy family NEED the carry at all?
;; Same base config as windowed_hard.clj (the §7.4 run of record: d=32, T=8,
;; w=1, seq 12, NC 14, vocab 16, batch 4, lr 1.5e-3, curriculum k=2→6, seed 0;
;; identical training-batch stream across conditions via a fresh rng seed 0
;; per condition). Conditions:
;;   (a) :learned      — carry trainable (replicates the run of record, and
;;                       provides the trained model for (d) and E4)
;;   (b) :frozen-0.5   — carry frozen at its init ā=0.5 (log-A grad zeroed)
;;   (c) :frozen-0.02  — carry frozen at ā≈0.02 (effectively no linear memory)
;;   (d) post-hoc      — trained (a) with ā := 0.0 / 0.5 set WITHOUT retraining,
;;                       accuracy re-measured per k
;; Decision rule (plan §C): (b)+(c) solve ⇒ carry dispensable here; (c) fails
;; but (b) solves ⇒ the resting value is load-bearing; (d) separates "never
;; consulted" from "consulted at rest".
;; E4: ρ_spec / ‖J_G‖₂ of the full recurrence Jacobian at the states the model
;; actually visits (H_T on task inputs), trained vs. init weights, plus
;; settle classification from H_T — the "no observed stability cost is
;; measured at init only" gap (review-2 pt 8).
;; Run:  clojure -M scripts/carry_intervention.clj
(require '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.copytask :as copy]
         '[mythjure.analysis :as an]
         '[clojure.pprint :as pp])

(def SEQ 12) (def NC 14) (def T 8) (def W 1) (def BATCH 4) (def LR 1.5e-3)
(def D 32)
(def STAGES [[2 300] [3 200] [4 200] [5 250] [6 400]])
(def ALL-K [2 3 4 5 6])

(defn init-model [log-A0]
  (model/init {:vocab-size 16 :d-model D :n-heads 4 :d-ff 128
               :seq-len SEQ :T T :seed 0 :window W :log-A0 log-A0}))

(defn accuracy [m k n seed]
  (let [rng (java.util.Random. (long seed))]
    (/ (count (filter (fn [{:keys [input qpos answer]}]
                        (= answer (apply max-key
                                         (vec (nth (:probs (:cache (model/forward m input input))) qpos))
                                         (range 16))))
                      (copy/batch k SEQ NC n rng)))
       (double n))))

(defn acc-sweep [m] (into (sorted-map) (for [k ALL-K] [k (accuracy m k 64 (+ 7000 k))])))

(defn train-condition
  "Run the full curriculum. freeze? ⇒ zero the :log-A grad every step."
  [label log-A0 freeze?]
  (println (format "\n########## condition %s (log-A0 %.4f, freeze? %s) ##########" label log-A0 freeze?))
  (let [rng (java.util.Random. 0)
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
                                   (println (format "  [%s] k=%d step %3d  loss %.3f  max(ā) %.3f"
                                                    label k step loss
                                                    (apply max (model/effective-abar m)))))
                                 {:m (assoc m :params p') :opt opt'}))
                             {:m m :opt opt} (range 1 (inc steps)))
                     acc (accuracy m k 64 (+ 7000 k))
                     ab  (model/effective-abar m)]
                 (println (format "  [%s] k=%d DONE acc=%.2f max(ā)=%.3f" label k acc (apply max ab)))
                 {:m m :opt opt
                  :results (conj results {:k k :accuracy acc
                                          :max-abar (apply max ab)
                                          :mean-abar (/ (reduce + ab) (count ab))})}))
             {:m (init-model log-A0) :opt nil :results []} STAGES)
        final-acc (acc-sweep (:m fin))]
    (println (format "  [%s] final-model acc sweep %s  (%.1f min)"
                     label (pr-str final-acc) (/ (- (System/currentTimeMillis) t0) 60000.0)))
    {:m (:m fin) :per-stage (:results fin) :final-acc final-acc}))

;; --- (a)(b)(c) ---------------------------------------------------------------
(def a05  (Math/log (- (Math/log 0.5))))    ; ā = 0.5
(def a002 (Math/log (- (Math/log 0.02))))   ; ā ≈ 0.02
(def cond-a (train-condition "learned"     a05  false))
(def cond-b (train-condition "frozen-0.5"  a05  true))
(def cond-c (train-condition "frozen-0.02" a002 true))

;; --- (d) post-hoc intervention on trained (a) --------------------------------
(println "\n########## (d) post-hoc carry intervention on trained (a) ##########")
(defn set-abar [m v]
  ;; ā = exp(−exp(log-A)); ā=0.0 via log-A=10 (exp(−e^10) underflows to 0.0)
  (let [la (if (zero? v) 10.0 (Math/log (- (Math/log v))))]
    (assoc-in m [:params :log-A] (vec (repeat D la)))))
(def post-hoc
  (into (sorted-map)
        (for [v [0.0 0.5]]
          (let [acc (acc-sweep (set-abar (:m cond-a) v))]
            (println (format "  ā:=%.1f  acc sweep %s" v (pr-str acc)))
            [v acc]))))
(println (format "  baseline (learned ā)  acc sweep %s" (pr-str (:final-acc cond-a))))

;; --- E4: trained vs init Jacobians at visited states -------------------------
(println "\n########## E4: recurrence Jacobian, trained vs init, k=6 inputs ##########")
(defn jac-at [m ex]
  (let [{:keys [cache]} (model/forward m (:input ex) (:input ex))
        {:keys [E ab HT]} cache
        G  (an/gated-update-fn (get-in m [:params :block]) E ab)
        sr (an/spectral-radius G HT)
        J  (an/jacobian-matrix G HT)
        op (an/operator-norm J)
        st (an/classify-settle G HT)]
    {:rho-spec-at-HT (:rho-spec sr)
     :growth-max-at-HT (:growth-max sr)
     :op-norm-at-HT (Double/parseDouble (format "%.4f" op))
     :settle-from-HT (select-keys st [:status :iters :resid])
     :rho-spec-at-H* (when (= :converged (:status st))
                       (:rho-spec (an/spectral-radius G (:H st))))}))
(def e4-examples (copy/batch 6 SEQ NC 2 (java.util.Random. 4242)))
(def e4
  (into (sorted-map)
        (for [[label m] [[:trained (:m cond-a)] [:init (init-model a05)]]]
          (let [rows (mapv #(jac-at m %) e4-examples)]
            (println (format "  %s:" (name label)))
            (doseq [r rows] (println "   " (pr-str r)))
            [label rows]))))

;; --- persist ------------------------------------------------------------------
(def out {:config {:seq SEQ :nc NC :T T :w W :batch BATCH :lr LR :d D :stages STAGES}
          :learned     {:per-stage (:per-stage cond-a) :final-acc (:final-acc cond-a)
                        :abar (vec (model/effective-abar (:m cond-a)))}
          :frozen-0.5  {:per-stage (:per-stage cond-b) :final-acc (:final-acc cond-b)}
          :frozen-0.02 {:per-stage (:per-stage cond-c) :final-acc (:final-acc cond-c)}
          :post-hoc-on-learned post-hoc
          :e4-jacobians e4})
(spit "scripts/carry_intervention.edn" (with-out-str (pp/pprint out)))
(spit "scripts/carry_intervention_model_a.edn" (pr-str (:params (:m cond-a))))
(println "\n--- carry-intervention done (wrote scripts/carry_intervention.edn) ---")
(System/exit 0)
