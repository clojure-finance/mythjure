;; Does gradient descent LEARN an unstable carry when it's unconstrained?
;; Reproduces Parcae Table 2 / Fig 3 at toy scale: train :parcae (ā=exp(−exp(log_A))
;; ∈(0,1), ρ(Ā)<1 by construction) vs :free (ā = raw per-channel vector, ρ(Ā)=max|ā|
;; unconstrained) side-by-side on Tiny Shakespeare across learning rates. Monitor
;; ρ(Ā) and the recurrent-state RMS ‖H_T‖. If :free learns ρ>1 with exploding ‖H_T‖
;; (esp. at high lr) while :parcae stays bounded, the constraint earns its keep.
;; Run:  clojure -M scripts/unconstrained_carry.clj
(require '[mythjure.data :as data]
         '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.linalg :as la]
         '[clojure.pprint :as pp])

(def D 24) (def SEQ 12) (def T 8) (def BATCH 4) (def STEPS 120)
(def LRS [1e-3 1e-2 3e-2 1e-1])

(println "loading corpus...")
(def encoded (data/encode (data/build-vocab (data/load-text)) (data/load-text)))
(def VOCAB 65)

(defn abar-of [m]
  (let [la (get-in m [:params :log-A])]
    (if (= (get-in m [:config :carry-mode]) :free) la (model/a-bar la))))
(defn rho [m] (apply max (map #(Math/abs (double %)) (abar-of m))))
(defn resid-rms [m ex]
  (let [HT (:HT (:cache (model/forward m (:input ex) (:target ex))))]
    (Math/sqrt (la/mean (map (fn [r] (la/mean (map #(* % %) r))) HT)))))
(defn finite? [x] (and (not (Double/isNaN x)) (not (Double/isInfinite x))))

(defn run [method lr]
  (let [log-A0 (if (= method :free) 0.5 (Math/log (- (Math/log 0.5))))   ; both start ā=0.5
        m0 (model/init {:vocab-size VOCAB :d-model D :n-heads 4 :d-ff 96
                        :seq-len SEQ :T T :seed 0 :carry-mode method :log-A0 log-A0})
        rng (java.util.Random. 0)]
    (println (format "\n=== method=%s lr=%g  (ρ₀=%.2f) ===" (name method) lr (rho m0)))
    (loop [step 1 m m0 opt nil]
      (if (> step STEPS)
        (let [r {:method method :lr lr :final-rho (rho m) :final-resid (resid-rms m (first (data/sample-batch encoded SEQ 1 rng))) :diverged false}]
          (println (format "%s lr=%g DONE  ρ=%.3f  ‖H_T‖=%.2f  CONVERGED" (name method) lr (:final-rho r) (:final-resid r)))
          r)
        (let [b (data/sample-batch encoded SEQ BATCH rng)
              {:keys [loss grads]} (train/batch-grad m b)
              opt (or opt (train/init-adam grads))
              [p' opt'] (train/adam-update (:params m) grads opt {:lr lr})
              m' (assoc m :params p')
              rr (resid-rms m' (first b))]
          (when (zero? (mod step 20))
            (println (format "  %s lr=%g step %3d  loss %.3f  ρ(Ā) %.3f  ‖H_T‖ %.2f"
                             (name method) lr step loss (rho m') rr)))
          (if (or (not (finite? loss)) (not (finite? rr)) (> rr 1e5))
            (do (println (format "%s lr=%g DIVERGED at step %d  ρ=%.3f  ‖H_T‖=%.3g"
                                 (name method) lr step (rho m') rr))
                {:method method :lr lr :final-rho (rho m') :final-resid rr :diverged true :step step})
            (recur (inc step) m' opt')))))))

(def results (vec (for [method [:parcae :free] lr LRS] (run method lr))))

(println "\n===== SUMMARY: constrained (:parcae) vs unconstrained (:free) carry =====")
(println "method   lr       final ρ(Ā)   ‖H_T‖        outcome")
(doseq [{:keys [method lr final-rho final-resid diverged step]} results]
  (println (format "%-8s %-8g %-11.3f  %-11.3g  %s" (name method) lr final-rho final-resid
                   (if diverged (str "DIVERGED@" step) "converged"))))
(spit "scripts/unconstrained_carry.edn" (with-out-str (pp/pprint results)))
(println "\n--- saved scripts/unconstrained_carry.edn ---")
(System/exit 0)
