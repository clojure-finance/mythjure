;; Seed expansion at ρ=0.9 for the width-collapse law (direction doc §2
;; follow-up to scripts/spectral_dsweep_torch.clj): the d-sweep's n=10 found
;; the spectral margin 1−ρ_spec shrinking ~2× per doubling of d, zero within
;; estimator noise at d=256.  This run pins (a) the margin→0 law and (b) the
;; non-convergence fraction, with 100 seeds per d ∈ {32, 64, 128, 256}
;; (head-dim 8; d=32/64 are near-free and sharpen the fit; d=32's first 10
;; seeds double as the reproduce-the-paper control vs scripts/seed_verify.edn).
;; Same E draw (seed 99) and schedules as the d-sweep: settle tol 1e-9 cap
;; 20000 probe 2000, matrix-free spectral radius 160/70, float64.  No dense/cf
;; cross-checks here — the d-sweep already did those at n=2; this run is
;; margin + convergence statistics only (agg gains :stderr; the
;; non-convergence fraction gets a Wilson 95% CI).
;; Budget-guarded (default 240 min, cheapest rung first) and CHECKPOINTED:
;; the output .edn is rewritten after every seed, so a killed run loses at
;; most one seed.  Overrides for spot-checks / other machines:
;;   MYTHJURE_SWEEP_SEEDS=N        seeds 1..N            (default 100)
;;   MYTHJURE_SWEEP_DMAX=D         drop rungs above D    (default 256)
;;   MYTHJURE_SWEEP_BUDGET_MIN=M   wall budget, minutes  (default 240)
;; Run:  clojure -M:torch scripts/spectral_seed_expansion_torch.clj   (Python w/ torch auto-discovered; MYTHJURE_PYTHON/MYTHJURE_LIBPYTHON override)
(require '[mythjure.block :as blk]
         '[mythjure.linalg :as la]
         '[mythjure.torch.core :as tc]
         '[mythjure.torch.tensor :as t]
         '[mythjure.analysis-torch :as ant]
         '[clojure.edn :as edn]
         '[clojure.pprint :as pp])

(tc/initialize!)

(def T0 (System/nanoTime))
(defn env-num [k parse default]
  (if-let [v (System/getenv k)] (parse v) default))
(def BUDGET-MIN (env-num "MYTHJURE_SWEEP_BUDGET_MIN" #(Double/parseDouble %) 240.0))
(def N-SEEDS (env-num "MYTHJURE_SWEEP_SEEDS" #(Long/parseLong %) 100))
(def DMAX (env-num "MYTHJURE_SWEEP_DMAX" #(Long/parseLong %) 256))
(defn elapsed-min [] (/ (- (System/nanoTime) T0) 6e10))
(defn remaining-min [] (- BUDGET-MIN (elapsed-min)))
(defn over? [] (neg? (remaining-min)))

(def RHO 0.9)
(def RUNGS (filterv (fn [[d _]] (<= d DMAX))
                    [[32 4] [64 8] [128 16] [256 32]])) ; [d-model n-heads], head-dim 8
(def SEEDS (range 1 (inc N-SEEDS)))
(def SEQ-LEN 16)
(def OUT-FILE "scripts/spectral_seed_expansion_torch.edn")

(defn r3 [x] (Double/parseDouble (format "%.3f" x)))
(defn r4 [x] (Double/parseDouble (format "%.4f" x)))
(defn r1e [x] (Double/parseDouble (format "%.1e" x)))
(defn agg [xs]
  (let [xs (mapv double xs)
        n (count xs) mu (/ (reduce + xs) n)
        sd (Math/sqrt (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs)) n))]
    {:mean (r3 mu) :std (r3 sd) :stderr (r4 (/ sd (Math/sqrt n)))
     :min (r3 (apply min xs)) :max (r3 (apply max xs))}))

(defn wilson-95 [k n]
  (let [z 1.96 p (/ (double k) n) z2 (* z z)
        den (+ 1.0 (/ z2 n))
        mid (/ (+ p (/ z2 (* 2.0 n))) den)
        hw (/ (* z (Math/sqrt (+ (/ (* p (- 1.0 p)) n) (/ z2 (* 4.0 n n))))) den)]
    {:k k :n n :frac (r3 p)
     :lo (r3 (max 0.0 (- mid hw))) :hi (r3 (min 1.0 (+ mid hw)))}))

;; analysis-torch's settle/classify-settle with a deadline check every 500
;; iterations — same tol/cap/probe schedule as the paper otherwise.  A seed
;; whose settle is cut off mid-run gets :status :budget-truncated and is
;; excluded from all statistics (recorded, not silently dropped).
(defn settle-b [G-t H0-t tol cap]
  (loop [k 0, H H0-t]
    (let [GH (G-t H)
          r (/ (t/norm (t/sub GH H)) (t/norm H))]
      (cond
        (< r tol) {:H H :resid r :iters k :converged? true}
        (>= k cap) {:H H :resid r :iters k :converged? false}
        (and (zero? (mod (inc k) 500)) (over?))
        {:H H :resid r :iters k :converged? false :deadline? true}
        :else (recur (inc k) GH)))))

(defn classify-settle-b [G-t H0-t]
  (let [st (settle-b G-t H0-t 1e-9 20000)]
    (cond
      (:converged? st) (assoc st :status :converged)
      (:deadline? st) (assoc st :status :budget-truncated)
      :else
      ;; :iters cumulative across both legs, matching analysis/classify-settle
      (let [st2 (update (settle-b G-t (:H st) 1e-9 2000) :iters + (:iters st))]
        (cond
          (:converged? st2) (assoc st2 :status :converged)
          (:deadline? st2) (assoc st2 :status :budget-truncated)
          (< (:resid st2) (* 0.75 (:resid st))) (assoc st2 :status :slow)
          :else (assoc st2 :status :non-convergent))))))

;; Unlike the d-sweep's seed-run, no :G/:H retained — nothing downstream needs
;; the tensors, and at 100 seeds/rung holding them would pin Python memory.
(defn seed-run [p-cfg E E-t s]
  (let [p (blk/init-params (assoc p-cfg :seed s))
        G-t (ant/gated-update-t p E RHO)
        st (classify-settle-b G-t E-t)]
    (merge {:seed s :status (:status st) :resid (r1e (:resid st)) :iters (:iters st)}
           (when (contains? #{:converged :slow} (:status st))
             (let [mf (ant/spectral-radius-t G-t (:H st))]
               {:rho-spec (:rho-spec mf) :growth-max (:growth-max mf)})))))

(defn rung-summary [d heads runs complete?]
  (let [fixed (filterv :rho-spec runs)
        n-nc (count (filter #(= :non-convergent (:status %)) runs))
        conv (filterv #(= :converged (:status %)) runs)]
    {:d d :n-heads heads
     :n-run (count runs)
     :complete? complete?
     :statuses (frequencies (map :status runs))
     :n-with-fixed-point (count fixed)
     :non-convergent-fraction (when (pos? (count runs)) (wilson-95 n-nc (count runs)))
     :rho-spec (when (seq fixed) (agg (mapv :rho-spec fixed)))
     :margin (when (seq fixed) (agg (mapv #(- 1.0 (:rho-spec %)) fixed)))
     :growth-max (when (seq fixed) (agg (mapv :growth-max fixed)))
     :settle-iters (when (seq conv) (agg (mapv :iters conv)))
     :seeds (vec runs)}))

(def CONFIG
  {:rho RHO :rungs RUNGS :seeds [1 N-SEEDS] :seq-len SEQ-LEN :head-dim 8
   :e-seed 99
   :schedules {:settle {:tol 1e-9 :cap 20000 :probe 2000}
               :mf-spectral {:iters 160 :warmup 70}}
   :budget-min BUDGET-MIN})

(def done-rungs (atom []))

(defn checkpoint! [current status extra]
  (spit OUT-FILE
        (with-out-str
          (pp/pprint (merge {:config CONFIG
                             :status status
                             :elapsed-min (r3 (elapsed-min))
                             :rungs (cond-> @done-rungs current (conj current))}
                            extra)))))

(defn rung [d heads]
  (println (format "=== d=%d (heads %d, ρ=%.1f, %d seeds)  [%.1f min elapsed, %.1f left] ==="
                   d heads RHO N-SEEDS (elapsed-min) (remaining-min)))
  (flush)
  (let [t0 (System/nanoTime)
        E (la/rand-matrix 99 SEQ-LEN d 1.0)
        E-t (t/from-clj E :dtype :float64)
        p-cfg {:d-model d :n-heads heads}
        runs (loop [ss SEEDS acc []]
               (if (or (empty? ss) (over?))
                 acc
                 (let [r (seed-run p-cfg E E-t (first ss))
                       acc' (conj acc r)]
                   (when (zero? (mod (count acc') 10))
                     (println (format "  seed %d/%d  %s  (%.1f min elapsed)"
                                      (count acc') N-SEEDS (name (:status r)) (elapsed-min)))
                     (flush))
                   (checkpoint! (rung-summary d heads acc' false) :in-progress nil)
                   (recur (rest ss) acc'))))
        n-trunc (count (filter #(= :budget-truncated (:status %)) runs))
        complete? (and (= N-SEEDS (count runs)) (zero? n-trunc))
        out (assoc (rung-summary d heads runs complete?)
                   :wall-min (r3 (/ (- (System/nanoTime) t0) 6e10)))]
    (println (format "  done: fixed %d/%d%s  margin %s (stderr %s)  non-conv %s  (%.1f min)"
                     (:n-with-fixed-point out) (:n-run out)
                     (if complete? "" " PARTIAL")
                     (str (get-in out [:margin :mean]))
                     (str (get-in out [:margin :stderr]))
                     (str (get-in out [:non-convergent-fraction :frac]))
                     (:wall-min out)))
    (flush)
    out))

(println (format "=== ρ=%.1f seed expansion, torch backend (float64), %d seeds/rung, budget %.0f min ==="
                 RHO N-SEEDS BUDGET-MIN))
(loop [rungs RUNGS, prev-wall nil, factor 4.0]
  (when (seq rungs)
    (let [[d h] (first rungs)
          projected (when prev-wall (* prev-wall factor))]
      (cond
        (over?)
        (do (println (format "-- deadline hit; skipping d=%d and above" d))
            (swap! done-rungs into
                   (map (fn [[d' h']] {:d d' :n-heads h' :skipped? true :reason :deadline})
                        rungs)))

        (and projected (< (remaining-min) (* 0.3 projected)))
        (do (println (format "-- d=%d projected ~%.1f min, only %.1f left; stopping ladder"
                             d projected (remaining-min)))
            (swap! done-rungs into
                   (map (fn [[d' h']] {:d d' :n-heads h' :skipped? true :reason :projected-over-budget})
                        rungs)))

        :else
        (let [r (rung d h)
              factor' (if (and prev-wall (pos? prev-wall) (:complete? r))
                        (max 2.0 (/ (:wall-min r) prev-wall))
                        factor)]
          (swap! done-rungs conj r)
          (checkpoint! nil :in-progress nil)
          (recur (rest rungs) (:wall-min r) factor'))))))

;; --- d=32 control: first 10 seeds vs the committed seed_verify.edn ρ=0.9 row
;; (same E seed, param seeds, and schedules ⇒ per-seed reproduction expected).
(defn deep-close? [a b]
  (cond
    (and (number? a) (number? b))
    (or (== a b)
        (< (Math/abs (- (double a) (double b)))
           (* 1e-4 (max 1.0 (Math/abs (double a))))))
    (and (map? a) (map? b))
    (and (= (set (keys a)) (set (keys b)))
         (every? (fn [[k v]] (deep-close? v (get b k))) a))
    (and (sequential? a) (sequential? b))
    (and (= (count a) (count b)) (every? true? (map deep-close? a b)))
    :else (= a b)))

(println "\n=== d=32 control (seeds 1–10) vs committed scripts/seed_verify.edn ρ=0.9 row ===")
(def control-ok?
  (let [d32 (first (filter #(= 32 (:d %)) @done-rungs))
        sub (filterv #(<= (:seed %) 10) (:seeds d32))]
    (if (< (count sub) 10)
      (do (println "  fewer than 10 seeds at d=32 — no control diff") nil)
      (let [row (first (filter #(= 0.9 (:rho %))
                               (edn/read-string (slurp "scripts/seed_verify.edn"))))
            fixed (filterv :rho-spec sub)
            ok? (and (= (:n-with-fixed-point row) (count fixed))
                     (deep-close? (:rho-spec-mf row)
                                  (dissoc (agg (mapv :rho-spec fixed)) :stderr))
                     (= (set (map :seed (:non-convergent row)))
                        (set (map :seed (filter #(= :non-convergent (:status %)) sub)))))]
        (println (format "  %s" (if ok? "MATCH" "*** MISMATCH ***")))
        (boolean ok?)))))

(println "\n=== margin law (ρ=0.9) ===")
(println "  d     n      fixed    margin mean±std (stderr)   non-conv k/n [95% CI]    settle iters mean")
(doseq [r @done-rungs :when (not (:skipped? r))]
  (println (format "  %-5d %-6d %-8s %-26s %-25s %s"
                   (:d r) (:n-run r)
                   (str (:n-with-fixed-point r) "/" (:n-run r) (if (:complete? r) "" "*"))
                   (if-let [m (:margin r)]
                     (format "%.3f ± %.3f (%.4f)" (:mean m) (:std m) (:stderr m)) "—")
                   (if-let [f (:non-convergent-fraction r)]
                     (format "%d/%d [%.3f, %.3f]" (:k f) (:n f) (:lo f) (:hi f)) "—")
                   (or (some-> r :settle-iters :mean str) "—"))))
(doseq [r @done-rungs :when (:skipped? r)]
  (println (format "  %-5d skipped (%s)" (:d r) (name (:reason r)))))
(let [ms (for [r @done-rungs :when (and (:complete? r) (:margin r))]
           [(:d r) (get-in r [:margin :mean])])]
  (doseq [[[d1 m1] [d2 m2]] (partition 2 1 ms)]
    (if (pos? m2)
      (println (format "  margin ratio d=%d→d=%d: %.2f× per doubling" d1 d2 (/ m1 m2)))
      (println (format "  margin ratio d=%d→d=%d: margin ≤ 0 at d=%d (collapsed)" d1 d2 d2)))))

(checkpoint! nil :done
             {:control-vs-seed-verify (if (nil? control-ok?) :not-run
                                          (if control-ok? :match :mismatch))})
(println (str "\nsaved " OUT-FILE))
(println (format "total wall: %.1f min" (elapsed-min)))
(System/exit 0)
