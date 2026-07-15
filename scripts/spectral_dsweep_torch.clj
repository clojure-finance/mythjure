;; d-ladder for the paper's fixed-point/spectral claims (§5.3): does the d=32
;; story survive width?  For d ∈ {32, 64, 128, 256} (head-dim pinned at 8 so
;; heads = d/8; d-ff = 4d and init std 1/√d scale automatically), seq 16,
;; 10 seeds × ρ ∈ {0.3, 0.9}, paper schedules (settle tol 1e-9 cap 20000
;; probe 2000, matrix-free spectral radius 160/70), float64:
;;   1. convergence fraction — is "1–2/10 seeds never settle" a d=32 artifact?
;;   2. spectral margin 1−ρ_spec — does the ρ→1 collapse trend move with d?
;;   3. ‖J_G‖₂ vs ρ_spec dense cross-check (first 2 fixed-point seeds/cell) —
;;      does the non-normality gap behind the spiral rebound persist?
;;   4. C/F = ‖J_f‖₂/(1−ρ) at one seed/cell — does ≈2.5 hold across d?
;; Budget-guarded (~25 min): the ladder runs cheapest-first, settle checks the
;; clock every 500 iterations (a d=256 non-convergent orbit at cap 20000 would
;; alone cost ~9 min), the next rung is attempted only if its projected cost
;; fits, and everything skipped or truncated is recorded as such.  The d=32
;; rung is the reproduce-the-paper control, diffed against the committed
;; scripts/seed_verify.edn.
;; Run:  clojure -M:torch scripts/spectral_dsweep_torch.clj   (Python w/ torch auto-discovered; MYTHJURE_PYTHON/MYTHJURE_LIBPYTHON override)
(require '[mythjure.block :as blk]
         '[mythjure.linalg :as la]
         '[mythjure.torch.core :as tc]
         '[mythjure.torch.tensor :as t]
         '[mythjure.analysis-torch :as ant]
         '[clojure.edn :as edn]
         '[clojure.pprint :as pp])

(tc/initialize!)

(def T0 (System/nanoTime))
(def BUDGET-MIN 25.0)
(defn elapsed-min [] (/ (- (System/nanoTime) T0) 6e10))
(defn remaining-min [] (- BUDGET-MIN (elapsed-min)))
(defn over? [] (neg? (remaining-min)))

(def RUNGS [[32 4] [64 8] [128 16] [256 32]]) ; [d-model n-heads], head-dim 8
(def RHOS [0.3 0.9])
(def SEEDS (range 1 11))
(def SEQ-LEN 16)
(def DENSE-N 2)

(defn r3 [x] (Double/parseDouble (format "%.3f" x)))
(defn r1e [x] (Double/parseDouble (format "%.1e" x)))
(defn agg [xs]
  (let [n (count xs) mu (/ (reduce + xs) n)
        sd (Math/sqrt (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs)) n))]
    {:mean (r3 mu) :std (r3 sd) :min (r3 (apply min xs)) :max (r3 (apply max xs))}))

;; analysis-torch's settle/classify-settle with a deadline check every 500
;; iterations — same tol/cap/probe schedule as the paper otherwise.  A seed
;; whose settle is cut off mid-run gets :status :budget-truncated and is
;; excluded from all statistics (recorded, not silently dropped).
(defn settle-b [G-t H0-t tol cap]
  (loop [k 0, H H0-t]
    (let [GH (G-t H)
          r  (/ (t/norm (t/sub GH H)) (t/norm H))]
      (cond
        (< r tol)  {:H H :resid r :iters k :converged? true}
        (>= k cap) {:H H :resid r :iters k :converged? false}
        (and (zero? (mod (inc k) 500)) (over?))
        {:H H :resid r :iters k :converged? false :deadline? true}
        :else      (recur (inc k) GH)))))

(defn classify-settle-b [G-t H0-t]
  (let [st (settle-b G-t H0-t 1e-9 20000)]
    (cond
      (:converged? st) (assoc st :status :converged)
      (:deadline? st)  (assoc st :status :budget-truncated)
      :else
      (let [st2 (settle-b G-t (:H st) 1e-9 2000)]
        (cond
          (:converged? st2)                     (assoc st2 :status :converged)
          (:deadline? st2)                      (assoc st2 :status :budget-truncated)
          (< (:resid st2) (* 0.75 (:resid st))) (assoc st2 :status :slow)
          :else                                 (assoc st2 :status :non-convergent))))))

(defn seed-run [p-cfg E E-t rho s]
  (let [p   (blk/init-params (assoc p-cfg :seed s))
        G-t (ant/gated-update-t p E rho)
        st  (classify-settle-b G-t E-t)]
    (merge {:seed s :status (:status st) :resid (:resid st) :iters (:iters st)
            :G G-t}
           (when (contains? #{:converged :slow} (:status st))
             (let [mf (ant/spectral-radius-t G-t (:H st))]
               {:H          (:H st)
                :rho-spec   (:rho-spec mf)
                :growth-max (:growth-max mf)})))))

(defn dense-check [r]
  (let [J  (ant/jacobian-matrix-t (:G r) (:H r))
        op (r3 (ant/operator-norm-t J :iters 150))]
    {:seed           (:seed r)
     :op-norm        op
     :rho-spec-dense (ant/spectral-radius-dense-t J)
     :nonnormality   (r3 (- op (:rho-spec r)))}))

(defn cf-check [p-cfg E rho r]
  (let [p    (blk/init-params (assoc p-cfg :seed (:seed r)))
        f-t  (ant/increment-t p E)
        lipf (ant/operator-norm-t (ant/jacobian-matrix-t f-t (:H r)))]
    {:seed (:seed r) :lipf-true (r3 lipf) :cf-true (r3 (/ lipf (- 1.0 rho)))}))

(defn cell [p-cfg E E-t rho]
  (let [t0 (System/nanoTime)
        runs (loop [ss SEEDS acc []]
               (if (or (empty? ss) (over?))
                 acc
                 (recur (rest ss) (conj acc (seed-run p-cfg E E-t rho (first ss))))))
        fixed  (filterv :rho-spec runs)
        dense  (vec (for [r (take DENSE-N fixed) :while (not (over?))]
                      (dense-check r)))
        cfseed (or (first (filter #(= 7 (:seed %)) fixed)) (first fixed))
        cf     (when (and cfseed (not (over?))) (cf-check p-cfg E rho cfseed))
        n-trunc (count (filter #(= :budget-truncated (:status %)) runs))
        out {:rho rho
             :n-run (count runs)
             :n-budget-truncated n-trunc
             :complete? (and (= 10 (count runs)) (zero? n-trunc))
             :n-with-fixed-point (count fixed)
             :statuses (frequencies (map :status runs))
             :non-convergent (vec (for [r runs :when (= :non-convergent (:status r))]
                                    {:seed (:seed r) :resid (r1e (:resid r))}))
             :slow (vec (for [r fixed :when (= :slow (:status r))]
                          {:seed (:seed r) :resid (r1e (:resid r))}))
             :rho-spec   (when (seq fixed) (agg (mapv :rho-spec fixed)))
             :margin     (when (seq fixed) (agg (mapv #(- 1.0 (:rho-spec %)) fixed)))
             :growth-max (when (seq fixed) (agg (mapv :growth-max fixed)))
             :dense dense
             :cf cf
             :seeds (mapv #(dissoc % :G :H) runs)}]
    (println (format "  ρ=%.2f  fixed %d/%d%s  margin %s  ‖J‖₂ %s  cf %s  (%.1f min)"
                     rho (count fixed) (count runs)
                     (if (:complete? out) "" " PARTIAL")
                     (str (get-in out [:margin :mean]))
                     (pr-str (mapv :op-norm dense))
                     (str (:cf-true cf))
                     (/ (- (System/nanoTime) t0) 6e10)))
    (flush)
    out))

(defn rung [d heads]
  (println (format "=== d=%d (heads %d, head-dim 8, d-ff %d)  [%.1f min elapsed, %.1f left] ==="
                   d heads (* 4 d) (elapsed-min) (remaining-min)))
  (flush)
  (let [t0    (System/nanoTime)
        E     (la/rand-matrix 99 SEQ-LEN d 1.0)
        E-t   (t/from-clj E :dtype :float64)
        p-cfg {:d-model d :n-heads heads}
        cells (loop [rs RHOS acc []]
                (if (or (empty? rs) (over?))
                  acc
                  (recur (rest rs) (conj acc (cell p-cfg E E-t (first rs))))))]
    {:d d :n-heads heads
     :wall-min (/ (- (System/nanoTime) t0) 6e10)
     :complete? (and (= (count cells) (count RHOS)) (every? :complete? cells))
     :cells cells}))

(println "=== spectral d-ladder, torch backend (float64), budget" BUDGET-MIN "min ===")
(def results
  (loop [rungs RUNGS, prev-wall nil, factor 4.0, acc []]
    (if (empty? rungs)
      acc
      (let [[d h]     (first rungs)
            projected (when prev-wall (* prev-wall factor))]
        (cond
          (over?)
          (do (println (format "-- deadline hit; skipping d=%d and above" d))
              (into acc (map (fn [[d' h']]
                               {:d d' :n-heads h' :skipped? true :reason :deadline})
                             rungs)))

          (and projected (< (remaining-min) (* 0.3 projected)))
          (do (println (format "-- d=%d projected ~%.1f min, only %.1f left; stopping ladder"
                               d projected (remaining-min)))
              (into acc (map (fn [[d' h']]
                               {:d d' :n-heads h' :skipped? true :reason :projected-over-budget})
                             rungs)))

          :else
          (let [r       (rung d h)
                factor' (if (and prev-wall (pos? prev-wall) (:complete? r))
                          (max 2.0 (/ (:wall-min r) prev-wall))
                          factor)]
            (recur (rest rungs) (:wall-min r) factor' (conj acc r))))))))

;; --- d=32 control: same seeds/E/schedules as the committed seed_verify.edn ---
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

(println "\n=== d=32 control vs committed scripts/seed_verify.edn ===")
(def control-ok?
  (let [committed (edn/read-string (slurp "scripts/seed_verify.edn"))
        d32       (first (filter #(= 32 (:d %)) results))]
    (if-not (and d32 (:complete? d32))
      (do (println "  d=32 rung incomplete — no control diff") nil)
      (every? true?
              (for [row committed]
                (let [c   (first (filter #(= (:rho row) (:rho %)) (:cells d32)))
                      ok? (and c
                               (= (:n-with-fixed-point row) (:n-with-fixed-point c))
                               (deep-close? (:rho-spec-mf row) (:rho-spec c))
                               (= (set (map :seed (:non-convergent row)))
                                  (set (map :seed (:non-convergent c)))))]
                  (println (format "  ρ=%.2f  %s" (double (:rho row))
                                   (if ok? "MATCH" "*** MISMATCH ***")))
                  (boolean ok?)))))))

(println "\n=== trend table (dense columns = first" DENSE-N "fixed-point seeds; cf = one seed) ===")
(println "  d     ρ     fixed   margin mean±std    ‖J_G‖₂        ρ_dense         nonnorm       C/F")
(doseq [r results :when (not (:skipped? r)), c (:cells r)]
  (println (format "  %-5d %.2f  %d/%-2d%s  %-18s %-13s %-15s %-13s %s"
                   (:d r) (:rho c) (:n-with-fixed-point c) (:n-run c)
                   (if (:complete? c) " " "*")
                   (if-let [m (:margin c)]
                     (format "%.3f ± %.3f" (:mean m) (:std m)) "—")
                   (pr-str (mapv :op-norm (:dense c)))
                   (pr-str (mapv :rho-spec-dense (:dense c)))
                   (pr-str (mapv :nonnormality (:dense c)))
                   (or (some-> c :cf :cf-true str) "—"))))
(doseq [r results :when (:skipped? r)]
  (println (format "  %-5d skipped (%s)" (:d r) (name (:reason r)))))

(spit "scripts/spectral_dsweep_torch.edn"
      (with-out-str
        (pp/pprint {:config {:rungs RUNGS :rhos RHOS :seeds (vec SEEDS)
                             :seq-len SEQ-LEN :head-dim 8 :dense-n DENSE-N
                             :schedules {:settle {:tol 1e-9 :cap 20000 :probe 2000}
                                         :mf-spectral {:iters 160 :warmup 70}
                                         :op-norm-jg-iters 150 :op-norm-jf-iters 300}
                             :budget-min BUDGET-MIN}
                    :elapsed-min (r3 (elapsed-min))
                    :control-vs-seed-verify (if (nil? control-ok?) :not-run
                                                (if control-ok? :match :mismatch))
                    :rungs results})))
(println "\nsaved scripts/spectral_dsweep_torch.edn")
(println (format "total wall: %.1f min" (elapsed-min)))
(System/exit 0)
