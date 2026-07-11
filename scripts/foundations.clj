;; Foundation numbers behind §3–§5 of the write-up — the analytical/illustrative
;; results that live in REPL functions rather than the training scripts. These
;; are deterministic (seeded) verification checks on toy, random-weight models,
;; not gradient-descent experiments, so they run in a second or two:
;;   §3   dynamics: unconstrained loop explodes vs Parcae-constrained stays bounded
;;   §4.1 nonlinear: bounded R can't save an unconstrained carry; Banach
;;        contraction threshold in the gain; the Parcae spiral geometry
;;   §4–5 lipschitz/analysis: Lip(block) ∝ 1/‖H‖ (activation-implosion danger
;;        zone); the two contraction laws + shrinking margin as ρ→1; per-channel
;;        carry governed by the MAX (slowest) channel
;; Run:  clojure -M scripts/foundations.clj
(require '[mythjure.dynamics :as dyn]
         '[mythjure.nonlinear :as nl]
         '[mythjure.block :as blk]
         '[mythjure.linalg :as la]
         '[mythjure.analysis :as an]
         '[clojure.pprint :as pp])

(defn- last-norm [m] (last (:norms m)))

;; --- §3  discretization: explosion vs bounded -----------------------------
(println "=== §3  dynamics: unconstrained carry explodes, Parcae carry stays bounded ===")
(def explosion (dyn/demo-explosion))          ; carry 1.3, T=20
(def stable    (dyn/demo-stable))             ; log_A=0, dt=1 ⇒ ā∈(0,1)
(println (format "  unconstrained ρ=%.3f  ‖h‖: %s  final=%.1f"
                 (:spectral-radius explosion) (dyn/sparkline (:norms explosion)) (last-norm explosion)))
(println (format "  constrained   ρ=%.3f  ‖h‖: %s  final=%.2f"
                 (:spectral-radius stable) (dyn/sparkline (:norms stable)) (last-norm stable)))

;; --- §4.1 nonlinear: carry is load-bearing, contraction threshold, spiral --
(println "\n=== §4.1 nonlinear R(h,e)=tanh(Wh+e) in the loop ===")
(def load-bearing (nl/demo-carry-is-load-bearing))
(println (format "  bounded R can't save an unconstrained carry: %s" (pr-str load-bearing)))
(def sweep (nl/contraction-sweep))
(println "  Banach contraction threshold (gain → converges?):")
(doseq [{:keys [gain final-step converges?]} sweep]
  (println (format "    gain %.3f  final-step %.4g  %s" gain final-step
                   (if converges? "converges" "wanders"))))
(def spiral (nl/spiral-metrics))
(println (format "  spiral: step sizes %s" (pr-str (:step-sizes spiral))))
(println (format "          cos-vs-first %s" (pr-str (:cos-vs-first spiral))))

;; Banach threshold prediction: ρ + gain·‖W‖_σ < 1 for the unit-gain W of the
;; toy (seed 42, 8×8). ‖W‖_σ via power iteration on the materialized matrix.
(def W-unit (nl/rand-matrix 42 8 8 1.0))
(def W-sigma (an/operator-norm W-unit))
(def banach-threshold (/ (- 1.0 (Math/exp -1.0)) W-sigma))
(println (format "  Banach: ‖W‖_σ=%.3f (unit gain, seed 42) ⇒ predicted convergence for gain < %.3f"
                 W-sigma banach-threshold))

;; Spiral PCA projection (paper Fig. 1a): trajectory of the gain-0.25 loop,
;; centered on the fixed point (final state), projected on its top two PCs.
(defn- pca-2d [states]
  (let [h*   (last states)
        X    (mapv #(mapv - % h*) states)
        d    (count h*)
        C    (vec (for [i (range d)]
                    (vec (for [j (range d)]
                           (reduce + (map #(* (nth % i) (nth % j)) X))))))
        mv   (fn [M v] (mapv #(reduce + (map * % v)) M))
        nrm  (fn [v] (Math/sqrt (reduce + (map #(* % %) v))))
        powi (fn [M] (loop [k 0 v (vec (repeat d (/ 1.0 (Math/sqrt d))))]
                       (if (= k 200) v
                           (let [w (mv M v)] (recur (inc k) (mapv #(/ % (nrm w)) w))))))
        v1   (powi C)
        l1   (reduce + (map * v1 (mv C v1)))
        C2   (vec (for [i (range d)]
                    (vec (for [j (range d)]
                           (- (get-in C [i j]) (* l1 (nth v1 i) (nth v1 j)))))))
        v2   (powi C2)
        prj  (fn [x] [(reduce + (map * x v1)) (reduce + (map * x v2))])
        pts  (mapv prj X)
        ;; orient PCs so the first point lands in the lower-left quadrant
        [sx sy] (let [[x0 y0] (first pts)] [(if (pos? x0) -1.0 1.0) (if (pos? y0) -1.0 1.0)])]
    (mapv (fn [[x y]] [(Double/parseDouble (format "%.4f" (* sx x)))
                       (Double/parseDouble (format "%.4f" (* sy y)))]) pts)))
(def spiral-pca (vec (take 28 (pca-2d (nl/run {:gain 0.25 :T 40})))))
(println (format "  spiral PCA (first 4 of 28 pts): %s" (pr-str (take 4 spiral-pca))))

;; --- §4–5  Lipschitz / contraction diagnostics around a real block --------
(println "\n=== §4–5  Lipschitz diagnostics (d=32, 4 heads, seq 16, random weights) ===")
(def p (blk/init-params {:d-model 32 :n-heads 4 :seed 7}))
(def E (la/rand-matrix 99 16 32 1.0))

(def scale-prof (an/scale-profile p E))
(println "  Lip(block) vs activation scale (small ‖H‖ = implosion danger zone):")
(doseq [[c {:keys [H-norm Lip-block Lip-incr]}] scale-prof]
  (println (format "    c=%-4s ‖H‖=%-7s Lip-block=%-6s Lip-incr=%s" c H-norm Lip-block Lip-incr)))

(def gated (an/gated-contraction p E))
(println "  gated contraction per scalar carry ρ (margin = 1 − Lip(G)):")
(doseq [[rho {:keys [Lip-G margin bound-ρ+LipI]}] gated]
  (println (format "    ρ=%.2f  Lip(G)=%-6s margin=%-6s ρ+Lip(incr)=%s" rho Lip-G margin bound-ρ+LipI)))

;; per-channel: the worst-axis margin is set by the MAX carry, not the mean.
;; Four profiles (paper Tab. per-channel): uniform 0.5, one slow channel @0.99,
;; a 0.5→0.99 ramp, and uniform 0.99.
(def a-bar-uniform-05 (vec (repeat 32 0.5)))
(def a-bar-one-slow   (assoc (vec (repeat 32 0.5)) 0 0.99))
(def a-bar-ramp       (mapv #(+ 0.5 (* 0.49 (/ % 31.0))) (range 32)))
(def a-bar-uniform-99 (vec (repeat 32 0.99)))
(def per-ch-profiles
  (into (sorted-map)
        (for [[k a] {:uniform-0.5 a-bar-uniform-05
                     :one-slow    a-bar-one-slow
                     :ramp        a-bar-ramp
                     :uniform-99  a-bar-uniform-99}]
          [k (an/per-channel-contraction p E a)])))
(def per-ch (:one-slow per-ch-profiles))
(println "  per-channel carry profiles — max governs, not mean:")
(doseq [[k v] per-ch-profiles]
  (println (format "    %-12s %s" (name k) (pr-str v))))

;; --- persist ---------------------------------------------------------------
(spit "scripts/foundations.edn"
      (with-out-str
        (pp/pprint {:s2-explosion (select-keys explosion [:spectral-radius :norms])
                    :s2-stable    (select-keys stable [:spectral-radius :norms])
                    :s21-carry-load-bearing load-bearing
                    :s21-contraction-sweep sweep
                    :s21-spiral (select-keys spiral [:step-sizes :cos-vs-first])
                    :s21-banach {:W-sigma (Double/parseDouble (format "%.3f" W-sigma))
                                 :threshold (Double/parseDouble (format "%.3f" banach-threshold))}
                    :s21-spiral-pca spiral-pca
                    :s3-scale-profile scale-prof
                    :s3-gated-contraction gated
                    :s3-per-channel per-ch
                    :s3-per-channel-profiles per-ch-profiles})))
(println "\n--- foundations done (wrote scripts/foundations.edn) ---")
(System/exit 0)
