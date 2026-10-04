;; E2+E3 of attic/paper-v2-plan.md §C (review-2 pts 5+8): direct
;; eigendecomposition / SVD cross-check of every power-iteration spectral
;; measurement of record, the causal block-lower-triangular structure check
;; (plan B3: upper token blocks exactly zero; spectrum = union of per-token
;; diagonal-block spectra), and TRUE per-channel-profile operator norms /
;; spectral radii to replace tab:per-channel's probe-based margins.
;; Mirrors scripts/spectral_torch.clj's configs exactly (block seed 7, E seed
;; 99, d=32 seq 16 f64; sweep seeds 1-10 at rho 0.3/0.9) and diffs against the
;; committed scripts/spectral_torch.edn where values exist.
;; Run:  clojure -M:torch scripts/spectral_eig_torch.clj
(require '[mythjure.block :as blk]
         '[mythjure.linalg :as la]
         '[mythjure.torch.core :as tc]
         '[mythjure.torch.tensor :as t]
         '[mythjure.analysis-torch :as ant]
         '[clojure.edn :as edn]
         '[clojure.pprint :as pp])

(tc/initialize!)
(def linalg (tc/attr tc/torch "linalg"))

(defn r4 [x] (Double/parseDouble (format "%.4f" x)))
(defn eig-abs [J] (vec (t/to-clj (tc/call tc/torch "abs" (tc/call linalg "eigvals" J)))))
(defn rho-eig [J] (apply max (eig-abs J)))
(defn op-svd  [J] (apply max (t/to-clj (tc/call linalg "svdvals" J))))

(defn tri-check
  "Causal structure: max Frobenius norm over strictly-upper token blocks
  (exactly 0 if causal masking makes J block lower triangular) and the max
  deviation between the sorted full spectrum and the sorted union of the
  per-token diagonal blocks' spectra."
  [J S d]
  (let [blk-of (fn [r c] (t/narrow (t/narrow J 0 (* r d) d) 1 (* c d) d))
        upper  (if (> S 1)
                 (apply max (for [r (range S) c (range S) :when (> c r)]
                              (t/norm (blk-of r c))))
                 0.0)
        de     (sort > (mapcat (fn [s] (eig-abs (blk-of s s))) (range S)))
        fe     (sort > (eig-abs J))
        dev    (apply max (map (fn [a b] (Math/abs (- a b))) de fe))]
    {:upper-block-max-norm upper :spectra-max-dev dev}))

(def record (edn/read-string (slurp "scripts/spectral_torch.edn")))
(def p (blk/init-params {:d-model 32 :n-heads 4 :seed 7}))
(def E (la/rand-matrix 99 16 32 1.0))
(def E-t (t/from-clj E :dtype :float64))
(def S 16) (def D 32)

;; --- Part 1: seed-7 representative instance, five carries (tab:spectral) ---
(println "=== Part 1: seed 7, direct eig/SVD vs power-iteration record ===")
(def part1
  (into (sorted-map)
        (for [rho [0.3 0.5 0.7 0.9 0.95]]
          (let [G  (ant/gated-update-t p E rho)
                st (ant/settle-t G E-t)
                _  (assert (:converged? st) (str "seed 7 rho " rho " did not settle"))
                J  (ant/jacobian-matrix-t G (:H st))
                re (rho-eig J) os (op-svd J)
                tri (tri-check J S D)
                rec (get-in record [:spectral-analysis rho])
                row {:rho-spec-eig (r4 re) :op-norm-svd (r4 os)
                     :rec-rho-spec (:rho-spec rec) :rec-op-norm (:op-norm rec)
                     :d-rho (r4 (Math/abs (- re (:rho-spec rec))))
                     :d-op  (r4 (Math/abs (- os (:op-norm rec))))
                     :tri tri :settle-iters (:iters st)}]
            (println (format "rho=%.2f  eig %.4f (rec %.3f, d %.4f)   svd %.4f (rec %.3f, d %.4f)   upper %.1e  specdev %.1e"
                             rho re (:rho-spec rec) (:d-rho row) os (:op-norm rec) (:d-op row)
                             (:upper-block-max-norm tri) (:spectra-max-dev tri)))
            [rho row]))))

;; --- Part 2: seed robustness, direct values (tab:seed-robustness) ----------
(println "\n=== Part 2: seeds 1-10 at rho 0.3 / 0.9, direct eig/SVD ===")
(def part2
  (into (sorted-map)
        (for [rho [0.3 0.9]]
          (let [runs (vec (for [s (range 1 11)]
                            (let [ps (blk/init-params {:d-model D :n-heads 4 :seed s})
                                  G  (ant/gated-update-t ps E rho)
                                  st (ant/classify-settle-t G E-t)]
                              (if (= :non-convergent (:status st))
                                (do (println (format "rho=%.1f seed %d NON-CONVERGENT (resid %.1e)" rho s (:resid st)))
                                    {:seed s :status (:status st)})
                                (let [J (ant/jacobian-matrix-t G (:H st))
                                      re (rho-eig J) os (op-svd J)]
                                  (println (format "rho=%.1f seed %d %-10s eig %.4f  svd %.4f" rho s (name (:status st)) re os))
                                  {:seed s :status (:status st)
                                   :rho-spec-eig (r4 re) :op-norm-svd (r4 os)})))))
                ok   (filterv :rho-spec-eig runs)
                mean (fn [k] (r4 (/ (reduce + (map k ok)) (count ok))))
                agg  {:n-with-fixed-point (count ok)
                      :non-convergent (mapv :seed (remove :rho-spec-eig runs))
                      :rho-spec-eig-mean (mean :rho-spec-eig)
                      :op-norm-svd-mean  (mean :op-norm-svd)
                      :n-above-1 (count (filter #(> (:rho-spec-eig %) 1.0) ok))
                      :rec (select-keys (get-in record [:seed-robustness rho])
                                        [:rho-spec-mean :n-above-1 :n-with-fixed-point])}]
            (println (format "rho=%.1f AGG  eig-mean %.4f (rec %.3f)  svd-mean %.4f  above-1 %d/%d  non-conv %s"
                             rho (:rho-spec-eig-mean agg) (get-in agg [:rec :rho-spec-mean])
                             (:op-norm-svd-mean agg) (:n-above-1 agg) (count ok)
                             (pr-str (:non-convergent agg))))
            [rho {:runs runs :agg agg}]))))

;; --- Part 3 (E3): per-channel profiles, TRUE norms (tab:per-channel) -------
(println "\n=== Part 3: per-channel profiles, true op-norm / rho-spec (replaces probe margins) ===")
(def profiles
  {:uniform-0.5 (vec (repeat D 0.5))
   :one-slow    (assoc (vec (repeat D 0.5)) 0 0.99)
   :ramp        (mapv #(+ 0.5 (* 0.49 (/ % 31.0))) (range D))
   :uniform-99  (vec (repeat D 0.99))})
(def probe-margins {:uniform-0.5 0.17 :one-slow 0.011 :ramp 0.010 :uniform-99 0.006})
(def part3
  (into (sorted-map)
        (for [[k a] (sort-by key profiles)]
          (let [G  (ant/gated-update-t p E a)
                st (ant/classify-settle-t G E-t)
                row (if (= :non-convergent (:status st))
                      {:status (:status st) :resid (:resid st)}
                      (let [J (ant/jacobian-matrix-t G (:H st))
                            re (rho-eig J) os (op-svd J)]
                        {:status (:status st) :settle-iters (:iters st)
                         :carry-max (apply max a) :carry-mean (r4 (/ (reduce + a) D))
                         :rho-spec-eig (r4 re) :op-norm-svd (r4 os)
                         :margin-spec (r4 (- 1.0 re)) :margin-op (r4 (- 1.0 os))
                         :probe-margin-paper (probe-margins k)}))]
            (println (format "%-12s %s" (name k) (pr-str row)))
            [k row]))))

(def out {:part1-seed7 part1 :part2-seeds part2 :part3-profiles part3})
(spit "scripts/spectral_eig_torch.edn" (with-out-str (pp/pprint out)))
(println "\nWrote scripts/spectral_eig_torch.edn")
