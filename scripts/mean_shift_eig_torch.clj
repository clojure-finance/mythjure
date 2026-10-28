;; v2-review follow-up (2026-10-04, review-2 pt 7 + pt 4): structural
;; consequences of LayerNorm's per-token shift invariance, verified on the
;; dense recurrence Jacobian of record (seed-7 block, E seed 99, d=32 seq 16,
;; float64 — the spectral_eig_torch.clj configuration).
;;  Part A (pt 7): J_f annihilates per-token mean-shift directions, so for
;;    SCALAR carry rho*I each mean-shift direction is an EXACT eigenvector of
;;    J_G with eigenvalue rho => rho_spec(J_G) >= rho and spectral margin
;;    <= 1 - rho, architecturally. Verify: ||J_G v - rho v|| ~ 0 for all 16
;;    token mean-shift directions; >= 16 eigenvalues within tol of rho; report
;;    the quotient (centered-dynamics) spectral radius with those removed.
;;  Part B (pt 7 correction): for NON-uniform diagonal carry, J_G v = D v is
;;    not a mean-shift vector, so the mean direction re-enters the centered
;;    dynamics; verify ||J_f v|| ~ 0 still, and measure how far D v departs
;;    from the mean-shift line.
;;  Part C (pt 4): the "input re-injection floor" is trajectory-relative, not
;;    structural: at the cancellation state H = c*1 - E the normalizer input
;;    H + E = c*1 has zero spread although E's own spread is healthy. Measure
;;    Lip(f) and rho_spec(J_G) there.
;; Run: clojure -M:torch scripts/mean_shift_eig_torch.clj
(require '[mythjure.block :as blk]
         '[mythjure.linalg :as la]
         '[mythjure.torch.core :as tc]
         '[mythjure.torch.tensor :as t]
         '[mythjure.analysis-torch :as ant]
         '[clojure.pprint :as pp])

(tc/initialize!)
(def linalg (tc/attr tc/torch "linalg"))
(defn r4 [x] (Double/parseDouble (format "%.4f" x)))

(def p (blk/init-params {:d-model 32 :n-heads 4 :seed 7}))
(def E (la/rand-matrix 99 16 32 1.0))
(def E-t (t/from-clj E :dtype :float64))
(def S 16) (def D 32) (def N (* S D))

(defn eig-pairs [J]
  (let [ev (tc/call linalg "eigvals" J)
        re (vec (t/to-clj (tc/call tc/torch "real" ev)))
        im (vec (t/to-clj (tc/call tc/torch "imag" ev)))]
    (mapv vector re im)))

(defn mean-shift-v
  "Unit vector: token s, all D channels equal (token-major flattening)."
  [s]
  (let [c (/ 1.0 (Math/sqrt D))]
    (vec (for [i (range N)] (if (<= (* s D) i (dec (* (inc s) D))) c 0.0)))))

(defn matvec [J v]
  (vec (t/to-clj (tc/call tc/torch "matmul" J (t/from-clj v :dtype :float64)))))

(defn vnorm [v] (Math/sqrt (reduce + (map #(* % %) v))))
(defn v- [a b] (mapv - a b))

(defn d-full-vec [abar] (vec (apply concat (repeat S abar))))

;; --- Part A: scalar carry, exact eigenvalue rho on mean-shift directions ---
(println "=== Part A: scalar carry — rho is an exact eigenvalue on mean-shift directions ===")
(def partA
  (into (sorted-map)
        (for [rho [0.3 0.5 0.7 0.9 0.95]]
          (let [G  (ant/gated-update-t p E rho)
                st (ant/settle-t G E-t)
                _  (assert (:converged? st))
                J  (ant/jacobian-matrix-t G (:H st))
                resid (apply max (for [s (range S)]
                                   (let [v (mean-shift-v s)]
                                     (vnorm (v- (matvec J v) (mapv #(* rho %) v))))))
                pairs (eig-pairs J)
                dist  (fn [[re im]] (Math/sqrt (+ (* (- re rho) (- re rho)) (* im im))))
                sorted (sort-by dist pairs)
                close  (take S sorted)
                rest-  (drop S sorted)
                n-close (count (filter #(< (dist %) 1e-6) pairs))
                quot-r (apply max (map (fn [[re im]] (Math/sqrt (+ (* re re) (* im im)))) rest-))
                full-r (apply max (map (fn [[re im]] (Math/sqrt (+ (* re re) (* im im)))) pairs))
                row {:max-eigvec-resid resid
                     :n-eigs-within-1e6-of-rho n-close
                     :worst-close-dist (apply max (map dist close))
                     :rho-spec-full (r4 full-r)
                     :rho-spec-quotient (r4 quot-r)
                     :margin-bound (r4 (- 1.0 rho))}]
            (println (format "rho=%.2f  eigvec-resid %.1e  #eigs@rho %d/%d  rho_spec full %.4f  quotient %.4f  (margin <= %.2f)"
                             rho resid n-close S full-r quot-r (- 1.0 rho)))
            [rho row]))))

;; --- Part B: non-uniform carry — J_f still kills mean shifts; D v leaves the mean line ---
(println "\n=== Part B: non-uniform carry — mean direction re-enters the centered dynamics ===")
(def profiles
  {:one-slow (assoc (vec (repeat D 0.5)) 0 0.99)
   :ramp     (mapv #(+ 0.5 (* 0.49 (/ % 31.0))) (range D))})
(def partB
  (into (sorted-map)
        (for [[k a] (sort-by key profiles)]
          (let [G  (ant/gated-update-t p E a)
                st (ant/classify-settle-t G E-t)
                _  (assert (= :converged (:status st)) (str k " did not settle"))
                J  (ant/jacobian-matrix-t G (:H st))
                dv (d-full-vec a)
                jf-resid (apply max
                                (for [s (range S)]
                                  (let [v  (mean-shift-v s)
                                        Dv (mapv * dv v)]
                                    (vnorm (v- (matvec J v) Dv)))))
                ;; component of D v off the mean-shift line (relative)
                off (let [v (mean-shift-v 0)
                          Dv (mapv * dv v)
                          m  (/ (reduce + Dv) (reduce + (map #(if (pos? %) 1.0 0.0) v)))
                          proj (mapv #(if (pos? %) m 0.0) v)]
                      (/ (vnorm (v- Dv proj)) (vnorm Dv)))
                row {:max-Jf-meanshift-resid jf-resid
                     :Dv-off-meanshift-relative (r4 off)}]
            (println (format "%-9s ||J_f v|| %.1e   off-line fraction of D v: %.4f" (name k) jf-resid off))
            [k row]))))

;; --- Part C: cancellation state — the implosion floor is not structural ---
(println "\n=== Part C: cancellation state H = c*1 - E (normalizer input spread -> 0) ===")
(defn min-token-spread [M]
  (apply min (for [row M]
               (let [mu (/ (reduce + row) (count row))
                     v  (/ (reduce + (map #(let [d (- % mu)] (* d d)) row)) (count row))]
                 (Math/sqrt v)))))
(def partC
  (into (sorted-map)
        (for [rho [0.3 0.9]]
          (let [G   (ant/gated-update-t p E rho)
                Hc  (mapv (fn [erow] (mapv #(- 0.5 %) erow)) E)
                Hct (t/from-clj Hc :dtype :float64)
                J   (ant/jacobian-matrix-t G Hct)
                dv  (d-full-vec (vec (repeat D rho)))
                Jf-op (apply max (t/to-clj (tc/call linalg "svdvals"
                                                    (tc/call tc/torch "sub" J (tc/call tc/torch "diag" (t/from-clj dv :dtype :float64))))))
                rho-spec (apply max (map (fn [[re im]] (Math/sqrt (+ (* re re) (* im im)))) (eig-pairs J)))
                row {:spread-E (r4 (min-token-spread E))
                     :spread-Hc+E (min-token-spread (mapv (fn [h e] (mapv + h e)) Hc E))
                     :Lip-f-at-Hc (r4 Jf-op)
                     :rho-spec-at-Hc (r4 rho-spec)}]
            (println (format "rho=%.2f  spread(E) %.3f  spread(Hc+E) %.1e  Lip(f)@Hc %.1f  rho_spec@Hc %.1f"
                             rho (:spread-E row) (:spread-Hc+E row) Jf-op rho-spec))
            [rho row]))))

(spit "scripts/mean_shift_eig_torch.edn"
      (with-out-str (pp/pprint {:partA-scalar partA :partB-nonuniform partB :partC-cancellation partC})))
(println "\nWrote scripts/mean_shift_eig_torch.edn")
(System/exit 0)
