;; Check of an externally proposed margin-sandwich theorem (2026-10-09):
;; claim: rho <= rho_spec(J_G) <= rho + max_s ||J_ss||_2, with J_ss the
;; per-token self-Jacobian (diagonal d x d block of J_G minus rho*I), so
;; K_meas := max_s ||J_ss||_2 / (1-rho) < 1 would certify contraction at
;; every carry; failing that, the field-of-values refinement
;;   rho_spec(rho I + J_ss) <= sqrt(rho^2 + 2 rho mu2(J_ss) + ||J_ss||_2^2)
;; (mu2 = log-norm = lambda_max of the symmetric part) certifies when the
;; leading m_meas := max_s mu2(J_ss) / (1-rho) < 1 and ||J_ss|| = O(1-rho).
;; Also checks the proposal's hypothesis (H): per-token increment spread
;; phi := min_s r(f_s) > 0 at the fixed point, r(v) = ||Pv||/sqrt(d),
;; f_s = (1-rho) h*_s exactly (scalar carry), against the injection spread
;; r_e := max_s r(e_s) (step-1 floor valid iff phi/(1-rho) > r_e).
;; Config = the paper's representative instance (block seed 7, E seed 99,
;; d=32 seq 16, float64), same as scripts/spectral_eig_torch.clj part 1.
;; Run:  clojure -M:torch scripts/sandwich_check.clj
(require '[mythjure.block :as blk]
         '[mythjure.linalg :as la]
         '[mythjure.torch.core :as tc]
         '[mythjure.torch.tensor :as t]
         '[mythjure.analysis-torch :as ant]
         '[clojure.pprint :as pp])

(tc/initialize!)
(def linalg (tc/attr tc/torch "linalg"))

(defn r4 [x] (Double/parseDouble (format "%.4f" x)))
(defn eig-abs [J] (vec (t/to-clj (tc/call tc/torch "abs" (tc/call linalg "eigvals" J)))))
(defn rho-eig [J] (apply max (eig-abs J)))
(defn op-svd  [J] (apply max (t/to-clj (tc/call linalg "svdvals" J))))
(defn mu2
  "Log-norm: largest eigenvalue of the symmetric part (J + J^T)/2."
  [J]
  (let [Jt (tc/call tc/torch "transpose" J 0 1)
        sym (tc/call tc/torch "div" (tc/call tc/torch "add" J Jt) 2.0)]
    (apply max (t/to-clj (tc/call linalg "eigvalsh" sym)))))

(def p (blk/init-params {:d-model 32 :n-heads 4 :seed 7}))
(def E (la/rand-matrix 99 16 32 1.0))
(def E-t (t/from-clj E :dtype :float64))
(def S 16) (def D 32)

(defn spread
  "r(v) = ||Pv|| / sqrt(d) for one token row (centered RMS)."
  [row]
  (let [n (count row) mu (/ (reduce + row) n)]
    (Math/sqrt (/ (reduce + (map #(let [c (- % mu)] (* c c)) row)) n))))

(def r-e (apply max (map spread E)))

(println "=== margin-sandwich check: seed 7, d=32, seq 16, five carries ===")
(println (format "injection spread r_e = max_s r(e_s) = %.4f" r-e))

(def results
  (into (sorted-map)
        (for [rho [0.3 0.5 0.7 0.9 0.95]]
          (let [G  (ant/gated-update-t p E rho)
                st (ant/settle-t G E-t)
                _  (assert (:converged? st) (str "rho " rho " did not settle"))
                H* (:H st)
                J  (ant/jacobian-matrix-t G H*)
                rho-spec (rho-eig J)
                ;; per-token self-blocks: J_ss = diag block - rho I
                I-d (t/from-clj (vec (for [i (range D)]
                                       (vec (for [j (range D)] (if (= i j) 1.0 0.0)))))
                                :dtype :float64)
                blocks (for [s (range S)]
                         (tc/call tc/torch "sub"
                                  (t/narrow (t/narrow J 0 (* s D) D) 1 (* s D) D)
                                  (tc/call tc/torch "mul" I-d rho)))
                rows (vec (for [[s Jss] (map-indexed vector blocks)]
                            (let [op (op-svd Jss) m2 (mu2 Jss)
                                  ;; shift-invariance: J_ss annihilates 1
                                  ones (t/from-clj (vec (repeat D 1.0)) :dtype :float64)
                                  k1 (t/norm (tc/call tc/torch "matmul" Jss ones))
                                  fov (Math/sqrt (max 0.0 (+ (* rho rho) (* 2 rho m2) (* op op))))
                                  blk-rho (rho-eig (tc/call tc/torch "add" Jss (tc/call tc/torch "mul" I-d rho)))]
                              {:s s :op-norm op :mu2 m2 :kills-ones k1
                               :fov-bound fov :op-bound (+ rho op) :block-rho-spec blk-rho})))
                worst-op (apply max-key :op-norm rows)
                worst-m2 (apply max-key :mu2 rows)
                k-meas (/ (:op-norm worst-op) (- 1.0 rho))
                m-meas (/ (:mu2 worst-m2) (- 1.0 rho))
                fov-max (apply max (map :fov-bound rows))
                op-max  (apply max (map :op-bound rows))
                ;; hypothesis (H): per-token increment spread at the fixed point
                H*-rows (t/to-clj H*)
                r-h (mapv spread H*-rows)
                phi (* (- 1.0 rho) (apply min r-h))     ; r(f_s) = (1-rho) r(h*_s)
                floor-ok? (> (/ phi (- 1.0 rho)) r-e)
                kills-max (apply max (map :kills-ones rows))
                row {:rho-spec (r4 rho-spec)
                     :k-meas (r4 k-meas) :m-meas (r4 m-meas)
                     :op-bound-max (r4 op-max) :fov-bound-max (r4 fov-max)
                     :op-certifies? (< op-max 1.0) :fov-certifies? (< fov-max 1.0)
                     :phi (r4 phi) :r-h-min (r4 (apply min r-h)) :step1-floor-valid? floor-ok?
                     :jss-kills-ones-max (format "%.1e" kills-max)}]
            (println (format "rho=%.2f  rho_spec %.4f | K_meas %.3f  m_meas %.3f | op-bound %.4f (certifies %s)  FoV-bound %.4f (certifies %s) | phi %.4f  floor-valid %s  Jss*1 max %s"
                             rho rho-spec k-meas m-meas op-max (:op-certifies? row)
                             fov-max (:fov-certifies? row) phi floor-ok?
                             (:jss-kills-ones-max row)))
            [rho row]))))

(println "\nSummary: bound is valid iff op-bound/FoV-bound >= rho_spec at every carry:")
(doseq [[rho r] results]
  (println (format "  rho=%.2f  valid(op) %s  valid(FoV) %s  slack(FoV) %.4f"
                   rho
                   (>= (:op-bound-max r) (:rho-spec r))
                   (>= (:fov-bound-max r) (:rho-spec r))
                   (- (:fov-bound-max r) (:rho-spec r)))))

(spit "scripts/sandwich_check.edn" (with-out-str (pp/pprint results)))
(println "\nWrote scripts/sandwich_check.edn")
