;; E5 of attic/paper-v2-plan.md §C (review-2 pt 7): is "activation implosion"
;; a measured instability or only an upper-bound story? From the §5.3 run of
;; record (block seed 7, E seed 99, d=32 seq 16, f64), settle to h* at
;; rho ∈ {0.3, 0.9}, then scale the state toward zero (alpha grid) in two
;; modes — :scale-h (H = alpha·h*, injection E fixed: the state implodes but
;; the re-injected input does not, so the normalizer input H+E bottoms out at
;; E-scale) and :scale-both (H = alpha·h*, E → alpha·E: the whole stream
;; implodes — the regime §6 actually fears). At each scaled state measure:
;; the TRUE Lip(f) (= ‖J_f‖₂ by direct SVD of the dense increment Jacobian),
;; ‖J_G‖₂ (SVD), rho_spec(J_G) (direct eig), the minimal per-token spread
;; s_min(H+E) (the real scale variable), the finite-T amplification of a
;; random perturbation carried by JVPs along the actual T=8 trajectory from
;; the scaled state, and where that trajectory then settles.
;; Run:  clojure -M:torch scripts/implosion_probe.clj
(require '[mythjure.block :as blk]
         '[mythjure.linalg :as la]
         '[mythjure.torch.core :as tc]
         '[mythjure.torch.tensor :as t]
         '[mythjure.analysis-torch :as ant]
         '[clojure.pprint :as pp])

(tc/initialize!)
(def linalg (tc/attr tc/torch "linalg"))
(defn r4 [x] (Double/parseDouble (format "%.4f" x)))
(defn rho-eig [J] (apply max (t/to-clj (tc/call tc/torch "abs" (tc/call linalg "eigvals" J)))))
(defn op-svd  [J] (apply max (t/to-clj (tc/call linalg "svdvals" J))))

(def p (blk/init-params {:d-model 32 :n-heads 4 :seed 7}))
(def E (la/rand-matrix 99 16 32 1.0))
(def E-t (t/from-clj E :dtype :float64))
(def T 8)

(defn s-min
  "Minimal per-token spread (std) of H+E — the LayerNorm scale variable."
  [Ht Et]
  (let [Z (t/to-clj (t/add Ht Et))]
    (apply min (for [row Z]
                 (let [n (count row) mu (/ (reduce + row) n)]
                   (Math/sqrt (/ (reduce + (map #(let [d (- % mu)] (* d d)) row)) n)))))))

(defn finite-T-amp
  "Carry a random unit perturbation along the actual trajectory from H0 via
  JVPs: v_{t+1} = J_G(h_t)·v_t, h_{t+1} = G(h_t). Returns ‖v_T‖."
  [G-t H0 seed]
  (let [rng (java.util.Random. (long seed))
        [S d] (t/shape H0)
        v0 (t/from-clj (vec (for [_ (range S)] (vec (repeatedly d #(.nextGaussian rng))))) :dtype :float64)
        v0 (t/mul v0 (/ 1.0 (t/norm v0)))]
    (loop [k 0, h H0, v v0]
      (if (= k T)
        (t/norm v)
        (recur (inc k) (G-t h) (ant/jvp-t G-t h v))))))

(def alphas [1.0 0.3 0.1 0.03 0.01 0.003])

(def out
  (into (sorted-map)
        (for [rho [0.3 0.9]]
          (let [G1  (ant/gated-update-t p E rho)
                st  (ant/settle-t G1 E-t)
                _   (assert (:converged? st))
                H*  (:H st)]
            (println (format "=== rho=%.1f  ‖h*‖=%.2f (settled, %d iters) ===" rho (t/norm H*) (:iters st)))
            [rho
             (into (sorted-map)
                   (for [mode [:scale-h :scale-both]]
                     [mode
                      (vec (for [a alphas]
                             (let [Ea  (if (= mode :scale-both) (t/mul E-t a) E-t)
                                   Eaj (t/to-clj Ea)
                                   G   (if (= mode :scale-both) (ant/gated-update-t p Eaj rho) G1)
                                   f   (ant/increment-t p Eaj)
                                   Ha  (t/mul H* a)
                                   Jf  (ant/jacobian-matrix-t f Ha)
                                   Jg  (ant/jacobian-matrix-t G Ha)
                                   lip (op-svd Jf)
                                   og  (op-svd Jg)
                                   re  (rho-eig Jg)
                                   amp (finite-T-amp G Ha 11)
                                   st2 (ant/classify-settle-t G Ha :cap 5000 :probe 1000)
                                   row {:alpha a :s-min (r4 (s-min Ha Ea))
                                        :lip-f (r4 lip) :op-norm-JG (r4 og) :rho-spec-JG (r4 re)
                                        :amp-T8 (r4 amp)
                                        :settle {:status (:status st2) :iters (:iters st2)
                                                 :H-norm (r4 (t/norm (:H st2)))}}]
                               (println (format "  %-11s a=%-6s s_min=%-8s Lip(f)=%-9s ‖J_G‖=%-9s rho_spec=%-8s amp_T8=%-9s settle=%s ‖H_end‖=%.2f"
                                                (name mode) a (:s-min row) (:lip-f row) (:op-norm-JG row)
                                                (:rho-spec-JG row) (:amp-T8 row)
                                                (name (get-in row [:settle :status])) (get-in row [:settle :H-norm])))
                               row)))]))]))))

(spit "scripts/implosion_probe.edn" (with-out-str (pp/pprint out)))
(println "\nWrote scripts/implosion_probe.edn")
