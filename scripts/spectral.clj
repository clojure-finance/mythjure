;; Spectral-radius / non-normality measurements behind §5.3 (directional margin)
;; and Appendix A of the write-up — the two numbers the paper had flagged as
;; "leave to future work". Deterministic (seeded) probes on the toy, random-weight
;; block at the fixed point, same setup as scripts/foundations.clj §3
;; (d=32, 4 heads, seq 16, seed 7). Runs ~80 min: the seed sweep's non-convergent
;; seeds burn the full 20k-iteration residual-settle cap (the O(N) dense Jacobian
;; materialization at N=512 is secondary).
;;
;;   (1) spectral radius ρ_spec(J_G(H*)) vs. true operator norm ‖J_G‖₂
;;       — separates the loop's ASYMPTOTIC per-iteration convergence rate (ρ_spec,
;;         the r_eff of Parcae's test-time scaling law) from its worst-case one-step
;;         amplification (‖J_G‖₂). Their gap is the non-normality of J_G.
;;   (2) input-injection scale ratio ‖H*+E‖/‖H*‖ across ρ
;;       — tests the Appendix-A claim that at low carry the injection E inflates the
;;         normalizer's scale variable, biasing the effective C/F constant.
;; Run:  clojure -M scripts/spectral.clj
(require '[mythjure.block :as blk]
         '[mythjure.linalg :as la]
         '[mythjure.analysis :as an]
         '[clojure.pprint :as pp])

(def p (blk/init-params {:d-model 32 :n-heads 4 :seed 7}))
(def E (la/rand-matrix 99 16 32 1.0))

;; --- (1) spectral radius vs operator norm --------------------------------
(println "=== spectral radius ρ_spec(J_G) vs operator norm ‖J_G‖₂ at the fixed point ===")
(println "    (d=32, 4 heads, seq 16, random weights — same block as foundations §3)")
(def spec (an/spectral-analysis p E))
(println (format "  %-5s %-9s %-11s %-9s %-10s %s"
                 "ρ" "ρ_spec" "ρ_spec(dns)" "‖J_G‖₂" "probedLip" "non-normality ‖J_G‖₂−ρ_spec"))
(doseq [[rho {:keys [rho-spec rho-spec-dense op-norm probed-Lip nonnormality]}] spec]
  (println (format "  %-5.2f %-9s %-11s %-9s %-10s %s"
                   rho rho-spec rho-spec-dense op-norm probed-Lip nonnormality)))
(println "  reading: ρ_spec<1 at every ρ ⇒ the loop is asymptotically contractive; ")
(println "           ‖J_G‖₂>1 at low ρ ⇒ transient (non-normal) one-step amplification,")
(println "           the datum behind the step-size rebound in the spiral figure. The")
(println "           finite-difference probed Lip(G) under-estimates the true ‖J_G‖₂.")

;; --- (2) input-injection scale ratio -------------------------------------
(println "\n=== input-injection scale ratio ‖H*+E‖/‖H*‖ across ρ (Appendix A) ===")
(def inj (an/injection-ratio p E))
(println (format "  %-5s %-9s %-9s %s" "ρ" "‖H*‖" "‖E‖" "‖H*+E‖/‖H*‖"))
(doseq [[rho {:keys [H*-norm E-norm ratio]}] inj]
  (println (format "  %-5.2f %-9s %-9s %s" rho H*-norm E-norm ratio)))
(println "  reading: ratio→1 as ρ→1 (‖H*‖≫‖E‖); the ~23% inflation at ρ=0.3 is")
(println "           consistent in sign with the low-carry softening of the true C/F")
(println "           (2.33 vs peak 2.78); the probe-table dip to 0.87 is additionally")
(println "           confounded by the probe's own ~2.6x underestimate (App. A).")

;; --- (3) TRUE self-stabilization constant C/F ----------------------------
(println "\n=== TRUE C/F = ‖J_f‖₂/(1−ρ) via power iteration (vs finite-diff probe) ===")
(def cf (an/cf-analysis p E))
(println (format "  %-5s %-11s %-11s %-9s %-9s %s"
                 "ρ" "Lip(f)true" "Lip(f)probe" "F" "balance" "C/F true"))
(doseq [[rho {:keys [lipf-true lipf-probe F balance cf-true]}] cf]
  (println (format "  %-5.2f %-11s %-11s %-9s %-9s %s"
                   rho lipf-true lipf-probe F balance cf-true)))
(println "  reading: the probe under-estimates ‖J_f‖₂ ≈2.6× at every ρ, so the")
(println "           implied C/F≈1 of tab:directional-margin is a PROBE ARTIFACT;")
(println "           the true C/F is ≈2.5 (roughly flat), NOT ≈1. C/F is a constant")
(println "           (the derivation's structural claim) but its value is ≈2.5, so")
(println "           the operator-norm bound ρ+Lip(f)=ρ+2.5(1−ρ) runs 2.5→1, riding")
(println "           the boundary only as ρ→1; ‘flat near 1’ lives in ρ_spec, not here.")

;; --- (4) seed robustness of the spectral margin --------------------------
;; Each seed is settled to a residual-verified fixed point; seeds that never
;; converge (bounded non-settling orbits) are reported separately — assigning
;; them a "ρ_spec at h*" would be meaningless. (An earlier fixed-iteration
;; settle mislabeled exactly those seeds as ρ_spec>1.)
(println "\n=== seed robustness: ρ_spec across 10 random-weight seeds (matrix-free) ===")
(def seedsw (an/spectral-seed-sweep :seeds (range 1 11)))
(println (format "  %-5s %-16s %-9s %-9s %s" "ρ" "ρ_spec mean±std" "margin" "ρ_spec>1" "non-convergent seeds"))
(doseq [[rho {:keys [rho-spec-mean rho-spec-std margin-mean n-above-1
                     n-with-fixed-point non-convergent slow]}] seedsw]
  (println (format "  %-5.2f %-16s %-9s %d/%d      %s%s"
                   rho (format "%.3f±%.3f" rho-spec-mean rho-spec-std) margin-mean
                   n-above-1 n-with-fixed-point
                   (pr-str (mapv :seed non-convergent))
                   (if (seq slow) (format "  (slow: %s)" (pr-str (mapv :seed slow))) ""))))
(println "  reading: every seed that reaches a fixed point is contractive there")
(println "           (ρ_spec<1), but the margin collapses low→high carry and a")
(println "           minority of seeds (1–2/10 at every carry tested) never reach")
(println "           a fixed point at all: the diagonal constraint ρ(Ā)<1 does NOT")
(println "           certify convergence of the full recurrence. (Dense cross-check")
(println "           + ‖J_G‖₂ per seed at ρ=0.3,0.9: scripts/seed_verify.clj.)")

;; --- persist -------------------------------------------------------------
(spit "scripts/spectral.edn"
      (with-out-str
        (pp/pprint {:spectral-analysis spec
                    :injection-ratio inj
                    :cf-analysis cf
                    :seed-robustness seedsw})))
(println "\n--- spectral done (wrote scripts/spectral.edn) ---")
(System/exit 0)
