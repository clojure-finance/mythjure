;; §7.6 two-path GENERALITY probe: construct a task whose optimal memory is
;; axis-aligned (per-channel, diagonal) — the carry's exact inductive bias — and
;; see whether gradient descent finally uses the carry or still routes through the
;; block. Task: per-channel leaky integral. Channel c has target decay a*_c; the
;; target readout is Y_c = u_c · (1−a*_c^T)/(1−a*_c) — LITERALLY what a carry with
;; ā_c=a*_c computes (with a pass-through block). seq=1 (no cross-position), so the
;; block's only role is cross-channel mixing, which is WASTEFUL for a per-channel
;; task. If learned ā_c tracks a*_c → carry used → boundary of "free constraint"
;; FOUND. If ā_c stays flat at 0.5 → block learned it anyway → generality HARDENS.
;; Run:  clojure -M scripts/diagonal_memory.clj
(require '[mythjure.backprop :as bp]
         '[mythjure.block :as blk]
         '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.linalg :as la]
         '[clojure.pprint :as pp])

(def D 16) (def T 8) (def STEPS 2500) (def BATCH 16) (def LR 5e-3)
(def a-star (mapv (fn [i] (+ 0.3 (* 0.6 (/ i (dec D))))) (range D)))     ; 0.30 .. 0.90
(def gain-star (mapv (fn [a] (/ (- 1.0 (Math/pow a T)) (- 1.0 a))) a-star))
(def logA0 (vec (repeat D (Math/log (- (Math/log 0.5))))))              ; ā₀=0.5

(defn example [rng]
  (let [u (mapv (fn [_] (.nextGaussian rng)) (range D))]
    {:E [u] :Y [(mapv * gain-star u)]}))

(defn reg-forward [bpar logA E Y]
  (let [ab (model/a-bar logA)
        H0 (mapv (fn [_] (la/zeros D)) E)
        rf (bp/recur-forward bpar E H0 ab nil T)
        HT (:HT rf)
        n (* (count HT) D)
        loss (/ (reduce + (mapv (fn [hr yr] (reduce + (map (fn [h y] (let [d (- h y)] (* d d))) hr yr))) HT Y)) n)]
    {:loss loss :ab ab :HT HT :caches (:caches rf) :E E :Y Y :n n}))

(defn reg-backward [bpar logA {:keys [ab HT Y caches E n]}]
  (let [dHT (mapv (fn [hr yr] (mapv (fn [h y] (/ (* 2.0 (- h y)) n)) hr yr)) HT Y)
        rb (bp/recur-backward bpar E ab nil dHT caches)]
    {:block (:grads rb) :log-A (model/dlogA<-dabar (:dA-bar rb) ab)}))

(defn corr [xs ys]
  (let [n (count xs) mx (/ (reduce + xs) n) my (/ (reduce + ys) n)
        cov (/ (reduce + (map (fn [x y] (* (- x mx) (- y my))) xs ys)) n)
        sx (Math/sqrt (/ (reduce + (map #(let [d (- % mx)] (* d d)) xs)) n))
        sy (Math/sqrt (/ (reduce + (map #(let [d (- % my)] (* d d)) ys)) n))]
    (/ cov (max 1e-12 (* sx sy)))))

(println (format "diagonal-memory probe: d=%d T=%d, per-channel a* ∈ [%.2f,%.2f], ā₀=0.5"
                 D T (first a-star) (last a-star)))
(def rng (java.util.Random. 0))
(def final
  (reduce (fn [{:keys [p opt]} step]
            (let [batch (repeatedly BATCH #(example rng))
                  grads (train/tscale (/ 1.0 BATCH)
                                      (reduce (fn [acc {:keys [E Y]}]
                                                (let [g (reg-backward (:block p) (:log-A p) (reg-forward (:block p) (:log-A p) E Y))]
                                                  (if acc (train/tadd acc g) g))) nil batch))
                  opt (or opt (train/init-adam grads))
                  [p' opt'] (train/adam-update p grads opt {:lr LR})]
              (when (zero? (mod step 250))
                (let [ab (model/a-bar (:log-A p'))
                      loss (:loss (reg-forward (:block p') (:log-A p') (:E (first batch)) (:Y (first batch))))]
                  (println (format "step %4d  loss %.4f  ā[min %.3f max %.3f]  corr(ā,a*)=%.3f"
                                   step loss (apply min ab) (apply max ab) (corr ab a-star)))))
              {:p p' :opt opt'}))
          {:p {:block (blk/init-params {:d-model D :n-heads 2 :d-ff 64 :seed 1}) :log-A logA0} :opt nil}
          (range 1 (inc STEPS))))

(def ab-final (model/a-bar (get-in final [:p :log-A])))
(println "\n===== RESULT: learned ā_c vs target a*_c (per channel) =====")
(println "chan  a*_c   ā_c    (ā₀ was 0.500)")
(doseq [c (range D)]
  (println (format "%3d  %.3f  %.3f" c (nth a-star c) (nth ab-final c))))
(println (format "\ncorr(ā, a*) = %.3f   ā range [%.3f, %.3f]" (corr ab-final a-star) (apply min ab-final) (apply max ab-final)))
(println (if (> (corr ab-final a-star) 0.5)
           ">>> ā TRACKS a* — carry used — BOUNDARY of the free-constraint FOUND"
           ">>> ā FLAT / uncorrelated — block learned it anyway — GENERALITY HARDENS"))
(spit "scripts/diagonal_memory.edn" (with-out-str (pp/pprint {:a-star a-star :abar ab-final :corr (corr ab-final a-star)})))
(System/exit 0)
