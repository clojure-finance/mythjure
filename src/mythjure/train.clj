(ns mythjure.train
  "Adam optimizer, minibatch training loop, and the three carry diagnostics.

  Diagnostics (per the BPTT-gradient prediction): each step we can watch whether
  the optimizer is pushing channels toward long memory / thin margin, and whether
  the system self-corrects.
    1. ā distribution        — are any channels climbing toward 1?
    2. dL/dā sign            — the correlation Σ H_t⊙dH_t; which way is the carry pushed?
    3. per-channel convergence ratio  ‖h_T−h_{T-1}‖/‖h_1−h_0‖ — has the channel
       actually settled by the final loop? Near 1 + climbing ā ⇒ headed for the cliff."
  (:require [mythjure.linalg :as la]
            [mythjure.model :as model]
            [mythjure.backprop :as bp]
            [mythjure.data :as data]))

;; ---------------------------------------------------------------------------
;; Gradient-tree helpers (leaves are matrices or vectors; structure = grads)
;; ---------------------------------------------------------------------------

(defn tzeros [g]
  (cond (map? g) (into {} (map (fn [[k v]] [k (tzeros v)]) g))
        (sequential? (first g)) (mapv #(mapv (constantly 0.0) %) g)
        :else (mapv (constantly 0.0) g)))

(defn tscale [s g]
  (cond (map? g) (into {} (map (fn [[k v]] [k (tscale s v)]) g))
        (sequential? (first g)) (mapv #(mapv (fn [x] (* s x)) %) g)
        :else (mapv #(* s %) g)))

(defn tadd [a b] (bp/gadd a b))

;; ---------------------------------------------------------------------------
;; Adam
;; ---------------------------------------------------------------------------

(defn- adam-leaf [p g m v lr b1 b2 eps bc1 bc2 matrix?]
  (let [upd (fn [pi gi mi vi]
              (let [m' (+ (* b1 mi) (* (- 1 b1) gi))
                    v' (+ (* b2 vi) (* (- 1 b2) gi gi))
                    p' (- pi (/ (* lr (/ m' bc1)) (+ (Math/sqrt (/ v' bc2)) eps)))]
                [p' m' v']))]
    (if matrix?
      (let [r (mapv (fn [pr gr mr vr] (mapv upd pr gr mr vr)) p g m v)]
        {:p (mapv #(mapv (fn [t] (nth t 0)) %) r)
         :m (mapv #(mapv (fn [t] (nth t 1)) %) r)
         :v (mapv #(mapv (fn [t] (nth t 2)) %) r)})
      (let [r (mapv upd p g m v)]
        {:p (mapv #(nth % 0) r) :m (mapv #(nth % 1) r) :v (mapv #(nth % 2) r)}))))

(defn adam-update
  "One Adam step. state = {:m :v :t}. Recurses over the grad tree, preserving any
  non-trainable param keys (e.g. :n-heads) untouched. Returns [params' state']."
  [params grads state {:keys [lr b1 b2 eps] :or {lr 1e-3 b1 0.9 b2 0.999 eps 1e-8}}]
  (let [t (inc (:t state)) bc1 (- 1 (Math/pow b1 t)) bc2 (- 1 (Math/pow b2 t))
        step (fn step [p g m v]
               (cond
                 (map? g)
                 (let [rs (map (fn [k] [k (step (get p k) (get g k) (get m k) (get v k))]) (keys g))]
                   {:p (reduce (fn [acc [k r]] (assoc acc k (:p r))) p rs)
                    :m (into {} (map (fn [[k r]] [k (:m r)]) rs))
                    :v (into {} (map (fn [[k r]] [k (:v r)]) rs))})
                 (sequential? (first g)) (adam-leaf p g m v lr b1 b2 eps bc1 bc2 true)
                 :else (adam-leaf p g m v lr b1 b2 eps bc1 bc2 false)))
        r (step params grads (:m state) (:v state))]
    [(:p r) {:m (:m r) :v (:v r) :t t}]))

(defn init-adam [grads] {:m (tzeros grads) :v (tzeros grads) :t 0})

;; ---------------------------------------------------------------------------
;; Batch gradient (sum per-example grads, average) + loss
;; ---------------------------------------------------------------------------

(defn batch-grad [m batch]
  (let [per (mapv (fn [{:keys [input target weights]}]
                    (let [{:keys [loss cache]} (if weights
                                                 (model/forward m input target weights)
                                                 (model/forward m input target))
                          bw (model/backward m cache)]
                      {:loss loss :grads (:grads bw) :dA-bar (:dA-bar bw)}))
                  batch)
        B (count batch)
        gsum (reduce tadd (map :grads per))
        dabar (reduce (fn [a g] (mapv + a g)) (map :dA-bar per))]
    {:loss (/ (reduce + (map :loss per)) (double B))
     :grads (tscale (/ 1.0 B) gsum)
     :dA-bar (tscale (/ 1.0 B) dabar)}))

;; ---------------------------------------------------------------------------
;; Diagnostics
;; ---------------------------------------------------------------------------

(defn- pct [xs q] (nth (vec (sort xs)) (int (* q (dec (count xs))))))

(defn channel-convergence
  "Per-channel ‖h_T−h_{T-1}‖/‖h_1−h_0‖ from a forward cache. Reconstructs states
  H0..HT from the model forward `cache`. Ratio→0 settled, →1 still moving."
  [cache]
  (let [rf (:recur cache)
        states (if (:states rf) (:states rf)                    ; ablated recurrence
                   (conj (mapv :H (:caches rf)) (:HT rf)))       ; standard recurrence
        T (dec (count states))
        col (fn [M c] (mapv #(nth % c) M))
        cnorm (fn [A B c] (la/norm (la/v- (col A c) (col B c))))
        d (count (first (last states)))]
    (mapv (fn [c]
            (let [first-step (cnorm (states 1) (states 0) c)
                  last-step  (cnorm (states T) (states (dec T)) c)]
              (/ last-step (max 1e-12 first-step))))
          (range d))))

(defn diagnostics
  "Snapshot the three carry diagnostics. `dA-bar` is dL/dā (from batch-grad)."
  [m sample-cache dA-bar]
  (let [ab (model/effective-abar m)                 ; respects :carry-mode/:faithful?
        conv (channel-convergence sample-cache)]
    {:a-bar  {:min (apply min ab) :max (apply max ab)
              :mean (/ (reduce + ab) (count ab)) :p90 (pct ab 0.9)}
     :dL/dā  {:mean (/ (reduce + dA-bar) (count dA-bar))
              :n-pos (count (filter pos? dA-bar)) :n-neg (count (filter neg? dA-bar))}
     :converge {:min (apply min conv) :max (apply max conv)
                :mean (/ (reduce + conv) (count conv)) :p90 (pct conv 0.9)}
     :worst-channel ;; highest ā — the one closest to the cliff
     (let [c (apply max-key ab (range (count ab)))]
       {:channel c :a-bar (nth ab c) :converge (nth conv c)})}))

;; ---------------------------------------------------------------------------
;; Training loop
;; ---------------------------------------------------------------------------

(defn train
  [m encoded {:keys [steps batch-size lr seed log-every]
              :or {steps 200 batch-size 8 lr 1e-3 seed 0 log-every 20}}]
  (let [seq-len (get-in m [:config :seq-len])
        rng (java.util.Random. seed)]
    (loop [step 1, mm m, opt nil, hist []]
      (if (> step steps)
        {:model mm :history hist}
        (let [batch (data/sample-batch encoded seq-len batch-size rng)
              {:keys [loss grads dA-bar]} (batch-grad mm batch)
              opt (or opt (init-adam grads))
              [params' opt'] (adam-update (:params mm) grads opt {:lr lr})
              mm' (assoc mm :params params')
              hist (if (or (= step 1) (zero? (mod step log-every)))
                     (let [{:keys [cache]} (model/forward mm (:input (first batch)) (:target (first batch)))
                           diag (diagnostics mm cache dA-bar)]
                       (println (format "step %4d  loss %.4f  ā[min %.3f mean %.3f max %.3f]  dL/dā[+%d/-%d]  conv[mean %.2f max %.2f]"
                                        step loss (get-in diag [:a-bar :min]) (get-in diag [:a-bar :mean]) (get-in diag [:a-bar :max])
                                        (get-in diag [:dL/dā :n-pos]) (get-in diag [:dL/dā :n-neg])
                                        (get-in diag [:converge :mean]) (get-in diag [:converge :max])))
                       (conj hist (assoc diag :step step :loss loss)))
                     hist)]
          (recur (inc step) mm' opt' hist))))))
