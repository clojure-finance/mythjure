(ns mythjure.train-torch
  "Torch backend for mythjure.train: the same manual Adam (identical update
  rule, elementwise on whole tensors), the same minibatch loop, and the same
  three carry diagnostics — so a torch training trajectory is directly
  comparable to the oracle's, step for step, given the same seed.

  Functional like the oracle: adam-update returns [params' state'] with fresh
  tensors (no in-place mutation), preserving REPL-friendly time travel of
  param snapshots. Data sampling reuses mythjure.data unchanged."
  (:require [mythjure.data :as data]
            [mythjure.torch.tensor :as t]
            [mythjure.torch.nn :as nn]
            [mythjure.backprop-torch :as bpt]
            [mythjure.model-torch :as mt]))

;; ---------------------------------------------------------------------------
;; Gradient-tree helpers (leaves are tensors; structure = grads)
;; ---------------------------------------------------------------------------

(defn tzeros [g]
  (if (map? g) (into {} (map (fn [[k v]] [k (tzeros v)]) g)) (t/zeros-like g)))

(defn tscale [s g]
  (if (map? g) (into {} (map (fn [[k v]] [k (tscale s v)]) g)) (t/mul g s)))

(def tadd bpt/gadd)

;; ---------------------------------------------------------------------------
;; Adam (oracle's exact update rule, whole-tensor)
;; ---------------------------------------------------------------------------

(defn- adam-leaf [p g m v lr b1 b2 eps bc1 bc2]
  (let [m' (t/add (t/mul m b1) (t/mul g (- 1.0 b1)))
        v' (t/add (t/mul v b2) (t/mul (t/mul g g) (- 1.0 b2)))
        p' (t/sub p (t/div (t/mul (t/div m' bc1) lr)
                           (t/add (t/sqrt (t/div v' bc2)) eps)))]
    {:p p' :m m' :v v'}))

(defn adam-update
  "One Adam step. state = {:m :v :t}. Recurses over the GRAD tree, so
  non-trainable param keys (:n-heads, :window, …) pass through untouched.
  Returns [params' state']."
  [params grads state {:keys [lr b1 b2 eps] :or {lr 1e-3 b1 0.9 b2 0.999 eps 1e-8}}]
  (let [tt (inc (:t state))
        bc1 (- 1.0 (Math/pow b1 tt))
        bc2 (- 1.0 (Math/pow b2 tt))
        step (fn step [p g m v]
               (if (map? g)
                 (let [rs (map (fn [k] [k (step (get p k) (get g k) (get m k) (get v k))]) (keys g))]
                   {:p (reduce (fn [acc [k r]] (assoc acc k (:p r))) p rs)
                    :m (into {} (map (fn [[k r]] [k (:m r)]) rs))
                    :v (into {} (map (fn [[k r]] [k (:v r)]) rs))})
                 (adam-leaf p g m v lr b1 b2 eps bc1 bc2)))
        r (step params grads (:m state) (:v state))]
    [(:p r) {:m (:m r) :v (:v r) :t tt}]))

(defn init-adam [grads] {:m (tzeros grads) :v (tzeros grads) :t 0})

;; ---------------------------------------------------------------------------
;; Batch gradient (sum per-example grads, average) + loss
;; ---------------------------------------------------------------------------

(defn batch-grad [m batch]
  (let [per (mapv (fn [{:keys [input target weights]}]
                    (let [{:keys [loss cache]} (if weights
                                                 (mt/forward m input target weights)
                                                 (mt/forward m input target))
                          bw (mt/backward m cache)]
                      {:loss (t/item loss) :grads (:grads bw) :dA-bar (:dA-bar bw)}))
                  batch)
        B (count batch)]
    {:loss (/ (reduce + (map :loss per)) (double B))
     :grads (tscale (/ 1.0 B) (reduce tadd (map :grads per)))
     :dA-bar (t/mul (reduce t/add (map :dA-bar per)) (/ 1.0 B))}))

;; ---------------------------------------------------------------------------
;; Diagnostics (same three quantities, same output shape as the oracle)
;; ---------------------------------------------------------------------------

(defn effective-abar
  "Clojure vector of the ā actually in use, respecting :carry-mode/:faithful?."
  [{:keys [config params]}]
  (let [{:keys [carry-mode faithful?]} config
        {:keys [log-A log-dt]} params]
    (t/to-clj
     (cond (= carry-mode :free) log-A
           faithful? (mt/a-bar-parcae log-A (nn/softplus log-dt))
           :else (mt/a-bar log-A)))))

(defn channel-convergence
  "Per-channel ‖h_T−h_{T-1}‖ / ‖h_1−h_0‖ from a torch forward cache."
  [cache]
  (let [rf (:recur cache)
        states (if (:states rf) (:states rf)
                   (conj (mapv :H (:caches rf)) (:HT rf)))
        T (dec (count states))
        cnorm (fn [A B] (t/sqrt (t/sum (let [d (t/sub A B)] (t/mul d d)) 0)))
        first-step (t/to-clj (cnorm (states 1) (states 0)))
        last-step  (t/to-clj (cnorm (states T) (states (dec T))))]
    (mapv (fn [l f] (/ l (max 1e-12 f))) last-step first-step)))

(defn- pct [xs q] (nth (vec (sort xs)) (int (* q (dec (count xs))))))

(defn diagnostics
  [m sample-cache dA-bar]
  (let [ab (effective-abar m)
        da (t/to-clj dA-bar)
        conv (channel-convergence sample-cache)]
    {:a-bar  {:min (apply min ab) :max (apply max ab)
              :mean (/ (reduce + ab) (count ab)) :p90 (pct ab 0.9)}
     :dL/dā  {:mean (/ (reduce + da) (count da))
              :n-pos (count (filter pos? da)) :n-neg (count (filter neg? da))}
     :converge {:min (apply min conv) :max (apply max conv)
                :mean (/ (reduce + conv) (count conv)) :p90 (pct conv 0.9)}
     :worst-channel
     (let [c (apply max-key ab (range (count ab)))]
       {:channel c :a-bar (nth ab c) :converge (nth conv c)})}))

;; ---------------------------------------------------------------------------
;; Training loop (structure identical to mythjure.train/train)
;; ---------------------------------------------------------------------------

(defn train
  [m encoded {:keys [steps batch-size lr seed log-every verbose?]
              :or {steps 200 batch-size 8 lr 1e-3 seed 0 log-every 20 verbose? true}}]
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
                     (let [{:keys [cache]} (mt/forward mm (:input (first batch)) (:target (first batch)))
                           diag (diagnostics mm cache dA-bar)]
                       (when verbose?
                         (println (format "step %4d  loss %.4f  ā[min %.3f mean %.3f max %.3f]  dL/dā[+%d/-%d]  conv[mean %.2f max %.2f]"
                                          step loss (get-in diag [:a-bar :min]) (get-in diag [:a-bar :mean]) (get-in diag [:a-bar :max])
                                          (get-in diag [:dL/dā :n-pos]) (get-in diag [:dL/dā :n-neg])
                                          (get-in diag [:converge :mean]) (get-in diag [:converge :max]))))
                       (conj hist (assoc diag :step step :loss loss)))
                     hist)]
          (recur (inc step) mm' opt' hist))))))
