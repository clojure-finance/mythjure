;; Torch-backend rerun of scripts/train_run.clj (§7.2): trains the LM on Tiny
;; Shakespeare with the SAME init, seed, and batch sequence via
;; mythjure.train-torch (whose Adam trajectory is bit-comparable to the
;; oracle's), then compares the history against the committed
;; scripts/train_history.edn.
;; Run:  MYTHJURE_PYTHON=… MYTHJURE_LIBPYTHON=… clojure -M:torch scripts/train_run_torch.clj
(require '[mythjure.data :as data]
         '[mythjure.model :as model]
         '[mythjure.torch.core :as tc]
         '[mythjure.torch.tensor :as t]
         '[mythjure.torch.nn :as nn]
         '[mythjure.model-torch :as mt]
         '[mythjure.train-torch :as tt]
         '[clojure.edn :as edn]
         '[clojure.pprint :as pp])

(tc/initialize!)

(defn generate
  "Temperature-0.8 sampling via the torch forward (same rng protocol as
  scripts/train_run.clj)."
  [m vocab seed-text n-chars rng]
  (let [seq-len (get-in m [:config :seq-len])
        seed-ids (data/encode vocab seed-text)]
    (loop [ctx (vec (take-last seq-len seed-ids)) out []]
      (if (= (count out) n-chars)
        (data/decode vocab (concat seed-ids out))
        (let [pad (max 0 (- seq-len (count ctx)))
              padded (into (vec (repeat pad 0)) ctx)
              dummy (vec (repeat seq-len 0))
              {:keys [cache]} (mt/forward m padded dummy)
              probs (last (t/to-clj (nn/softmax (:logits cache))))
              temp 0.8
              logp (mapv #(/ (Math/log (max 1e-12 %)) temp) probs)
              mx (apply max logp)
              ex (mapv #(Math/exp (- % mx)) logp)
              s (reduce + ex)
              p (mapv #(/ % s) ex)
              r (.nextDouble rng)
              nxt (loop [i 0 acc 0.0]
                    (let [acc' (+ acc (nth p i))]
                      (if (or (>= acc' r) (= i (dec (count p)))) i (recur (inc i) acc'))))]
          (recur (vec (take-last seq-len (conj ctx nxt))) (conj out nxt)))))))

(println "loading corpus...")
(def text (data/load-text))
(def vocab (data/build-vocab text))
(def encoded (data/encode vocab text))

(def m0 (update (model/init {:vocab-size (:size vocab) :d-model 32 :n-heads 4
                             :d-ff 128 :seq-len 32 :T 4 :seed 0 :log-A0 0.5})
                :params mt/params->torch))
(println "--- training (torch backend, float64; identical schedule to train_run.clj) ---")
(def t0 (System/nanoTime))
(def result (tt/train m0 encoded {:steps 150 :batch-size 6 :lr 3e-3
                                  :seed 0 :log-every 10}))
(println (format "wall: %.1f s" (/ (- (System/nanoTime) t0) 1e9)))

(println "--- sample (temp 0.8) ---")
(println (generate (:model result) vocab "ROMEO:" 240 (java.util.Random. 42)))

(def hist (mapv #(select-keys % [:step :loss :a-bar :dL/dā :converge :worst-channel])
                (:history result)))
(spit "scripts/train_history_torch.edn" (with-out-str (pp/pprint hist)))
(println "--- saved scripts/train_history_torch.edn ---")

(defn deep-close? [a b]
  (cond
    (and (number? a) (number? b))
    (or (== a b)
        (< (Math/abs (- (double a) (double b)))
           (* 1e-6 (max 1.0 (Math/abs (double a))))))
    (and (map? a) (map? b))
    (and (= (set (keys a)) (set (keys b)))
         (every? (fn [[k v]] (deep-close? v (get b k))) a))
    (and (sequential? a) (sequential? b))
    (and (= (count a) (count b)) (every? true? (map deep-close? a b)))
    :else (= a b)))

(def committed (edn/read-string (slurp "scripts/train_history.edn")))
(def ok? (deep-close? committed hist))
(println (if ok?
           "MATCH — torch training reproduces the committed train_history.edn."
           "*** MISMATCH vs committed train_history.edn ***"))
(System/exit (if ok? 0 1))
