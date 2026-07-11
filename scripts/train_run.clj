;; Background training run: trains the looped LM on Tiny Shakespeare, prints the
;; three carry diagnostics, samples text, and saves history to an EDN file.
;; Run from project root:  clojure -M scripts/train_run.clj
(require '[mythjure.data :as data]
         '[mythjure.model :as model]
         '[mythjure.train :as train]
         '[mythjure.linalg :as la]
         '[clojure.pprint :as pp])

(defn generate [m vocab seed-text n-chars rng]
  (let [seq-len (get-in m [:config :seq-len])
        seed-ids (data/encode vocab seed-text)]
    (loop [ctx (vec (take-last seq-len seed-ids)) out []]
      (if (= (count out) n-chars)
        (data/decode vocab (concat seed-ids out))
        (let [pad (max 0 (- seq-len (count ctx)))
              padded (into (vec (repeat pad 0)) ctx)
              dummy (vec (repeat seq-len 0))
              {:keys [cache]} (model/forward m padded dummy)
              probs (last (:probs cache))
              ;; temperature 0.8 sampling
              t 0.8
              logp (mapv #(/ (Math/log (max 1e-12 %)) t) probs)
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
(println "vocab" (:size vocab) "chars" (count encoded))

(def m0 (model/init {:vocab-size (:size vocab) :d-model 32 :n-heads 4
                     :d-ff 128 :seq-len 32 :T 4 :seed 0 :log-A0 0.5}))
(println "model: d-model 32, heads 4, seq 32, T 4, ā0 ≈"
         (format "%.3f" (first (model/a-bar (get-in m0 [:params :log-A])))))
(println "--- training ---")

(def result (train/train m0 encoded {:steps 150 :batch-size 6 :lr 3e-3
                                     :seed 0 :log-every 10}))

(println "--- sample (temp 0.8) ---")
(println (generate (:model result) vocab "ROMEO:" 240 (java.util.Random. 42)))

(spit "scripts/train_history.edn"
      (with-out-str (pp/pprint (mapv #(select-keys % [:step :loss :a-bar :dL/dā :converge :worst-channel])
                                     (:history result)))))
(println "--- saved history to scripts/train_history.edn ---")
(System/exit 0)
