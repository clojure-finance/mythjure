(ns mythjure.data
  "Character-level corpus handling for Tiny Shakespeare: load, build vocab,
  encode, and sample training windows. A training example is a contiguous window
  of `seq-len+1` characters; the model predicts each next character, so inputs =
  window[0..seq-len), targets = window[1..seq-len]."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]))

(defn load-text
  ([] (load-text "resources/tinyshakespeare.txt"))
  ([path] (slurp (io/file path))))

(defn build-vocab
  "Sorted distinct characters → {:chars [..] :stoi {ch idx} :itos {idx ch} :size n}."
  [text]
  (let [chars (vec (sort (distinct text)))]
    {:chars chars
     :stoi (into {} (map-indexed (fn [i c] [c i]) chars))
     :itos (into {} (map-indexed (fn [i c] [i c]) chars))
     :size (count chars)}))

(defn encode [vocab s] (mapv (:stoi vocab) s))
(defn decode [vocab ids] (str/join (map (:itos vocab) ids)))

(defn sample-window
  "Random window of length seq-len+1 from the encoded corpus, returned as
  {:input [seq-len ids] :target [seq-len ids]} (target = input shifted by 1).
  `rng` is a java.util.Random for reproducibility."
  [encoded seq-len ^java.util.Random rng]
  (let [start (.nextInt rng (- (count encoded) seq-len 1))
        win (subvec encoded start (+ start seq-len 1))]
    {:input (subvec win 0 seq-len)
     :target (subvec win 1 (inc seq-len))}))

(defn sample-batch
  [encoded seq-len batch-size rng]
  (vec (repeatedly batch-size #(sample-window encoded seq-len rng))))
