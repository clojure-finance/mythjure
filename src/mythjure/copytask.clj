(ns mythjure.copytask
  "Synthetic copy-across-a-gap task — a single dial (gap k) with a quantitative
  prediction for the carry. A sequence is

      [A] [PAD × k] [QUERY] [A] [PAD ...]

  and the loss counts ONLY at the QUERY position, where the model must predict A
  (the token that appeared k+1 positions earlier). The next-token target at the
  query position is the [A] placed right after it; causal masking hides that copy
  from the query, so the only A the query can use is the one at position 0.

  Prediction (for a model whose memory mechanism is the loop carry): bridging the
  gap needs a channel that retains state with time-constant τ ≈ k, i.e. ā ≈ 1−1/k.
  Sweeping k therefore predicts a specific learned max(ā) per k. (Caveat worth
  testing empirically: in a transformer block, ATTENTION bridges positions in one
  hop regardless of k, so full attention may solve this with ā≈0 — in which case
  the carry is never pressured and ā stays put. That null is itself the result.)"
  (:require [mythjure.linalg :as la]))

(def PAD 0)
(def QUERY 1)

(defn example
  "One copy example for gap k. n-content content tokens live at ids 2..2+n-content."
  [k seq-len n-content ^java.util.Random rng]
  (let [a (+ 2 (.nextInt rng n-content))
        toks (vec (concat [a] (repeat k PAD) [QUERY a] (repeat (- seq-len k 3) PAD)))
        input (subvec toks 0 seq-len)
        target (conj (subvec input 1 seq-len) PAD)   ; next-token; only qpos matters
        qpos (inc k)]
    {:input input :target target
     :weights (assoc (vec (repeat seq-len 0.0)) qpos 1.0)
     :answer a :qpos qpos}))

(defn batch [k seq-len n-content batch-size rng]
  (vec (repeatedly batch-size #(example k seq-len n-content rng))))
