(ns mythjure.looped
  "Step 3: drive the real transformer block (mythjure.block) RECURRENTLY — the
  looped-transformer forward pass at the sequence level — and reconnect it to the
  Parcae stability story from mythjure.dynamics.

  The recurrent hidden state is the whole [seq_len × d_model] activation matrix
  H. E is the (fixed) Prelude output, re-injected every iteration ('input
  injection'). Three drivers, increasingly faithful:

    naive-loop    H_{t+1} = Block(H_t)                         ; no injection
    injected-loop H_{t+1} = Block(H_t + E)                     ; input injection
    gated-loop    H_{t+1} = A_bar ⊙ H_t + increment(H_t + E)   ; + Parcae carry

  where increment(z) = Block(z) − z is the block's attn+mlp contribution.

  Key finding reproduced here: a looped pre-LN block grows its residual stream
  LINEARLY and without bound (each iteration adds a roughly constant chunk,
  because LayerNorm fixes the block's input scale). The Parcae carry A_bar ∈
  (0,1)^d — the very same negative-diagonal discretization from
  mythjure.dynamics — turns that runaway arithmetic series into a bounded
  geometric one. Carry near 1 ⇒ richer accumulation but a taller, slower
  saturation (the 1/(1−a) time constant), exactly as in the linear toy."
  (:require [mythjure.linalg :as la]
            [mythjure.block :as blk]
            [mythjure.dynamics :as dyn]))

(defn rms
  "Root-mean-square of every entry — the residual-stream magnitude."
  [M]
  (Math/sqrt (la/mean (map (fn [r] (la/mean (map #(* % %) r))) M))))

(defn increment
  "The block's per-iteration contribution: Block(z) − z (attn + mlp)."
  [params z]
  (la/mat- (blk/forward params z) z))

;; ---------------------------------------------------------------------------
;; Recurrent drivers — each returns the sequence of states [H0 H1 ... HT]
;; ---------------------------------------------------------------------------

(defn naive-loop [params H0 T]
  (->> (iterate #(blk/forward params %) H0) (take (inc T)) vec))

(defn injected-loop [params E H0 T]
  (->> (iterate #(blk/forward params (la/mat+ % E)) H0) (take (inc T)) vec))

(defn gated-loop
  "Parcae-gated recurrent loop. `a-bar` is a length-d_model vector in (0,1)
  (build it with mythjure.dynamics/a-bar-constrained). Carry is applied
  per-channel to every position of the state."
  [params E H0 a-bar T]
  (let [carry (fn [H] (mapv #(la/hadamard % a-bar) H))]
    (->> (iterate (fn [H] (la/mat+ (carry H) (increment params (la/mat+ H E))))
                  H0)
         (take (inc T))
         vec)))

(defn rms-trajectory [states] (mapv rms states))

;; ---------------------------------------------------------------------------
;; Demo
;; ---------------------------------------------------------------------------

(defn demo
  "Compare residual-stream growth: ungated injected loop (unbounded) vs.
  Parcae-gated loops at several carries (bounded). Returns RMS trajectories."
  [& {:keys [d-model n-heads seq-len T seed]
      :or {d-model 32 n-heads 4 seq-len 16 T 30 seed 7}}]
  (let [p (blk/init-params {:d-model d-model :n-heads n-heads :seed seed})
        E (la/rand-matrix 99 seq-len d-model 1.0)
        a-bar (fn [carry] (vec (repeat d-model carry)))
        traj (fn [states] (mapv #(Double/parseDouble (format "%.2f" %))
                                (rms-trajectory states)))]
    {:ungated   (traj (injected-loop p E E T))
     :carry-0.5 (traj (gated-loop p E E (a-bar 0.5) T))
     :carry-0.7 (traj (gated-loop p E E (a-bar 0.7) T))
     :carry-0.9 (traj (gated-loop p E E (a-bar 0.9) T))}))

(comment
  (require '[mythjure.dynamics :as dyn])
  (demo)
  ;; sparklines:
  (let [p (blk/init-params {:d-model 32 :n-heads 4 :seed 7})
        E (la/rand-matrix 99 16 32 1.0)]
    {:ungated (dyn/sparkline (rms-trajectory (injected-loop p E E 30)))
     :gated   (dyn/sparkline (rms-trajectory
                              (gated-loop p E E (dyn/a-bar-constrained (la/zeros 32) 1.0) 30)))}))
