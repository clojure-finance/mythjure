(ns mythjure.torch.optim
  "Optimizer plumbing over torch tensors held in Clojure param maps.

  The param map convention: an arbitrarily nested Clojure map whose leaves are
  torch tensors (see the direction doc §1.4). Tied parameters (embedding =
  unembedding) are the SAME tensor object reachable via two paths — both
  `all-param-tensors` and `set-grads!` identity-deduplicate, otherwise Adam
  would step the tied tensor twice (silent 2× lr) or a naive set-grads! would
  overwrite one path's gradient with the other's."
  (:require [mythjure.torch.core :as core]))

(defn- tensor-leaves
  "Seq of [path tensor] for every tensor leaf in a nested param map."
  ([m] (tensor-leaves m []))
  ([m path]
   (cond
     (map? m) (mapcat (fn [[k v]] (tensor-leaves v (conj path k))) m)
     (core/tensor? m) [[path m]]
     :else nil)))

(defn all-param-tensors
  "All unique tensor leaves of a param map, identity-deduplicated (tied
  parameters appear once). Returns a Clojure vector of tensors."
  [params]
  (let [seen (java.util.IdentityHashMap.)]
    (into []
          (keep (fn [[_ t]]
                  (when-not (.containsKey seen t)
                    (.put seen t true)
                    t)))
          (tensor-leaves params))))

(defn- py-set-grad! [p g]
  (core/call p "__setattr__" "grad" (core/call g "clone")))

(defn set-grads!
  "Copy manually computed gradients into each param tensor's .grad.
  `grads` mirrors the structure of `params` (same paths). On the SECOND
  encounter of a tied tensor the gradient is ADDED, not overwritten."
  [params grads]
  (let [seen (java.util.IdentityHashMap.)]
    (doseq [[path g] (tensor-leaves grads)]
      (let [p (get-in params path)]
        (when-not (core/tensor? p)
          (throw (ex-info "grad path has no tensor param" {:path path})))
        (if (.containsKey seen p)
          (core/call (core/attr p "grad") "add_" g)
          (do (py-set-grad! p g)
              (.put seen p true))))))
  nil)

(defn adam
  "PyTorch Adam over the unique tensors of a param map."
  [params & {:keys [lr] :or {lr 3e-4}}]
  (core/call-kw core/optim "Adam" [(all-param-tensors params)] {:lr lr}))

(defn step! [optimizer] (core/call optimizer "step") nil)
(defn zero-grad! [optimizer] (core/call optimizer "zero_grad") nil)

(defn clip-grad-norm!
  "Clip gradients of a param map in place; returns the pre-clip total norm."
  [params max-norm]
  (-> (core/call core/nn-utils "clip_grad_norm_" (all-param-tensors params) max-norm)
      (core/call "item")
      core/->jvm))
