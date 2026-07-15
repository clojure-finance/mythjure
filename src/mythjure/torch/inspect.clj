(ns mythjure.torch.inspect
  "REPL/clojure-mcp observability for torch tensors. Everything returns plain
  Clojure data so the interactive workflow keeps working when the leaves of
  the param map are opaque Python objects. Safe on any tensor size — only
  scalars and small heads ever cross the bridge."
  (:require [mythjure.torch.core :as core]
            [mythjure.torch.tensor :as t]))

(defn tensor-summary
  "Shape, dtype, device, and basic statistics as a Clojure map. Stats are
  computed at float64 — .double(), not .float(), so f64 tensors don't lose
  precision and int/half tensors still get mean/std."
  [x]
  (let [xf (core/call x "double")]
    {:shape   (t/shape x)
     :dtype   (str (core/attr x "dtype"))
     :device  (str (core/attr x "device"))
     :min     (t/item (t/tmin xf))
     :max     (t/item (t/tmax xf))
     :mean    (t/item (core/call core/torch "mean" xf))
     :std     (t/item (core/call core/torch "std" xf))
     :norm    (t/norm xf)
     :has-nan? (core/->jvm
                (t/item (core/call core/torch "any"
                                   (core/call core/torch "isnan" x))))}))

(defn tensor-head
  "First n elements (flattened) as a Clojure vector."
  [x & {:keys [n] :or {n 10}}]
  (let [flat (core/call x "flatten")
        k    (min n (long (t/item (core/call flat "numel"))))]
    (t/to-clj (core/call core/torch "narrow" flat 0 0 k))))

(defn param-report
  "Walk a nested param map; per tensor leaf report path, shape, and norm."
  ([params] (param-report params []))
  ([params path]
   (cond
     (map? params)
     (vec (mapcat (fn [[k v]] (param-report v (conj path k))) params))

     (core/tensor? params)
     [{:path path
       :shape (t/shape params)
       :norm  (t/norm (core/call params "double"))}]

     :else nil)))
