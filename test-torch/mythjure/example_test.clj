(ns mythjure.example-test
  "Pins the §4 gate artifacts: the non-mythjure examples must run end to end
  through the general-purpose facade surfaces alone (two_moons, mnist), and
  the looped-encoder template (looped_encoder — the downstream-consumer
  pattern over mythjure.torch.looped) must keep working as published. Each
  example's own asserts (accuracy floors; MNIST and looped_encoder also the
  exact checkpoint round-trip) fire on regression; these tests fail on any
  throw. The MNIST test is guarded on the cached download so the suite stays
  network-free."
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is]]))

(deftest mnist-example-runs
  ;; Guarded on the cached download so the suite never needs the network:
  ;; run `clojure -M:torch examples/mnist.clj` once to enable this test.
  (if (.exists (io/file "data/mnist/train-images-idx3-ubyte"))
    (let [scratch (create-ns 'mythjure.mnist-example-scratch)]
      (binding [*ns* scratch]
        (clojure.core/refer 'clojure.core)
        (load-file "examples/mnist.clj"))
      (is true "examples/mnist.clj completed (its own accuracy + checkpoint asserts passed)"))
    (println "  [skip] mnist-example-runs: data/mnist not cached — run: clojure -M:torch examples/mnist.clj")))

(deftest two-moons-example-runs
  (let [scratch (create-ns 'mythjure.example-test-scratch)]
    (binding [*ns* scratch]
      (clojure.core/refer 'clojure.core)
      (load-file "examples/two_moons.clj"))
    (is true "examples/two_moons.clj completed (its own accuracy asserts passed)")))

(deftest looped-encoder-example-runs
  (let [scratch (create-ns 'mythjure.looped-encoder-example-scratch)]
    (binding [*ns* scratch]
      (clojure.core/refer 'clojure.core)
      (load-file "examples/looped_encoder.clj"))
    (is true "examples/looped_encoder.clj completed (its own accuracy + checkpoint asserts passed)")))
