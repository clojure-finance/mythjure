(ns mythjure.example-test
  "Pins the §4 gate artifact: the non-mythjure toy example must run end to end
  through the general-purpose facade surfaces alone. The example's own asserts
  (>=0.9 test accuracy for BOTH model idioms) fire on regression; this test
  fails on any throw."
  (:require [clojure.test :refer [deftest is]]))

(deftest two-moons-example-runs
  (let [scratch (create-ns 'mythjure.example-test-scratch)]
    (binding [*ns* scratch]
      (clojure.core/refer 'clojure.core)
      (load-file "examples/two_moons.clj"))
    (is true "examples/two_moons.clj completed (its own accuracy asserts passed)")))
