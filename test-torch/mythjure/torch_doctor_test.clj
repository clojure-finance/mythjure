(ns mythjure.torch-doctor-test
  "Packaging/doctor suite (direction doc §5.1 item 3): Python discovery,
  initialize! pre-flight errors, the doctor report, and the readable
  python-error wrapping in core/call.

  Runs under :test-torch. The env vars are optional here like everywhere
  else — env-var-specific assertions guard on their presence; one test
  spawns a fresh JVM with MYTHJURE_LIBPYTHON unset and proves the lib is
  derived from sysconfig."
  (:require [clojure.java.shell :as shell]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing use-fixtures]]
            [mythjure.torch.core :as core]
            [mythjure.torch.doctor :as doctor]
            [mythjure.torch.tensor :as t]))

(use-fixtures :once (fn [f] (core/initialize!) (f)))

(def ^:private env-python (System/getenv "MYTHJURE_PYTHON"))

;; whatever initialize! would embed — valid with or without env vars
(def ^:private target-python (:python-executable (core/resolve-python)))

;; ---------------------------------------------------------------------------
;; python-info (subprocess interrogation)
;; ---------------------------------------------------------------------------

(deftest python-info-interrogates-the-interpreter
  (let [info (core/python-info target-python)]
    (is (nil? (:error info)))
    (is (re-matches #"\d+\.\d+\.\d+" (:version info)))
    (is (:enable-shared? info))
    (is (some? (:torch-version info)))
    (testing "the discovered libpython is a real file in the interpreter's LIBDIR"
      (is (some? (:libpython info)))
      (is (.isFile (java.io.File. ^String (:libpython info))))
      (is (str/starts-with? (:libpython info) (:libdir info))))))

(deftest python-info-on-a-missing-executable-reports-not-crashes
  (let [info (core/python-info "/nonexistent/python3")]
    (is (some? (:error info)))))

;; ---------------------------------------------------------------------------
;; resolve-python (precedence + shim resolution)
;; ---------------------------------------------------------------------------

(deftest resolve-python-option-beats-env-and-path
  (let [r (core/resolve-python :python-executable target-python)]
    (is (= :option (:exe-source r)))
    (is (= target-python (:python-executable r)))))

(deftest resolve-python-falls-back-to-env
  ;; only meaningful when the env var is actually set
  (when env-python
    (let [r (core/resolve-python)]
      (is (= :env (:exe-source r)))
      (is (= env-python (:python-executable r))))))

(deftest resolve-python-finds-something-without-options
  ;; on any machine that can run this suite, bare resolution must succeed
  (let [r (core/resolve-python)]
    (is (some? (:python-executable r)))
    (is (contains? #{:env :path} (:exe-source r)))
    (is (some? (:library-path r)))))

(deftest resolve-python-de-shims-sys-executable
  ;; Pointing at the pyenv SHIM must resolve to the real interpreter — the
  ;; embedded exe and its libpython then come from the same installation.
  (let [shim (str (System/getProperty "user.home") "/.pyenv/shims/python3")]
    (when (.isFile (java.io.File. shim))
      (let [r (core/resolve-python :python-executable shim)]
        (is (not= shim (:python-executable r)))
        (is (= shim (:requested-executable r)))
        (is (= (get-in r [:info :executable]) (:python-executable r)))))))

;; ---------------------------------------------------------------------------
;; env vars are a fallback: fresh JVM, MYTHJURE_LIBPYTHON unset, lib discovered
;; ---------------------------------------------------------------------------

(deftest ^:slow libpython-is-discovered-without-the-env-var
  (let [{:keys [exit out err]}
        (shell/sh "env" "-u" "MYTHJURE_LIBPYTHON"
                  "clojure" "-M:torch" "-e"
                  (pr-str '(do (require 'mythjure.torch.core 'mythjure.torch.tensor)
                               (let [r ((resolve 'mythjure.torch.core/resolve-python))]
                                 ((resolve 'mythjure.torch.core/initialize!))
                                 (prn [:lib-source (:lib-source r)
                                       :sum ((resolve 'mythjure.torch.tensor/item)
                                             ((resolve 'mythjure.torch.tensor/sum)
                                              ((resolve 'mythjure.torch.tensor/from-clj) [1.0 2.0 3.0])))]))))
                  :dir (System/getProperty "user.dir"))]
    (is (zero? exit) (str "fresh-JVM run failed: " err))
    (is (str/includes? out ":lib-source :discovered") out)
    (is (str/includes? out ":sum 6.0") out)))

;; ---------------------------------------------------------------------------
;; doctor
;; ---------------------------------------------------------------------------

(deftest checkup-passes-on-the-working-config
  (let [{:keys [ok? checks]} (doctor/checkup)]
    (is ok?)
    (is (= #{"python executable" "interpreter" "shared build" "libpython" "torch" "embedded"}
           (set (map :name checks))))
    (testing "embedded cross-check sees the live interpreter"
      (let [emb (first (filter #(= "embedded" (:name %)) checks))]
        (is (= :ok (:status emb)))
        (is (str/includes? (:msg emb) "torch"))))))

(deftest checkup-fails-readably-on-a-missing-python
  (let [{:keys [ok? checks]} (doctor/checkup :python-executable "/nonexistent/python3")]
    (is (not ok?))
    (is (some #(and (= :fail (:status %)) (= "interpreter" (:name %))) checks))))

(deftest checkup-catches-the-pyenv-trap
  ;; exe from one installation + libpython from another = the 2026-07-15 trap.
  ;; The subprocess probe alone can't see it (torch imports fine there); the
  ;; consistency check must FAIL it. Only meaningful when a libpython from a
  ;; genuinely DIFFERENT installation than the target interpreter exists.
  (let [own     (get-in (core/resolve-python) [:info :libpython])
        foreign (first (filter #(.isFile (java.io.File. ^String %))
                               ["/usr/lib/x86_64-linux-gnu/libpython3.12.so"
                                "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0"]))]
    (when (and own foreign
               (not= (.getCanonicalPath (java.io.File. ^String foreign))
                     (.getCanonicalPath (java.io.File. ^String own))))
      (let [{:keys [ok? checks]} (doctor/checkup :library-path foreign)]
        (is (not ok?))
        (is (some #(and (= :fail (:status %)) (= "libpython" (:name %))) checks))))))

(deftest doctor-prints-and-returns-the-checkup
  (let [ret (atom nil)
        out (with-out-str (reset! ret (doctor/doctor)))]
    (is (str/includes? out "mythjure.torch doctor"))
    (is (str/includes? out "✓"))
    (is (= (:ok? @ret) true))))

;; ---------------------------------------------------------------------------
;; readable python errors from core/call
;; ---------------------------------------------------------------------------

(deftest shape-mismatch-errors-are-readable
  (let [x (t/from-clj [[1.0 2.0] [3.0 4.0]])]
    (let [e (is (thrown? clojure.lang.ExceptionInfo (core/call x "reshape" 3 5)))]
      (testing "one-line summary: python error first, no traceback dump"
        (is (str/starts-with? (ex-message e) "call .reshape failed: RuntimeError:")))
      (testing "tensor shapes attached (the missing fact in shape mismatches)"
        (is (str/includes? (ex-message e) "[:self [2 2]]"))
        (is (= [[:self [2 2]]] (:tensor-shapes (ex-data e)))))
      (testing "original exception preserved as cause"
        (is (some? (ex-cause e)))))))

(deftest arg-tensor-shapes-are-attached
  (let [y (t/from-clj [[1.0 2.0 3.0] [4.0 5.0 6.0]])]
    (let [e (is (thrown? clojure.lang.ExceptionInfo (t/matmul y y)))]
      (is (str/includes? (ex-message e) "mat1 and mat2 shapes cannot be multiplied"))
      (is (= [[0 [2 3]] [1 [2 3]]] (:tensor-shapes (ex-data e)))))))

(deftest ffi-pointer-guard-still-throws-first
  ;; the assert-pyobj guard (JVM scalar → FFI pointer → segfault class) must
  ;; keep firing BEFORE any python call is attempted
  (let [e (is (thrown? clojure.lang.ExceptionInfo (core/call 42 "item")))]
    (is (str/includes? (ex-message e) "expected a Python object"))))
