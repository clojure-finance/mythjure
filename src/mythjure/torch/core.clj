(ns mythjure.torch.core
  "libpython-clj lifecycle + PyTorch module handles for the torch façade.

  Requires the :torch alias (libpython-clj on the classpath) and a Python
  with PyTorch installed. Nothing here runs until `initialize!` is called,
  so merely having this namespace on the base classpath costs nothing.

  Python selection: pass :python-executable / :library-path, or set the
  MYTHJURE_PYTHON and MYTHJURE_LIBPYTHON env vars, or let libpython-clj
  auto-detect from PATH. With pyenv BOTH matter: point :python-executable at
  the real interpreter (…/.pyenv/versions/X/bin/python3, not the shim) AND
  :library-path at that version's libpythonX.Y.so — otherwise libpython-clj
  can silently load the SYSTEM libpython, giving you the wrong interpreter
  and the wrong site-packages (symptom: ModuleNotFoundError for a package
  that `python3 -c \"import …\"` clearly has). The pyenv Python must be built
  with --enable-shared.

  All other mythjure.torch.* namespaces call PyTorch through the module
  handles and helpers defined here. Model code must never require
  libpython-clj directly — new ops go into the façade."
  (:require [libpython-clj2.python :as py]))

(defonce ^:private initialized* (atom false))

;; Module handles, bound by initialize!. Vars (not delays) so the REPL can
;; inspect them and other façade namespaces can refer to them directly.
(defonce torch nil)
(defonce F nil)
(defonce optim nil)
(defonce nn-utils nil)

(def ^:dynamic *device*
  "Default device for tensor creation. Rebind (or alter-var-root) to \"cuda\"
  when a GPU is available; every façade constructor threads this through."
  "cpu")

(defn initialize!
  "Embed CPython and import the torch modules. Idempotent; returns :ok.
  Options: :python-executable — path to the Python binary to embed."
  [& {:keys [python-executable library-path]}]
  (when (compare-and-set! initialized* false true)
    (let [exe (or python-executable (System/getenv "MYTHJURE_PYTHON"))
          lib (or library-path (System/getenv "MYTHJURE_LIBPYTHON"))]
      (apply py/initialize!
             (cond-> []
               exe (conj :python-executable exe)
               lib (conj :library-path lib))))
    (alter-var-root #'torch    (constantly (py/import-module "torch")))
    (alter-var-root #'F        (constantly (py/import-module "torch.nn.functional")))
    (alter-var-root #'optim    (constantly (py/import-module "torch.optim")))
    (alter-var-root #'nn-utils (constantly (py/import-module "torch.nn.utils"))))
  :ok)

(defn initialized? [] @initialized*)

;; ---------------------------------------------------------------------------
;; dtype / device resolution
;; ---------------------------------------------------------------------------

(defn dtype
  "Resolve a keyword to a torch dtype object. Doubles (:float64) are the
  validation dtype — they match the pure-Clojure oracle bit-for-bit closely;
  :float32 is the training dtype."
  [k]
  (py/get-attr torch (name k)))

;; ---------------------------------------------------------------------------
;; Interop helpers (used by the other façade namespaces)
;; ---------------------------------------------------------------------------

(defn- assert-pyobj
  "Guard against passing a raw JVM value where a Python object is required.
  Without this, a Long reaching the FFI is interpreted as a POINTER and the
  JVM segfaults inside CPython (learned the hard way: `.numel()` returns an
  auto-converted JVM Long; calling .item() on it crashed the process)."
  [obj ctx]
  (when (or (nil? obj) (number? obj) (string? obj) (boolean? obj) (coll? obj))
    (throw (ex-info (str ctx ": expected a Python object, got a plain JVM value "
                         "(libpython-clj auto-converts Python numbers/strings on return)")
                    {:value obj :type (type obj)}))))

(defn call
  "Call a method on a Python object: (call t \"reshape\" 2 3)."
  [obj method & args]
  (assert-pyobj obj (str "call ." method))
  (apply py/call-attr obj method args))

(defn call-kw
  "Call a method with positional args and keyword args:
  (call-kw F \"layer_norm\" [x [d]] {:weight g :bias b})."
  [obj method pos-args kw-args]
  (assert-pyobj obj (str "call-kw ." method))
  (py/call-attr-kw obj method pos-args kw-args))

(defn attr
  "Read a Python attribute: (attr t \"shape\")."
  [obj name]
  (assert-pyobj obj (str "attr ." name))
  (py/get-attr obj name))

(defn ->jvm
  "Convert a Python value to Clojure data."
  [x]
  (py/->jvm x))

(defn py-tuple
  "Convert a Clojure collection to a real Python tuple. Needed where torch's
  C++ layer type-checks for tuple-of-ints (e.g. layer_norm's normalized_shape,
  Tensor.reshape) and rejects the auto-bridged JVM list."
  [coll]
  (py/->py-tuple coll))

(defn tensor?
  "True if x is a live torch tensor (checked without a Python round-trip on
  obvious non-objects)."
  [x]
  (boolean
   (and (some? x)
        (not (coll? x))
        (not (number? x))
        (not (string? x))
        (try (py/->jvm (call torch "is_tensor" x))
             (catch Exception _ false)))))

(defn py-dict
  "Convert a Clojure map to a real Python dict (e.g. for load_state_dict —
  the auto-bridged JVM map is rejected by torch's type checks)."
  [m]
  (py/->py-dict m))

(defn py-keys
  "Keys of a Python mapping (dict / OrderedDict) as a vector of strings.
  The keys() view object does NOT auto-convert across the bridge (it comes
  back as an opaque pointer) — it must be materialized with builtins list()
  first."
  [obj]
  (assert-pyobj obj "py-keys")
  (mapv str (py/->jvm (py/call-attr (py/import-module "builtins") "list"
                                    (py/call-attr obj "keys")))))

(defn raw-call
  "Sanctioned escape hatch for anything the façade doesn't cover yet.
  Prefer adding a named façade fn once a call site stabilizes."
  [obj method & args]
  (apply py/call-attr obj method args))
