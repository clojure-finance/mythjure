(ns mythjure.torch.core
  "libpython-clj lifecycle + PyTorch module handles for the torch façade.

  Requires the :torch alias (libpython-clj on the classpath) and a Python
  with PyTorch installed. Nothing here runs until `initialize!` is called,
  so merely having this namespace on the base classpath costs nothing.

  Python selection: pass :python-executable / :library-path, or set the
  MYTHJURE_PYTHON and MYTHJURE_LIBPYTHON env vars, or do nothing — before
  embedding, `initialize!` interrogates the chosen executable in a SUBPROCESS
  (`python-info`) and derives the matching libpython from its own sysconfig,
  so pointing at a pyenv shim or a venv wrapper resolves to the right
  interpreter and the right shared library automatically. The env vars are a
  fallback/override, not a requirement. Historical trap this design removes:
  with only :python-executable set, libpython-clj could silently load the
  SYSTEM libpythonX.Y.so — wrong interpreter, wrong site-packages (symptom:
  ModuleNotFoundError for a package the same `python3 -c \"import …\"`
  clearly has). The Python must be built with --enable-shared; run
  (mythjure.torch.doctor/doctor) for a full pre-flight report.

  All other mythjure.torch.* namespaces call PyTorch through the module
  handles and helpers defined here. Model code must never require
  libpython-clj directly — new ops go into the façade."
  (:require [clojure.java.shell :as shell]
            [clojure.string :as str]
            [libpython-clj2.python :as py]))

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

;; ---------------------------------------------------------------------------
;; Python discovery (subprocess only — nothing is embedded until initialize!)
;; ---------------------------------------------------------------------------

(def ^:private probe-script
  "Python one-shot that prints key=value lines with everything initialize!
  and the doctor need to know about an interpreter, without embedding it."
  (str "import sys, sysconfig\n"
       "def cv(k):\n"
       "    v = sysconfig.get_config_var(k)\n"
       "    print(k + '=' + ('' if v is None else str(v)))\n"
       "print('executable=' + sys.executable)\n"
       "print('version=' + '%d.%d.%d' % sys.version_info[:3])\n"
       "print('prefix=' + sys.prefix)\n"
       "print('base_prefix=' + sys.base_prefix)\n"
       "cv('LIBDIR')\n"
       "cv('LDLIBRARY')\n"
       "cv('INSTSONAME')\n"
       "cv('Py_ENABLE_SHARED')\n"
       "try:\n"
       "    import torch\n"
       "    print('torch_version=' + torch.__version__)\n"
       "except Exception as e:\n"
       "    print('torch_error=' + type(e).__name__ + ': ' + str(e))\n"))

(defn python-info
  "Interrogate a Python executable in a SUBPROCESS (nothing gets embedded, so
  this is safe to call before — or instead of — initialize!). Returns
    {:executable      sys.executable (a pyenv shim/venv wrapper resolves to
                      the real interpreter here)
     :version         \"3.12.11\"
     :prefix :base-prefix
     :enable-shared?  boolean — libpython-clj requires an --enable-shared build
     :libpython       first existing LIBDIR/{LDLIBRARY,INSTSONAME}, or nil
     :torch-version   present iff `import torch` succeeds
     :torch-error     present iff it doesn't}
  or {:error msg} when the executable can't be run at all."
  [exe]
  (try
    (let [{:keys [exit out err]} (shell/sh (str exe) "-c" probe-script)]
      (if-not (zero? exit)
        {:error (str/trim (str out "\n" err))}
        (let [raw (into {}
                        (keep #(when-let [i (str/index-of % "=")]
                                 [(keyword (str/lower-case (subs % 0 i)))
                                  (subs % (inc i))]))
                        (str/split-lines out))
              lib (first (for [f [(:ldlibrary raw) (:instsoname raw)]
                               :when (seq f)
                               :let [p (java.io.File. (str (:libdir raw)) f)]
                               :when (.isFile p)]
                           (.getPath p)))]
          (cond-> {:executable     (:executable raw)
                   :version        (:version raw)
                   :prefix         (:prefix raw)
                   :base-prefix    (:base_prefix raw)
                   :libdir         (:libdir raw)
                   :enable-shared? (= "1" (:py_enable_shared raw))
                   :libpython      lib}
            (:torch_version raw) (assoc :torch-version (:torch_version raw))
            (:torch_error raw)   (assoc :torch-error (:torch_error raw))))))
    (catch java.io.IOException e {:error (.getMessage e)})))

(defn- first-on-path
  "First of `names` found as an executable on PATH, or nil."
  [names]
  (first (for [n names
               dir (str/split (or (System/getenv "PATH") "") #":")
               :let [f (java.io.File. dir n)]
               :when (and (.isFile f) (.canExecute f))]
           (.getPath f))))

(defn resolve-python
  "Decide which Python to embed, without embedding it. Precedence per part:
    executable — :python-executable opt → MYTHJURE_PYTHON → python3/python on PATH
    libpython  — :library-path opt → MYTHJURE_LIBPYTHON → derived from the
                 executable's own sysconfig (LIBDIR + LDLIBRARY/INSTSONAME)
  Returns {:python-executable :library-path :exe-source :lib-source :info}
  where :info is (python-info exe); never throws — missing parts are nil and
  the sources say where each part came from (:option :env :path :discovered)."
  [& {:keys [python-executable library-path]}]
  (let [[exe exe-source] (cond
                           python-executable [python-executable :option]
                           (seq (System/getenv "MYTHJURE_PYTHON")) [(System/getenv "MYTHJURE_PYTHON") :env]
                           :else (when-let [p (first-on-path ["python3" "python"])] [p :path]))
        info (when exe (python-info exe))
        [lib lib-source] (cond
                           library-path [library-path :option]
                           (seq (System/getenv "MYTHJURE_LIBPYTHON")) [(System/getenv "MYTHJURE_LIBPYTHON") :env]
                           (:libpython info) [(:libpython info) :discovered])
        ;; Embed what the interpreter says it IS (sys.executable): a pyenv
        ;; shim de-shims to the real binary; a venv python stays the venv
        ;; python (so its site-packages win).
        real-exe (or (:executable info) exe)]
    (cond-> {:python-executable real-exe
             :exe-source        exe-source
             :library-path      lib
             :lib-source        lib-source
             :info              info}
      (and exe (not= exe real-exe)) (assoc :requested-executable exe))))

(defonce ^:private embedded* (atom nil))

(defn embedded-python
  "The resolve-python result initialize! actually embedded, or nil if not
  yet initialized (used by the doctor to cross-check the live interpreter)."
  []
  @embedded*)

(defn- init-error [msg resolution]
  (ex-info (str msg "\nRun (mythjure.torch.doctor/doctor) for a full report.")
           (assoc (dissoc resolution :info) :info (:info resolution))))

(defn initialize!
  "Embed CPython and import the torch modules. Idempotent; returns :ok.
  Options: :python-executable / :library-path override MYTHJURE_PYTHON /
  MYTHJURE_LIBPYTHON; with neither, the interpreter is found on PATH and its
  libpython derived via sysconfig (see resolve-python). Throws with a
  readable message — and leaves the flag unset so a fixed config can retry —
  when no Python is found, it isn't an --enable-shared build, no shared
  libpython exists, or torch isn't importable."
  [& {:keys [python-executable library-path]}]
  (when (compare-and-set! initialized* false true)
    (try
      (let [{:keys [info] :as r} (resolve-python :python-executable python-executable
                                                 :library-path library-path)
            exe (:python-executable r)
            lib (:library-path r)]
        (when-not exe
          (throw (init-error "No Python found: pass :python-executable, set MYTHJURE_PYTHON, or put python3 on PATH." r)))
        (when (:error info)
          (throw (init-error (str "Python at " exe " could not be run: " (:error info)) r)))
        (when-not (:enable-shared? info)
          (throw (init-error (str "Python at " exe " is not an --enable-shared build; libpython-clj needs the shared libpython. "
                                  "pyenv: PYTHON_CONFIGURE_OPTS=--enable-shared pyenv install <version>.") r)))
        (when-not lib
          (throw (init-error (str "No shared libpython found for " exe " (looked for LDLIBRARY/INSTSONAME under "
                                  (:libdir info) "); pass :library-path or set MYTHJURE_LIBPYTHON.") r)))
        (when-not (.isFile (java.io.File. (str lib)))
          (throw (init-error (str "libpython path " lib " (from " (name (:lib-source r)) ") is not a file.") r)))
        ;; The pyenv trap (2026-07-15): an explicit libpython from a DIFFERENT
        ;; installation than the interpreter embeds the foreign interpreter
        ;; with the wrong site-packages. Refuse rather than segfault-adjacent.
        (when-let [own (:libpython info)]
          (when (and (not= (:lib-source r) :discovered)
                     (not= (.getCanonicalPath (java.io.File. (str lib)))
                           (.getCanonicalPath (java.io.File. (str own)))))
            (throw (init-error (str "libpython " lib " (from " (name (:lib-source r)) ") belongs to a different "
                                    "installation than " exe " (its own is " own "). Unset MYTHJURE_LIBPYTHON / "
                                    "drop :library-path to use the discovered one.") r))))
        (when-let [terr (:torch-error info)]
          (throw (init-error (str "Python at " exe " cannot import torch (" terr "); "
                                  "install PyTorch into that environment (pip install torch).") r)))
        (py/initialize! :python-executable exe :library-path lib)
        (reset! embedded* r))
      (alter-var-root #'torch    (constantly (py/import-module "torch")))
      (alter-var-root #'F        (constantly (py/import-module "torch.nn.functional")))
      (alter-var-root #'optim    (constantly (py/import-module "torch.optim")))
      (alter-var-root #'nn-utils (constantly (py/import-module "torch.nn.utils")))
      (catch Throwable t
        (reset! initialized* false)
        (throw t))))
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

(defn- shape-of
  "Shape of a live tensor as a Clojure vector, nil for anything else. Error-
  path helper only — must not route back through call/assert-pyobj."
  [x]
  (when-not (or (nil? x) (number? x) (string? x) (boolean? x) (coll? x))
    (try (some-> (py/get-attr x "shape") py/->jvm vec)
         (catch Exception _ nil))))

(defn- rethrow-readable
  "Wrap a Python-side failure in an ex-info whose first line is the actual
  Python error (libpython-clj buries it under a full traceback) plus the
  shapes of any tensor args — the usual missing fact in a shape mismatch."
  [^Throwable e ctx obj args]
  (let [lines  (->> (str/split-lines (or (.getMessage e) ""))
                    (remove str/blank?))
        ;; libpython-clj usually surfaces \"ErrorType: msg\" directly; when a
        ;; full Python traceback comes through, the error is its LAST line.
        py-msg (or (if (str/starts-with? (or (first lines) "") "Traceback")
                     (last lines)
                     (first lines))
                   "python call failed")
        shapes (cond-> []
                 (shape-of obj) (conj [:self (shape-of obj)])
                 :always (into (keep-indexed (fn [i a] (when-let [s (shape-of a)] [i s])) args)))]
    (throw (ex-info (str ctx " failed: " py-msg
                         (when (seq shapes) (str " | tensor shapes " (vec shapes))))
                    {:ctx ctx :tensor-shapes shapes}
                    e))))

(defn call
  "Call a method on a Python object: (call t \"reshape\" 2 3)."
  [obj method & args]
  (assert-pyobj obj (str "call ." method))
  (try (apply py/call-attr obj method args)
       (catch clojure.lang.ExceptionInfo e (throw e))
       (catch Exception e (rethrow-readable e (str "call ." method) obj args))))

(defn call-kw
  "Call a method with positional args and keyword args:
  (call-kw F \"layer_norm\" [x [d]] {:weight g :bias b})."
  [obj method pos-args kw-args]
  (assert-pyobj obj (str "call-kw ." method))
  (try (py/call-attr-kw obj method pos-args kw-args)
       (catch clojure.lang.ExceptionInfo e (throw e))
       (catch Exception e (rethrow-readable e (str "call-kw ." method) obj pos-args))))

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

(defn py-list
  "Convert a Clojure collection to a real Python list. Some torch ops accept
  the auto-bridged JVM vector (cat/stack do today), but that is per-op C++
  type-checking luck — use this when passing a list-of-tensors explicitly."
  [coll]
  (py/->py-list coll))

(defn py-vec
  "Materialize a Python sequence (tuple/list, incl. torch's namedtuples) as a
  Clojure vector whose elements stay LIVE Python objects — unlike ->jvm, no
  deep conversion, so tensors come back as tensors (torch.split/chunk/topk
  return tuples of tensors)."
  [obj]
  ;; NOTE: bridged Python tuples/lists implement java.util.List AND look
  ;; coll?-ish to Clojure, so this branch must come before the pyobj guard.
  (if (instance? java.util.List obj)
    (vec obj)
    (do (assert-pyobj obj "py-vec")
        (mapv #(py/get-item obj %)
              (range (py/->jvm (py/call-attr (py/import-module "builtins") "len" obj)))))))

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
