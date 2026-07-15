(ns mythjure.torch.doctor
  "Pre-flight diagnostics for the torch façade — run this FIRST when anything
  about Python/torch setup misbehaves:

    (require '[mythjure.torch.doctor :as doctor])
    (doctor/doctor)

  Everything except the embedded-state check runs in a SUBPROCESS, so the
  doctor is safe to call before initialize! — including when initialization
  itself is what's broken (an embedded CPython cannot be un-embedded; the
  whole point is to catch a bad configuration before it reaches the JVM).

  `doctor` prints a human-readable report and returns the same findings as
  data; `checkup` is the pure data version. Both take the same
  :python-executable / :library-path options as core/initialize!.

  Façade-internal namespace: allowed to touch libpython-clj directly (for
  the live embedded-interpreter cross-check)."
  (:require [clojure.string :as str]
            [libpython-clj2.python :as py]
            [mythjure.torch.core :as core]))

(defn- check [status name msg & [fix]]
  (cond-> {:status status :name name :msg msg}
    fix (assoc :fix fix)))

(defn- lib-consistency
  "The pyenv trap detector: an explicitly configured libpython that is NOT
  the one the interpreter's own sysconfig points at means the embedded
  CPython and the site-packages come from DIFFERENT installations."
  [{:keys [library-path lib-source info]}]
  (let [discovered (:libpython info)]
    (cond
      (nil? library-path)
      (check :fail "libpython"
             (str "no shared libpython found (sysconfig LIBDIR = " (:libdir info) ")")
             "pass :library-path / set MYTHJURE_LIBPYTHON, or rebuild Python with --enable-shared")

      (not (.isFile (java.io.File. (str library-path))))
      (check :fail "libpython"
             (str library-path " (from " (name lib-source) ") is not a file")
             "fix MYTHJURE_LIBPYTHON / :library-path")

      ;; THE pyenv trap (hit 2026-07-15): exe from one installation, libpython
      ;; from another. The subprocess probe imports torch fine (it only sees
      ;; the exe), but the EMBEDDED interpreter is the foreign libpython with
      ;; the wrong site-packages — ModuleNotFoundError at initialize! time.
      (and discovered
           (not= lib-source :discovered)
           (not= (.getCanonicalPath (java.io.File. (str library-path)))
                 (.getCanonicalPath (java.io.File. (str discovered)))))
      (check :fail "libpython"
             (str library-path " (from " (name lib-source) ") is NOT the interpreter's own "
                  discovered " — the embedded interpreter would use the wrong site-packages")
             "unset MYTHJURE_LIBPYTHON / drop :library-path to use the discovered one, or point both at the same installation")

      :else
      (check :ok "libpython"
             (str library-path
                  (if (= lib-source :discovered)
                    " (discovered via sysconfig)"
                    (str " (from " (name lib-source) ", matches interpreter)")))))))

(defn- embedded-check
  "Cross-check the LIVE embedded interpreter (if any) against the target
  resolution — catches 'the env changed since this JVM embedded Python'."
  [{:keys [python-executable]}]
  (if-not (core/initialized?)
    (check :ok "embedded" "not yet initialized — (core/initialize!) will embed the Python above")
    (try
      (let [sys      (py/import-module "sys")
            live-exe (str (py/get-attr sys "executable"))
            ;; torch.__version__ is a str SUBCLASS (TorchVersion); ->jvm
            ;; walks it like a sequence — str is the correct conversion.
            live-tv  (str (py/get-attr core/torch "__version__"))]
        (if (or (nil? python-executable) (= live-exe python-executable))
          (check :ok "embedded"
                 (str "initialized; embedded " live-exe " (torch " live-tv ")"))
          (check :warn "embedded"
                 (str "embedded interpreter " live-exe " differs from the currently-resolved "
                      python-executable)
                 "an embedded CPython cannot be swapped — restart the JVM to pick up the new config")))
      (catch Exception e
        (check :fail "embedded" (str "initialized but live check failed: " (ex-message e)))))))

(defn checkup
  "Run all environment checks; returns
    {:ok? bool  ; no :fail (warnings don't fail the checkup)
     :resolution <core/resolve-python result>
     :checks [{:status :ok|:warn|:fail :name :msg :fix?} ...]}
  Same options as core/initialize!. Pure data — see `doctor` for the
  printed report."
  [& {:keys [python-executable library-path]}]
  (let [{:keys [info exe-source requested-executable] :as r}
        (core/resolve-python :python-executable python-executable
                             :library-path library-path)
        exe    (:python-executable r)
        checks (if-not exe
                 [(check :fail "python executable"
                         "none found (no :python-executable option, no MYTHJURE_PYTHON, no python3/python on PATH)"
                         "install Python 3 with PyTorch, or point MYTHJURE_PYTHON at one")]
                 (concat
                  [(check :ok "python executable"
                          (str exe " (from " (name exe-source) ")"
                               (when requested-executable
                                 (str " — resolved from " requested-executable))))]
                  (if (:error info)
                    [(check :fail "interpreter" (str "cannot run: " (:error info)))]
                    [(check :ok "interpreter"
                            (str "Python " (:version info)
                                 (when (not= (:prefix info) (:base-prefix info))
                                   (str " (venv; base " (:base-prefix info) ")"))))
                     (if (:enable-shared? info)
                       (check :ok "shared build" "--enable-shared ✓")
                       (check :fail "shared build"
                              "NOT built with --enable-shared; libpython-clj cannot embed it"
                              "pyenv: PYTHON_CONFIGURE_OPTS=--enable-shared pyenv install <version>"))
                     (lib-consistency r)
                     (if-let [tv (:torch-version info)]
                       (check :ok "torch" tv)
                       (check :fail "torch"
                              (str "not importable: " (:torch-error info))
                              (str (:executable info) " -m pip install torch")))
                     (embedded-check r)])))]
    {:ok?        (not-any? #(= :fail (:status %)) checks)
     :resolution r
     :checks     (vec checks)}))

(defn doctor
  "Print a readable environment report; returns the `checkup` data.
  Same options as core/initialize!."
  [& {:as opts}]
  (let [{:keys [ok? checks] :as result} (apply checkup (mapcat identity opts))
        glyph {:ok "✓" :warn "!" :fail "✗"}
        width (apply max (map (comp count :name) checks))]
    (println "mythjure.torch doctor")
    (println (apply str (repeat 21 "─")))
    (doseq [{:keys [status name msg fix]} checks]
      (println (str " " (glyph status) " "
                    name (apply str (repeat (- width (count name)) " "))
                    "  " msg))
      (when fix
        (println (str "   " (apply str (repeat width " ")) "  fix: " fix))))
    (println (cond
               (not ok?) "Problems found — fix the ✗ items above before (core/initialize!)."
               (some #(= :warn (:status %)) checks) "Checks passed with warnings (!) — read them before (core/initialize!)."
               :else "All checks passed — this Python will embed cleanly."))
    result))
