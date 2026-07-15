(ns mythjure.torch-mutation-test
  "Mutation & view semantics (direction doc §5.1 item 2).

  The convention under test:
   1. No unmarked façade op mutates its arguments — mutation is opt-in via
      the !-suffixed ops only. Aliasing is only observable through mutation,
      so the rule of thumb is: no !, no aliasing bugs.
   2. !-ops guard-throw AT THE MUTATION SITE when the target is an autograd
      intermediate (grad_fn present) — instead of torch's version-counter
      error later at backward!, far from the mutation. :unsafe true bypasses;
      leaves are left to torch (refused under grad mode, legal under no-grad
      — the optimizer pattern).
   3. Aliasing is declared data — tensor/aliasing-table — and every claim is
      pinned empirically here (data-ptr / mutation witnesses), so a torch
      upgrade that changes an op's aliasing fails a test instead of silently
      corrupting user weights.

  Runs under the :test-torch alias:  clojure -M:test-torch"
  (:require [clojure.set :as set]
            [clojure.test :refer [deftest is testing use-fixtures]]
            [mythjure.torch.autograd :as ag]
            [mythjure.torch.core :as core]
            [mythjure.torch.module :as mod]
            [mythjure.torch.tensor :as t]))

(use-fixtures :once (fn [f] (core/initialize!) (f)))

(defn- ->t64 [x] (t/from-clj x :dtype :float64))
(defn- X [] (->t64 [[1.0 2.0] [3.0 4.0]]))
(defn- V [] (->t64 [1.0 2.0 3.0 4.0]))

(defn- shares-ptr? [a b] (= (t/data-ptr a) (t/data-ptr b)))

(defn- mutates-target?
  "True when (op! x …) writes through x and returns x itself."
  [op! & args]
  (let [x (V)
        r (apply op! x args)]
    (and (shares-ptr? r x)
         (not= [1.0 2.0 3.0 4.0] (t/to-clj x)))))

;; ---------------------------------------------------------------------------
;; 1. The aliasing table: every claim gets an empirical witness, and the
;;    witnesses cover exactly the table (both directions), so adding an
;;    aliased op without pinning it here fails.
;; ---------------------------------------------------------------------------

(def ^:private aliasing-witnesses
  {'transpose    #(let [x (X)] (shares-ptr? x (t/transpose x)))
   'narrow       #(let [x (V)]
                    (and (shares-ptr? x (t/narrow x 0 0 2))
                         ;; offset view: different ptr but still aliased —
                         ;; the mutation witness, not data-ptr, proves it
                         (do (t/fill! (t/narrow x 0 1 2) 9.0)
                             (= [1.0 9.0 9.0 4.0] (t/to-clj x)))))
   'squeeze      #(let [x (t/unsqueeze (V) 0)] (shares-ptr? x (t/squeeze x 0)))
   'unsqueeze    #(let [x (V)] (shares-ptr? x (t/unsqueeze x 0)))
   'reshape      #(let [x (X)]
                    (and (shares-ptr? x (t/reshape x [4]))                     ; contiguous → view
                         (not (shares-ptr? x (t/reshape (t/transpose x) [4]))))) ; non-contiguous → copy
   'to-device    #(let [x (V)] (shares-ptr? x (t/to-device x "cpu")))          ; already there → self
   'einsum       #(let [x (X)]
                    (and (shares-ptr? x (t/einsum "ij->ji" x))                 ; pure permutation → view
                         (not (shares-ptr? x (t/einsum "ij,jk->ik" x x)))))    ; contraction → fresh
   'split        #(let [x (V)] (shares-ptr? x (first (t/split x 2))))
   'chunk        #(let [x (V)] (shares-ptr? x (first (t/chunk x 2))))
   'unbind       #(let [x (X)] (shares-ptr? x (first (t/unbind x))))
   'permute      #(let [x (X)] (shares-ptr? x (t/permute x [1 0])))
   'broadcast-to #(let [x (V)] (shares-ptr? x (t/broadcast-to x [2 4])))
   'flatten      #(let [x (X)]
                    (and (shares-ptr? x (t/flatten x))                         ; contiguous → view
                         (not (shares-ptr? (t/transpose x) (t/flatten (t/transpose x))))))
   'clone        #(let [x (V) c (t/clone x)]
                    (and (not (shares-ptr? x c))
                         (do (t/fill! c 9.0)                                   ; fresh storage:
                             (= [1.0 2.0 3.0 4.0] (t/to-clj x)))))             ; original untouched
   'contiguous   #(let [x (X)]
                    (and (shares-ptr? x (t/contiguous x))                      ; already contiguous → self
                         (not (shares-ptr? x (t/contiguous (t/transpose x)))))) ; else → copy
   'add!         #(mutates-target? t/add! 1.0)
   'sub!         #(mutates-target? t/sub! 1.0)
   'mul!         #(mutates-target? t/mul! 2.0)
   'div!         #(mutates-target? t/div! 2.0)
   'clamp!       #(mutates-target? t/clamp! :max 2.0)
   'zero!        #(mutates-target? t/zero!)
   'fill!        #(mutates-target? t/fill! 7.0)})

(deftest every-aliasing-claim-has-a-passing-witness
  (let [claimed   (set (keys (t/aliasing-table)))
        witnessed (set (keys aliasing-witnesses))]
    (is (empty? (set/difference claimed witnessed))
        (str "aliasing claims without a witness: " (set/difference claimed witnessed)))
    (is (empty? (set/difference witnessed claimed))
        (str "witnesses without an aliasing claim: " (set/difference witnessed claimed)))
    (doseq [[sym witness] (sort aliasing-witnesses)]
      (testing (str sym " behaves as (t/aliasing-table) claims")
        (is (witness))))))

(deftest unlisted-ops-return-fresh-storage
  ;; spot-check the table's complement: out-of-place ops allocate
  (let [x (X)]
    (is (not (shares-ptr? x (t/add x 1.0))))
    (is (not (shares-ptr? x (t/abs x))))
    (is (not (shares-ptr? x (t/index-select x 0 (t/from-clj [0 1] :dtype :int64)))))
    (is (not (shares-ptr? x (t/masked-fill x (t/gt x (->t64 2.0)) 0.0))))))

;; ---------------------------------------------------------------------------
;; 2. The demo-1 corruption class, pinned as intended behavior: views alias,
;;    ! writes through them, clone is the firebreak.
;; ---------------------------------------------------------------------------

(deftest mutation-through-a-view-reaches-the-base
  (let [a (X)
        b (t/reshape a [4])]
    (t/mul! b 100.0)
    (is (= [[100.0 200.0] [300.0 400.0]] (t/to-clj a)))))

(deftest clone-isolates
  (let [a (X)
        b (t/clone (t/reshape a [4]))]
    (t/mul! b 100.0)
    (is (= [[1.0 2.0] [3.0 4.0]] (t/to-clj a)))))

;; ---------------------------------------------------------------------------
;; 3. The autograd guard.
;; ---------------------------------------------------------------------------

(deftest inplace-on-an-intermediate-throws-at-the-mutation-site
  (let [leaf  (ag/requires-grad! (X))
        inter (t/mul leaf 2.0)
        e     (try (t/add! inter 1.0) nil
                   (catch clojure.lang.ExceptionInfo e e))]
    (is (some? e))
    (is (re-find #"grad_fn MulBackward0" (ex-message e)))
    (is (re-find #":unsafe" (ex-message e)))
    (is (= {:op "add!" :grad-fn "MulBackward0"} (ex-data e)))
    ;; the guard fired BEFORE any write — that is its whole point
    (is (= [[2.0 4.0] [6.0 8.0]] (t/to-clj inter)))))

(deftest unsafe-bypasses-the-guard-and-is-stripped-from-kwargs
  (let [inter (t/mul (ag/requires-grad! (X)) 2.0)]
    ;; success proves :unsafe never reached python (add_ has no such kwarg)
    (is (= [[3.0 5.0] [7.0 9.0]] (t/to-clj (t/add! inter 1.0 :unsafe true))))
    ;; kwarg-taking ! ops strip it too
    (is (= [[2.0 2.0] [2.0 2.0]]
           (t/to-clj (t/clamp! (t/mul (ag/requires-grad! (X)) 2.0)
                               :max 2.0 :unsafe true))))))

(deftest leaf-semantics-are-torchs-own
  (let [leaf (ag/requires-grad! (X))]
    ;; under grad mode torch itself refuses — readable via rethrow-readable
    (is (thrown-with-msg? Exception #"leaf Variable that requires grad"
                          (t/add! leaf 1.0)))
    ;; under no-grad it is the optimizer pattern, and stays legal
    (ag/no-grad (t/add! leaf 1.0))
    (is (= [[2.0 3.0] [4.0 5.0]] (t/to-clj leaf)))))

(deftest plain-tensors-mutate-without-ceremony
  (let [x (V)]
    (is (= [3.0 4.0 5.0 6.0] (t/to-clj (-> x (t/add! 1.0) (t/add! 1.0)))))))

;; ---------------------------------------------------------------------------
;; 4. Aliasing facts beyond mythjure.torch.tensor — detach and state_dict cut
;;    the GRAPH, not the STORAGE. Neither has a grad_fn afterward, so the
;;    !-guard cannot see these; the docstrings are the defense, pinned here.
;; ---------------------------------------------------------------------------

(deftest detach-shares-storage
  (let [x (ag/requires-grad! (V))
        d (ag/detach x)]
    (is (shares-ptr? x d))
    (ag/no-grad (t/fill! d 9.0))
    (is (= [9.0 9.0 9.0 9.0] (t/to-clj x)))))

(deftest state-dict-default-shares-storage-with-live-weights
  (let [m    (mod/make "Linear" [2 2])
        sd-w (get (mod/state-dict m) "weight")
        live (:weight (mod/params m))]
    (is (nil? (ag/grad-fn sd-w)))          ; detached — invisible to the guard
    (is (shares-ptr? sd-w live))           ; ...but NOT independent
    (t/fill! sd-w 0.0)
    (is (zero? (t/norm live)))             ; the write reached the module
    ;; the snapshot recipe from the docstring:
    (let [snap (t/clone (get (mod/state-dict m) "weight"))]
      (t/fill! snap 7.0)
      (is (zero? (t/norm live))))))
