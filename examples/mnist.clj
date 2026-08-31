;; MNIST tutorial + the data-loading probe (direction doc §5.1 item 4, stage 2
;; of the toy-example plan — synthetic first, MNIST second). Like
;; examples/two_moons.clj this is written OUTSIDE-USER style: the
;; mythjure.torch.* facade only, none of mythjure's experiment code. Where
;; two_moons varied the MODEL idiom, this example varies the DATA idiom —
;; the same MLP trains one epoch each through:
;;   (A) Clojure-side batching  — seeded shuffle in Clojure, index-select +
;;       narrow slices, every batch a facade call;
;;   (B) torch DataLoader       — torch.utils.data.TensorDataset/DataLoader
;;       reached by attr-chaining (no torchvision needed); the bridged
;;       loader is java.lang.Iterable, so doseq consumes it directly.
;; Both epochs are timed — the throughput gap IS the finding.
;;
;; Two honest edges an outside user hits here (recorded as §5.0 evidence):
;;   - torchvision is NOT assumed: it is an extra Python install and can drag
;;     a torch version change with it, so this example reads the raw IDX
;;     files itself (downloaded once into data/mnist/, git-ignored).
;;   - Bulk bytes cannot cross the bridge: a JVM byte[] does not implement
;;     the Python buffer protocol (torch.frombuffer rejects it), so the file
;;     is read PYTHON-side via numpy (ubiquitous alongside torch, though
;;     not a hard torch dependency: pip install numpy if missing)
;;     through the facade's ingest lane: core/import-module for
;;     the numpy handle, tensor/from-numpy to wrap the array (shared
;;     buffer; the casts below allocate fresh tensors).
;;
;; Run:  clojure -M:torch examples/mnist.clj    (~11MB download on first run)
(require '[clojure.java.io :as io]
         '[mythjure.torch.core :as tc]
         '[mythjure.torch.tensor :as t]
         '[mythjure.torch.nn :as nn]
         '[mythjure.torch.op :as op]
         '[mythjure.torch.autograd :as ag]
         '[mythjure.torch.optim :as optim]
         '[mythjure.torch.module :as module])

(tc/initialize!)

;; ---------------------------------------------------------------------------
;; Data: download + cache the raw IDX files, ingest via numpy -> from_numpy.
;; ---------------------------------------------------------------------------

(def data-dir "data/mnist")
(def mirror "https://ossci-datasets.s3.amazonaws.com/mnist/")
(def files {:train-x "train-images-idx3-ubyte"
            :train-y "train-labels-idx1-ubyte"
            :test-x  "t10k-images-idx3-ubyte"
            :test-y  "t10k-labels-idx1-ubyte"})

(defn fetch!
  "Download <name>.gz from the mirror and decompress to data/mnist/<name>;
  no-op when the file is already cached."
  [name]
  (let [out (io/file data-dir name)]
    (when-not (.exists out)
      (io/make-parents out)
      (println "  downloading" (str mirror name ".gz"))
      (with-open [in (java.util.zip.GZIPInputStream.
                      (io/input-stream (str mirror name ".gz")))]
        (io/copy in out)))
    (.getPath out)))

;; The facade helpers (tc/call, tc/call-kw) work on ANY Python object,
;; numpy included.
(def np (tc/import-module "numpy"))

(defn read-idx
  "Read an IDX file as a flat uint8 torch tensor (header included) — the
  bytes never enter the JVM: numpy reads the file, from-numpy wraps it."
  [path]
  (t/from-numpy (tc/call-kw np "fromfile" [path] {:dtype "uint8"})))

(defn images
  "IDX3 -> [N 784] float32 in [0,1]: drop the 16-byte header, reshape, cast."
  [path]
  (let [raw (read-idx path)
        n   (quot (- (first (t/shape raw)) 16) 784)]
    (-> (t/narrow raw 0 16 (* n 784))
        (t/reshape [n 784])
        (tc/call "to" (tc/dtype :float32))
        (t/div 255.0))))

(defn labels
  "IDX1 -> [N] int64: drop the 8-byte header, cast."
  [path]
  (let [raw (read-idx path)]
    (-> (t/narrow raw 0 8 (- (first (t/shape raw)) 8))
        (tc/call "to" (tc/dtype :int64)))))

(println "MNIST data (cached in" (str data-dir "/)"))
(def train-x (images (fetch! (:train-x files))))
(def train-y (labels (fetch! (:train-y files))))
(def test-x  (images (fetch! (:test-x files))))
(def test-y  (labels (fetch! (:test-y files))))
(def n-train (first (t/shape train-x)))
(println "  train" (t/shape train-x) "test" (t/shape test-x))

;; ---------------------------------------------------------------------------
;; Model: the param-map idiom from examples/two_moons.clj, 784 -> 128 -> 10.
;; ---------------------------------------------------------------------------

(op/torch-fn "manual_seed" [42])   ; still tier-2 (no curated seed surface)

(defn linear-init [in out]
  {:w (t/mul (t/randn [in out]) (/ 1.0 (Math/sqrt in)))
   :b (t/zeros [out])})

(defn dense [x {:keys [w b]}] (t/add (t/matmul x w) b))

(defn forward [p x] (-> x (dense (:l1 p)) nn/gelu (dense (:l2 p))))

(def params (ag/requires-grad-params! {:l1 (linear-init 784 128)
                                       :l2 (linear-init 128 10)}))
(def opt (optim/adam params :lr 1e-3))

(defn train-step!
  "One minibatch: loss -> backward -> Adam. Returns the loss as a double."
  [bx by]
  (let [loss (nn/cross-entropy (forward params bx) by)]
    (optim/zero-grad! opt)
    (ag/backward! loss)
    (optim/step! opt)
    (t/item loss)))

(defn accuracy [p x y]
  (/ (t/item (t/sum (t/eq (t/argmax (forward p x) :dim 1) y)))
     (double (first (t/shape y)))))

(def batch-size 128)

;; ---------------------------------------------------------------------------
;; Epoch idiom A: Clojure-side batching (seeded shuffle, index-select+narrow).
;; ---------------------------------------------------------------------------

(defn shuffled-indices
  "Seeded Fisher-Yates over 0..n-1, as an int64 tensor."
  [n seed]
  (let [al (java.util.ArrayList. ^java.util.Collection (range n))]
    (java.util.Collections/shuffle al (java.util.Random. (long seed)))
    (t/from-clj (vec al) :dtype :int64)))

(defn epoch-clj!
  "One epoch, batches sliced facade-side. Returns wall seconds."
  [seed]
  (let [t0   (System/nanoTime)
        perm (shuffled-indices n-train seed)
        xs   (t/index-select train-x 0 perm)
        ys   (t/index-select train-y 0 perm)]
    (dotimes [b (quot n-train batch-size)]
      (let [off  (* b batch-size)
            loss (train-step! (t/narrow xs 0 off batch-size)
                              (t/narrow ys 0 off batch-size))]
        (when (zero? (rem b 100))
          (println (format "  [clj batching]  batch %3d  loss %.4f" b loss)))))
    (/ (- (System/nanoTime) t0) 1e9)))

;; ---------------------------------------------------------------------------
;; Epoch idiom B: torch DataLoader through the bridge.
;; ---------------------------------------------------------------------------

(defn epoch-dataloader!
  "One epoch through torch.utils.data.DataLoader (shuffle by torch's seeded
  RNG). The loader is consumed as an ordinary Clojure iterable; each batch is
  a Python [images labels] pair unpacked with tc/py-vec. Returns wall seconds."
  []
  (let [t0       (System/nanoTime)
        data-mod (-> tc/torch (tc/attr "utils") (tc/attr "data"))
        ds       (tc/call data-mod "TensorDataset" train-x train-y)
        dl       (tc/call-kw data-mod "DataLoader" [ds]
                             {:batch_size batch-size :shuffle true :drop_last true})
        b*       (volatile! 0)]
    (doseq [batch dl]
      (let [[bx by] (tc/py-vec batch)
            loss    (train-step! bx by)]
        (when (zero? (rem @b* 100))
          (println (format "  [DataLoader]    batch %3d  loss %.4f" @b* loss)))
        (vswap! b* inc)))
    (/ (- (System/nanoTime) t0) 1e9)))

;; ---------------------------------------------------------------------------
;; Train one epoch per idiom, compare, evaluate, checkpoint round-trip.
;; ---------------------------------------------------------------------------

(println "Epoch 1 — Clojure-side batching")
(def secs-a (epoch-clj! 7))
(println (format "  %.1f s" secs-a))

(println "Epoch 2 — torch DataLoader through the bridge")
(def secs-b (epoch-dataloader!))
(println (format "  %.1f s" secs-b))

(def acc (accuracy params test-x test-y))
(println (format "Test accuracy after 2 epochs: %.4f" acc))
(println (format "Throughput: clj batching %.1f s/epoch, DataLoader %.1f s/epoch"
                 secs-a secs-b))

;; Checkpoint round trip: params -> .pt file -> params, accuracy identical.
(def ckpt (str data-dir "/mnist_mlp.pt"))
(module/save! params ckpt)
(def reloaded (module/load-params ckpt))
(def acc2 (accuracy reloaded test-x test-y))
(println (format "Reloaded checkpoint accuracy: %.4f" acc2))

(assert (>= acc 0.93) (str "test accuracy too low: " acc))
(assert (= acc acc2) "checkpoint round-trip changed the model")
(println "OK — MNIST MLP >=0.93 test accuracy; checkpoint round-trips exactly.")
