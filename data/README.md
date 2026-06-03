# Data Files

`scripts/run_split_mnist.py` expects preprocessed binary files under `data/`.

Use:

```bash
python3 scripts/download_mnist.py
```

Expected generated files:

```text
data/mnist_train.bin
data/mnist_test.bin
data/fashion-mnist_train.bin
data/fashion-mnist_test.bin
```

The binary format used by the runner is:

```text
uint32 little-endian: number of examples
uint32 little-endian: image size, expected to be 784
repeated examples:
  uint8 label
  uint8[784] flattened image pixels
```

The runner loads these files, converts pixels to float32, scales by `1/255`,
and standardizes each pixel dimension with the dataset mean and standard
deviation.

Dataset binaries are generated artifacts and should not be committed to source
releases.
