#!/usr/bin/env python3
"""Download MNIST and Fashion-MNIST, export to simple binary format for Rust.

Output format per file:
  - 4 bytes: uint32 LE num_samples
  - 4 bytes: uint32 LE image_size (784)
  - For each sample:
    - 1 byte: label (0-9)
    - 784 bytes: pixel values (0-255)
"""

import gzip
import os
import struct
import urllib.request

DATASETS = {
    "mnist": {
        "train-images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "train-labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "test-images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "test-labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
    },
    "fashion-mnist": {
        "train-images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "train-labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
        "test-images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        "test-labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
    },
}


def download(url, path):
    if os.path.exists(path):
        print(f"  {path} already exists")
        return
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, path)


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2051
        n = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]
        data = f.read(n * rows * cols)
        return n, rows * cols, data


def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2049
        n = struct.unpack(">I", f.read(4))[0]
        data = f.read(n)
        return n, data


def export(images_path, labels_path, out_path):
    n_img, img_size, img_data = read_idx_images(images_path)
    n_lbl, lbl_data = read_idx_labels(labels_path)
    assert n_img == n_lbl
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", n_img, img_size))
        for i in range(n_img):
            f.write(bytes([lbl_data[i]]))
            f.write(img_data[i * img_size : (i + 1) * img_size])
    print(f"  Exported {n_img} samples to {out_path}")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    for dataset_name, urls in DATASETS.items():
        print(f"\n=== {dataset_name} ===")
        raw_dir = os.path.join(out_dir, "raw", dataset_name)
        os.makedirs(raw_dir, exist_ok=True)

        for name, url in urls.items():
            download(url, os.path.join(raw_dir, name + ".gz"))

        export(
            os.path.join(raw_dir, "train-images.gz"),
            os.path.join(raw_dir, "train-labels.gz"),
            os.path.join(out_dir, f"{dataset_name}_train.bin"),
        )
        export(
            os.path.join(raw_dir, "test-images.gz"),
            os.path.join(raw_dir, "test-labels.gz"),
            os.path.join(out_dir, f"{dataset_name}_test.bin"),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
