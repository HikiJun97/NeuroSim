import argparse
import os
import pickle
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate CIFAR-10 python-version batches (cifar-10-batches-py/) for training usability."
    )
    p.add_argument(
        "--root",
        default="./cifar-10-batches-py",
        help="Path to cifar-10-batches-py directory (contains data_batch_1..5, test_batch, batches.meta).",
    )
    return p.parse_args()


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def require_files(root: str):
    required = [*(f"data_batch_{i}" for i in range(1, 6)), "test_batch", "batches.meta"]
    missing = [f for f in required if not os.path.exists(os.path.join(root, f))]
    if missing:
        raise FileNotFoundError(f"Missing required files in {root}: {missing}")


def check_batch(name: str, batch: dict, expect_n: int):
    # Import numpy lazily so error message is clear if environment is missing it.
    try:
        import numpy as np
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "numpy is required to unpickle and validate CIFAR-10 batch files. "
            "Please install numpy in your environment and re-run."
        ) from e

    required_keys = [b"batch_label", b"labels", b"data", b"filenames"]
    for k in required_keys:
        if k not in batch:
            keys = list(batch.keys())
            raise KeyError(f"{name}: missing key {k!r}. Present keys={keys}")

    data = batch[b"data"]
    labels = batch[b"labels"]
    filenames = batch[b"filenames"]

    # Data sanity (CIFAR-10 python version uses uint8, N x 3072)
    if not isinstance(data, np.ndarray):
        raise TypeError(f"{name}: data type {type(data)}; expected numpy.ndarray")
    if data.ndim != 2:
        raise ValueError(f"{name}: data.ndim={data.ndim}; expected 2")
    if data.shape != (expect_n, 3072):
        raise ValueError(
            f"{name}: data.shape={data.shape}; expected ({expect_n}, 3072)"
        )
    if data.dtype != np.uint8:
        raise TypeError(f"{name}: data.dtype={data.dtype}; expected uint8")

    # Labels sanity
    if len(labels) != expect_n:
        raise ValueError(f"{name}: labels len={len(labels)}; expected {expect_n}")
    lab_min = int(np.min(labels))
    lab_max = int(np.max(labels))
    if not (0 <= lab_min <= lab_max <= 9):
        raise ValueError(
            f"{name}: label range {lab_min}..{lab_max}; expected within 0..9"
        )

    # Filenames sanity
    if len(filenames) != expect_n:
        raise ValueError(f"{name}: filenames len={len(filenames)}; expected {expect_n}")

    print(
        f"OK {name}: data={data.shape} {data.dtype}, labels range {lab_min}..{lab_max}"
    )


def main():
    args = parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))

    print("python:", sys.version.split()[0])
    print("root:", root)

    require_files(root)

    # Validate 5 training batches (each 10k) and 1 test batch (10k)
    for i in range(1, 6):
        name = f"data_batch_{i}"
        batch = load_pickle(os.path.join(root, name))
        check_batch(name, batch, expect_n=10000)

    test_batch = load_pickle(os.path.join(root, "test_batch"))
    check_batch("test_batch", test_batch, expect_n=10000)

    # Validate meta (label names)
    meta = load_pickle(os.path.join(root, "batches.meta"))
    label_names = meta.get(b"label_names")
    if label_names is None:
        raise KeyError(
            f"batches.meta: missing b'label_names'. Present keys={list(meta.keys())}"
        )
    if len(label_names) != 10:
        raise ValueError(
            f"batches.meta: label_names len={len(label_names)}; expected 10"
        )
    decoded = [
        x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in label_names
    ]
    print("OK batches.meta: label_names =", decoded)

    print("DONE: cifar-10-batches-py looks valid and usable for training.")


if __name__ == "__main__":
    main()
