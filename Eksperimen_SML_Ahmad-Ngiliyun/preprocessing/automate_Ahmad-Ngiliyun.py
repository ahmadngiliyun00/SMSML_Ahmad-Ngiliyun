#!/usr/bin/env python3
"""
automate_Ahmad-Ngiliyun.py
- Membaca dataset mentah dari:   namadataset_raw/data.csv
- Melakukan preprocessing dasar:
  - Pisah fitur X dan label y
  - Encode label -> integer + simpan label_mapping.json
  - One-hot untuk kolom kategorikal
  - Train-test split
  - Simpan artefak ke: preprocessing/namadataset_preprocessing/
    (X_train.npy, X_test.npy, y_train.npy, y_test.npy, feature_names.npy)
"""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="../namadataset_raw/data.csv", help="Path dataset mentah (csv)")
    p.add_argument("--outdir", default="./namadataset_preprocessing", help="Folder output preprocessing")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    # WAJIB kamu sesuaikan:
    p.add_argument("--target-col", default="label", help="Nama kolom target/label di CSV")
    # Opsional:
    p.add_argument("--drop-cols", default="", help="Kolom yang dibuang, pisahkan koma. contoh: id,timestamp")
    return p.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    input_path = (script_dir / args.input).resolve()
    outdir = (script_dir / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {input_path}")

    df = pd.read_csv(input_path, sep=";")

    if args.drop_cols.strip():
        drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if args.target_col not in df.columns:
        raise ValueError(
            f"Kolom target '{args.target_col}' tidak ada. Kolom tersedia: {list(df.columns)}"
        )

    # y = label, X = fitur
    y_raw = df[args.target_col].astype(str)
    X = df.drop(columns=[args.target_col])

    # One-hot untuk kolom kategorikal (object/category/bool)
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # Pastikan numerik, isi NaN
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Encode label -> integer
    classes = sorted(y_raw.unique().tolist())
    label_to_id = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(label_to_id).astype(int)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values.astype(np.float32),
        y.values.astype(np.int64),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y.values if len(classes) > 1 else None,
    )

    # Simpan artefak
    np.save(outdir / "X_train.npy", X_train)
    np.save(outdir / "X_test.npy", X_test)
    np.save(outdir / "y_train.npy", y_train)
    np.save(outdir / "y_test.npy", y_test)
    np.save(outdir / "feature_names.npy", np.array(X.columns.tolist(), dtype=object))

    with open(outdir / "label_mapping.json", "w") as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": {v: k for k, v in label_to_id.items()}}, f, indent=2)

    print("✅ Preprocessing selesai")
    print(f"Input   : {input_path}")
    print(f"Output  : {outdir}")
    print(f"X shape : train={X_train.shape}, test={X_test.shape}")
    print(f"y shape : train={y_train.shape}, test={y_test.shape}")
    print(f"Classes : {classes}")


if __name__ == "__main__":
    main()