"""Load data ADIKARA 2025 (prediksi gagal bayar pinjaman)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

TARGET = "status_gagal_bayar"
ID = "id_transaksi"
CATEGORICAL = ["provinsi", "jenis_pinjaman", "status_peminjam", "sektor_usaha", "pendidikan", "jenis_jaminan"]
NUMERIC = ["jumlah_pinjaman", "total_pengembalian", "durasi_hari", "porsi_pengembalian_lender"]


def find_data_dir(local: str | Path = "../data/raw") -> Path:
    """Folder yang berisi train.csv kompetisi ini (di /kaggle/input, ikuti symlink)."""
    for root, _, files in os.walk("/kaggle/input", followlinks=True):
        if {"train.csv", "test.csv"} <= set(files):
            if TARGET in pd.read_csv(Path(root) / "train.csv", nrows=1).columns:
                return Path(root)
    return Path(local)


def load(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = Path(data_dir)
    sample = next(d.glob("sample_submission*.csv"))
    return pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv"), pd.read_csv(sample)
