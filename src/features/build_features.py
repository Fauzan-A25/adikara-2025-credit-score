"""Fitur turunan untuk prediksi gagal bayar, dikelompokkan agar bisa diuji satu per satu (ablation)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load_data import CATEGORICAL, NUMERIC

GROUPS = {
    "A. rasio pengembalian": ["markup", "markup_dari_lender"],
    "B. porsi lender": ["porsi_lender", "lender_per_pinjaman"],
    "C. waktu pencairan": ["bulan_ke", "hari_dalam_minggu"],
    "D. pola data kosong": ["n_kosong"],
}


def base_frame(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fitur mentah, dengan kategori memakai satu kamus untuk train dan test."""
    Xtr, Xte = train[NUMERIC + CATEGORICAL].copy(), test[NUMERIC + CATEGORICAL].copy()
    for c in CATEGORICAL:
        cats = sorted(set(Xtr[c].dropna()) | set(Xte[c].dropna()))
        Xtr[c] = pd.Categorical(Xtr[c], categories=cats)
        Xte[c] = pd.Categorical(Xte[c], categories=cats)
    return Xtr, Xte


def derived(d: pd.DataFrame, typical_share: dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    loan = d.jumlah_pinjaman.replace(0, np.nan)
    out["markup"] = d.total_pengembalian / loan
    # jika total pengembalian kosong: perkirakan dari porsi lender dibagi porsi lender yang umum untuk jenis pinjamannya
    share = d.jenis_pinjaman.map(typical_share).astype(float)
    out["markup_dari_lender"] = d.porsi_pengembalian_lender / share / loan
    out["porsi_lender"] = d.porsi_pengembalian_lender / d.total_pengembalian.replace(0, np.nan)
    out["lender_per_pinjaman"] = d.porsi_pengembalian_lender / loan
    t = pd.to_datetime(d.tanggal_pencairan)
    out["bulan_ke"] = (t.dt.year - 2022) * 12 + t.dt.month
    out["hari_dalam_minggu"] = t.dt.dayofweek
    out["n_kosong"] = d[NUMERIC + ["jenis_pinjaman"]].isna().sum(axis=1)
    return out


def typical_lender_share(train: pd.DataFrame) -> dict[str, float]:
    s = (train.porsi_pengembalian_lender / train.total_pengembalian).round(2)
    s = s[(s > 0) & (s < 1)]
    return s.groupby(train.jenis_pinjaman).agg(lambda v: v.mode().iloc[0]).to_dict()


def build(base: pd.DataFrame, raw: pd.DataFrame, share: dict[str, float], groups: list[str] | None = None) -> pd.DataFrame:
    groups = list(GROUPS) if groups is None else groups
    der = derived(raw, share)
    cols = [c for g in groups for c in GROUPS[g]]
    return pd.concat([base, der[cols]], axis=1)
