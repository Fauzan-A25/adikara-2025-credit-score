"""Generate notebook riset ADIKARA 2025 (prediksi gagal bayar, metrik macro F1) dan script runner Kaggle.

Output:
- kaggle-notebook/adikara-2025-credit-default.ipynb  (notebook bersih, di-push sebagai notebook Kaggle)
- kaggle/run_on_kaggle.py                             (script kernel terpisah untuk mengambil notebook ber-output)
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
NB_NAME = "adikara-2025-credit-default.ipynb"
FINDINGS = json.loads((ROOT / "_findings.json").read_text()) if (ROOT / "_findings.json").exists() else {}


def T(key: str) -> str:
    return FINDINGS.get(key, "_(temuan diisi setelah run pertama)_")


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ════════════════════════════════════════════════════════════════════ Header
md(
    f"""
# Memprediksi Gagal Bayar Pinjaman UMKM: Sinyal Terkuat Ada di Rasio Pengembalian

| | |
| --- | --- |
| **Kompetisi** | [ADIKARA 2025: Indonesian Credit Score](https://www.kaggle.com/competitions/adikara-2025-indonesia-kredit-macet) (Fakultas Informatika Telkom University, Kaggle, Desember 2025) |
| **Tujuan** | Memprediksi `status_gagal_bayar` (1 = gagal bayar) untuk transaksi pinjaman digital (metrik macro F1) |
| **Data** | 20.012 transaksi berlabel dan 5.003 transaksi uji, 11 atribut pinjaman dan peminjam |
| **Penulis** | Fauzan Ahsanudin Alfikri · [Kaggle: fauzan2](https://www.kaggle.com/fauzan2) |
| **Tanggal** | Disusun ulang November 2026, menggantikan notebook kompetisi Desember 2025 |
| **Lingkungan** | Dijalankan penuh di Kaggle (CPU) |

**Pertanyaan penelitian.** Atribut apa yang paling membedakan pinjaman yang gagal bayar, dan bagaimana mengubah
probabilitas model menjadi label 0/1 yang memaksimalkan macro F1?

**Ringkasan temuan.**
{T("summary")}

**Alur:** 1. Setup → 2. Memahami data → 3. EDA → 4. Preprocessing dan desain validasi → 5. Baseline →
6. Feature engineering dan ablation → 7. Ambang keputusan → 8. Interpretasi → 9. Submission → 10. Kesimpulan
"""
)

md(
    """
## 1. Setup

Semua import, parameter, dan path ada di sel ini. Fitur turunan ada di `src/features`, dan cross-validation serta pemilihan
ambang ada di `src/models`.
"""
)

code(
    """
import os
import shutil
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SRC_ROOT = Path.cwd().parent
for r, _, f in os.walk("/kaggle/input", followlinks=True):
    if Path(r).name == "features" and "build_features.py" in f:
        pkg = Path(r).parent
        if pkg.name == "src":
            SRC_ROOT = pkg.parent
        else:
            SRC_ROOT = Path("/tmp/project")
            shutil.copytree(pkg, SRC_ROOT / "src", dirs_exist_ok=True)
        break
sys.path.insert(0, str(SRC_ROOT))
from src.data.load_data import CATEGORICAL, ID, NUMERIC, TARGET, find_data_dir, load
from src.features.build_features import GROUPS, base_frame, build, derived, typical_lender_share
from src.models.train_model import THRESHOLDS, best_threshold, cv_predict, f1_curve, lgb_model, macro_f1_by_seed
from src.visualization.plots import PALETTE, pct_axis, save_fig, set_style

# ── Parameter eksperimen
SEEDS = range(5)                 # Stratified 5-Fold diulang dengan 5 seed

# ── Path
DATA_DIR = find_data_dir(local="../data/raw")
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("../reports")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
set_style()
C1, C2, C3, CN = PALETTE["normal"], PALETTE["fraud"], PALETTE["accent"], PALETTE["neutral"]
T0 = time.time()
print("DATA_DIR:", DATA_DIR, "| OUT_DIR:", OUT_DIR)
"""
)

# ════════════════════════════════════════════════════════════════════ Data
md(
    """
## 2. Memahami data

Setiap baris adalah satu pinjaman: tanggal pencairan, provinsi, jenis pinjaman, status peminjam (baru atau berulang),
jumlah pinjaman, total pengembalian, durasi, porsi pengembalian untuk lender, sektor usaha, pendidikan, dan jenis jaminan.
"""
)

code(
    """
train, test, sample = load(DATA_DIR)
y = train[TARGET].values
print(f"Train: {train.shape} | Test: {test.shape} | gagal bayar: {y.mean():.2%} ({y.sum():,} transaksi)")
miss = pd.DataFrame({"kosong train": train.isna().mean(), "kosong test": test.isna().mean()})
display(train.head(3))
miss[miss.sum(1) > 0].style.format("{:.1%}")
"""
)

md("**Temuan.** " + T("data"))

# ════════════════════════════════════════════════════════════════════ EDA
md("## 3. Exploratory Data Analysis\n\n### 3.1 Atribut kategori: hanya dua yang berbeda")

code(
    """
fig, axes = plt.subplots(2, 3, figsize=(17, 8))
for a, c in zip(axes.flat, CATEGORICAL):
    g = train.groupby(train[c].fillna("(kosong)"))[TARGET].agg(["mean", "size"]).sort_values("mean")
    a.barh(g.index.astype(str), g["mean"], color=[C2 if v > 2 * y.mean() else C1 for v in g["mean"]])
    a.axvline(y.mean(), color=CN, ls="--", lw=1)
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        a.text(m, i, f" {m:.0%} (n={n:,})", va="center", fontsize=8)
    pct_axis(a, 0, axis="x"); a.set(title=c, xlabel="proporsi gagal bayar")
save_fig(FIG_DIR, "01_categorical")
"""
)

md("**Temuan.** " + T("categorical"))

md(
    """
### 3.2 Rasio pengembalian: total pengembalian dibagi jumlah pinjaman

Pinjaman yang lancar dikembalikan hampir sebesar pokoknya. Grafik berikut melihat apa yang terjadi jika total
pengembalian jauh di atas pokok.
"""
)

code(
    """
markup = train.total_pengembalian / train.jumlah_pinjaman.replace(0, np.nan)
bins = [0, 1, 1.01, 1.03, 1.05, 1.1, 1.2, 1.5, np.inf]
mb = train.groupby(pd.cut(markup, bins, right=True, include_lowest=True), observed=True)[TARGET].agg(["mean", "size"])
fig, ax = plt.subplots(1, 2, figsize=(16, 4.6))
ax[0].bar([str(i) for i in mb.index], mb["mean"], color=C2)
for i, (m, n) in enumerate(zip(mb["mean"], mb["size"])):
    ax[0].text(i, m + 0.01, f"{m:.0%}\\nn={n:,}", ha="center", fontsize=8)
pct_axis(ax[0], 0); ax[0].set(xlabel="rasio pengembalian (total / pokok)", ylabel="proporsi gagal bayar", title="Gagal bayar per rentang rasio pengembalian")
ax[0].tick_params(axis="x", rotation=30)
for k, col, lab in [(0, C1, "lancar"), (1, C2, "gagal bayar")]:
    ax[1].hist(markup[(y == k)].clip(0.95, 2), bins=60, alpha=0.6, density=True, color=col, label=lab)
ax[1].set(yscale="log", xlabel="rasio pengembalian (dipotong di 2)", ylabel="densitas (log)", title="Distribusi rasio per kelas"); ax[1].legend()
save_fig(FIG_DIR, "02_markup")
print(f"AUC rasio pengembalian saja (baris terisi): {roc_auc_score(y[markup.notna()], markup.dropna()):.4f} | AUC jumlah pinjaman saja: {roc_auc_score(y, train.jumlah_pinjaman):.4f}")
"""
)

md("**Temuan.** " + T("markup"))

md("### 3.3 Durasi, waktu pencairan, dan porsi lender")

code(
    """
t = pd.to_datetime(train.tanggal_pencairan)
q = train.groupby(t.dt.to_period("Q"))[TARGET].agg(["mean", "size"])
fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
db = train.groupby(pd.cut(train.durasi_hari, [-2, 0, 6, 7, 14, 30, 60, 2000]), observed=True)[TARGET].agg(["mean", "size"])
ax[0].bar([str(i) for i in db.index], db["mean"], color=C1); pct_axis(ax[0], 0)
ax[0].set(xlabel="durasi (hari)", ylabel="proporsi gagal bayar", title="Gagal bayar per durasi"); ax[0].tick_params(axis="x", rotation=30)
ax2 = ax[1].twinx()
ax2.bar(q.index.astype(str), q["size"], color=CN, alpha=0.35)
ax[1].plot(q.index.astype(str), q["mean"], marker="o", color=C2); ax[1].set_zorder(ax2.get_zorder() + 1); ax[1].patch.set_visible(False)
pct_axis(ax[1], 0); ax[1].set(ylabel="proporsi gagal bayar", title="Per kuartal pencairan (batang = jumlah)"); ax[1].tick_params(axis="x", rotation=60)
ax2.set_ylabel("jumlah transaksi")
share = (train.porsi_pengembalian_lender / train.total_pengembalian.replace(0, np.nan)).round(2)
top = share.value_counts().head(8).index
sb = train[share.isin(top)].groupby(share[share.isin(top)])[TARGET].agg(["mean", "size"])
ax[2].bar(sb.index.astype(str), sb["mean"], color=C3); pct_axis(ax[2], 0)
ax[2].set(xlabel="porsi lender (8 nilai tersering)", ylabel="proporsi gagal bayar", title="Gagal bayar per porsi lender")
save_fig(FIG_DIR, "03_duration_time_share")
q.T.style.format("{:.3f}")
"""
)

md("**Temuan.** " + T("time"))

md("### 3.4 Data kosong di train dan test tidak sama")

code(
    """
mt = train.total_pengembalian.isna()
print(f"Total pengembalian kosong: train {mt.mean():.1%}, test {test.total_pengembalian.isna().mean():.1%}")
print(f"Gagal bayar saat total pengembalian kosong: {y[mt].mean():.2%} vs terisi {y[~mt].mean():.2%}")
Xb_tr, Xb_te = base_frame(train, test)
adv_X = pd.concat([Xb_tr, Xb_te], ignore_index=True)
adv_y = np.r_[np.zeros(len(train)), np.ones(len(test))]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
adv_p = cross_val_predict(lgb_model(0, n_estimators=200), adv_X, adv_y, cv=cv, method="predict_proba")[:, 1]
full = adv_X.notna().all(axis=1).values
adv_p2 = cross_val_predict(lgb_model(0, n_estimators=200), adv_X[full], adv_y[full], cv=cv, method="predict_proba")[:, 1]
print(f"Adversarial AUC train vs test: {roc_auc_score(adv_y, adv_p):.3f} | hanya baris tanpa nilai kosong: {roc_auc_score(adv_y[full], adv_p2):.3f}")
miss_tr = train[NUMERIC + ['jenis_pinjaman']].isna().any(axis=1)
print(f"Baris train dengan minimal satu nilai kosong: {miss_tr.mean():.1%} | test: {test[NUMERIC + ['jenis_pinjaman']].isna().any(axis=1).mean():.1%}")
"""
)

md("**Temuan.** " + T("missing"))

# ════════════════════════════════════════════════════════════════════ Preprocessing
md(
    """
## 4. Preprocessing dan desain validasi

Keputusan yang diambil dari EDA:

- **Nilai kosong dibiarkan** untuk ditangani LightGBM, karena polanya tidak informatif terhadap label.
- **Kategori memakai satu kamus** untuk train dan test.
- **Validasi meniru test.** Total pengembalian kosong di 10% baris train tetapi hanya 1% baris test. Model tetap dilatih
  pada semua baris, tetapi skor utama dihitung pada **baris train yang total pengembaliannya terisi**, karena itulah kondisi
  hampir semua baris test. Skor pada semua baris juga dilaporkan.
- **Ambang dipilih dari OOF.** Macro F1 butuh label 0/1, jadi ambang probabilitas dipilih per seed dari prediksi
  out-of-fold, bukan ditetapkan 0,5.
"""
)

code(
    """
SHARE = typical_lender_share(train)
TEST_LIKE = train.total_pengembalian.notna().values
print("Porsi lender yang umum per jenis pinjaman:", SHARE)
print(f"Baris 'seperti test': {TEST_LIKE.sum():,} dari {len(train):,}")
RESULTS, OOF = {}, {}

def record(name, oof):
    f_tl, thr = macro_f1_by_seed(oof, y, TEST_LIKE)
    f_all, _ = macro_f1_by_seed(oof, y)
    o = np.atleast_2d(oof)
    OOF[name] = o.mean(0)
    RESULTS[name] = {"macro F1 (seperti test)": np.mean(f_tl), "std antar seed": np.std(f_tl), "macro F1 (semua baris)": np.mean(f_all),
                     "AUC": np.mean([roc_auc_score(y, p) for p in o]), "ambang": np.mean(thr)}
    return f"{name}: macro F1 seperti test {np.mean(f_tl):.4f} ± {np.std(f_tl):.4f} | semua baris {np.mean(f_all):.4f} | ambang {np.mean(thr):.2f}"
"""
)

# ════════════════════════════════════════════════════════════════════ Baseline
md(
    """
## 5. Baseline

Semua skor adalah hasil out-of-fold dari Stratified 5-Fold yang diulang dengan 5 seed.

- **Aturan satu atribut:** gagal bayar jika status peminjam = baru atau jenis pinjaman = konsumtif.
- **Regresi logistik:** atribut mentah, one-hot untuk kategori, median untuk nilai kosong, log untuk nominal uang.
- **LightGBM:** atribut mentah tanpa fitur turunan.
"""
)

code(
    """
rule = ((train.status_peminjam == "Peminjam_Baru") | (train.jenis_pinjaman == "Konsumtif")).astype(float).values
print(record("aturan: peminjam baru atau konsumtif", np.tile(rule, (len(SEEDS), 1))))
Xlr = train[NUMERIC + CATEGORICAL].copy()
for c in ["jumlah_pinjaman", "total_pengembalian", "porsi_pengembalian_lender"]:
    Xlr[c] = np.log1p(Xlr[c].clip(lower=0))
def logreg(seed):
    pre = make_column_transformer((make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
                                  (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CATEGORICAL))
    return make_pipeline(pre, LogisticRegression(max_iter=3000, C=1.0))
print(record("regresi logistik, atribut mentah", cv_predict(logreg, Xlr, y, SEEDS)))
print(record("LightGBM, atribut mentah", cv_predict(lgb_model, Xb_tr, y, SEEDS)))
"""
)

md("**Temuan.** " + T("baseline"))

# ════════════════════════════════════════════════════════════════════ FE
md(
    """
## 6. Feature engineering dan ablation

Empat kelompok fitur turunan, masing-masing dari temuan EDA, ditambahkan satu per satu:

| Kelompok | Fitur | Dasar dari EDA |
| --- | --- | --- |
| A. rasio pengembalian | total / pokok, dan perkiraannya dari porsi lender jika total kosong | bagian 3.2 |
| B. porsi lender | porsi lender terhadap total dan terhadap pokok | bagian 3.3 |
| C. waktu pencairan | bulan ke-, hari dalam minggu | bagian 3.3 |
| D. pola data kosong | jumlah atribut kosong per baris | bagian 3.4 |
"""
)

code(
    """
ABL = []
steps = []
for g in GROUPS:
    steps.append(g)
    name = "LightGBM + " + ", ".join(s.split(".")[0] for s in steps)
    X = build(Xb_tr, train, SHARE, steps)
    print(record(name, cv_predict(lgb_model, X, y, SEEDS)))
    ABL.append((g, name, X.shape[1]))
abl = pd.DataFrame([{"langkah": "atribut mentah", "fitur": Xb_tr.shape[1], **RESULTS["LightGBM, atribut mentah"]}] +
                   [{"langkah": "+ " + g, "fitur": k, **RESULTS[n]} for g, n, k in ABL])
abl["perubahan"] = abl["macro F1 (seperti test)"].diff()
fig, ax = plt.subplots(figsize=(11, 4.3))
ax.errorbar(abl.langkah, abl["macro F1 (seperti test)"], yerr=abl["std antar seed"], marker="o", color=C2, capsize=4, label="baris seperti test")
ax.plot(abl.langkah, abl["macro F1 (semua baris)"], marker="s", color=C1, ls="--", label="semua baris")
ax.set(ylabel="macro F1 OOF", title="Ablation kumulatif kelompok fitur"); ax.legend(); ax.tick_params(axis="x", rotation=15)
save_fig(FIG_DIR, "04_ablation")
abl.style.format({c: "{:.4f}" for c in ["macro F1 (seperti test)", "std antar seed", "macro F1 (semua baris)", "AUC", "perubahan"]} | {"ambang": "{:.2f}"})
"""
)

md("**Temuan.** " + T("ablation"))

md(
    """
### 6.1 Set fitur akhir

Kelompok dipertahankan jika menaikkan macro F1 (seperti test) lebih dari satu standar deviasi antar seed dibanding langkah
sebelumnya; kelompok lain dibuang.
"""
)

code(
    """
KEEP = []
prev = RESULTS["LightGBM, atribut mentah"]["macro F1 (seperti test)"]
for g, n, _ in ABL:
    gain = RESULTS[n]["macro F1 (seperti test)"] - prev
    if gain > RESULTS[n]["std antar seed"]:
        KEEP.append(g)
    prev = RESULTS[n]["macro F1 (seperti test)"]
KEEP = KEEP or ["A. rasio pengembalian"]
FINAL_NAME = "LightGBM akhir: " + ", ".join(s.split(".")[0] for s in KEEP)
X_final = build(Xb_tr, train, SHARE, KEEP)
OOF_FINAL = cv_predict(lgb_model, X_final, y, SEEDS)
print("Kelompok dipertahankan:", KEEP)
print(record(FINAL_NAME, OOF_FINAL))
"""
)

md(
    """
### 6.2 Tanpa total pengembalian: model saat pengajuan pinjaman

Total pengembalian dan porsi lender kemungkinan baru diketahui setelah pinjaman berjalan. Untuk melihat seberapa banyak
informasi tersisa saat pengajuan, model dilatih tanpa kedua kolom itu (tetap dengan waktu pencairan).
"""
)

code(
    """
X_pre = build(Xb_tr.drop(columns=["total_pengembalian", "porsi_pengembalian_lender"]), train, SHARE, ["C. waktu pencairan"])
print(record("LightGBM tanpa total pengembalian & porsi lender", cv_predict(lgb_model, X_pre, y, SEEDS)))
"""
)

md("**Temuan.** " + T("preapproval"))

# ════════════════════════════════════════════════════════════════════ Threshold
md("## 7. Ambang keputusan untuk macro F1")

code(
    """
p_fin = OOF[FINAL_NAME]
curve_tl, curve_all = f1_curve(y[TEST_LIKE], p_fin[TEST_LIKE]), f1_curve(y, p_fin)
THR, F1_TL = best_threshold(y[TEST_LIKE], p_fin[TEST_LIKE])
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
ax[0].plot(THRESHOLDS, curve_tl, color=C2, label="baris seperti test")
ax[0].plot(THRESHOLDS, curve_all, color=C1, ls="--", label="semua baris")
ax[0].axvline(THR, color="black", lw=0.8, ls=":", label=f"ambang terpilih {THR:.2f}")
ax[0].axvline(0.5, color=CN, lw=0.8, ls="--", label="0,5")
ax[0].set(xlabel="ambang probabilitas", ylabel="macro F1 OOF", title="Macro F1 terhadap ambang"); ax[0].legend(fontsize=8)
cm = confusion_matrix(y[TEST_LIKE], p_fin[TEST_LIKE] >= THR)
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax[1], xticklabels=["prediksi lancar", "prediksi gagal"], yticklabels=["lancar", "gagal bayar"])
ax[1].set(title=f"Confusion matrix OOF (seperti test), ambang {THR:.2f}")
save_fig(FIG_DIR, "05_threshold")
print(f"Macro F1 pada ambang 0,5: {f1_score(y[TEST_LIKE], p_fin[TEST_LIKE] >= 0.5, average='macro'):.4f} | pada ambang {THR:.2f}: {F1_TL:.4f}")
print(f"F1 kelas gagal bayar: {f1_score(y[TEST_LIKE], p_fin[TEST_LIKE] >= THR):.4f} | F1 kelas lancar: {f1_score(1 - y[TEST_LIKE], p_fin[TEST_LIKE] < THR):.4f}")
"""
)

md("**Temuan.** " + T("threshold"))

# ════════════════════════════════════════════════════════════════════ Interpretation
md("## 8. Interpretasi\n\n### 8.1 Fitur terpenting")

code(
    """
m = lgb_model(0).fit(X_final, y)
imp = pd.Series(m.booster_.feature_importance("gain"), index=X_final.columns)
imp = (imp / imp.sum()).sort_values()
fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
ax[0].barh(imp.index, imp.values, color=[C2 if c in sum(GROUPS.values(), []) else C1 for c in imp.index])
pct_axis(ax[0], 0, axis="x"); ax[0].set(xlabel="porsi total gain", title="Kepentingan fitur (oranye = turunan)")
der = derived(train, SHARE)
grid = np.linspace(1.0, 1.4, 41)
probe = X_final[TEST_LIKE].sample(2000, random_state=0).copy()
pdp = []
for g in grid:
    probe["markup"] = g
    pdp.append(m.predict_proba(probe)[:, 1].mean())
ax[1].plot(grid, pdp, color=C2)
ax[1].axhline(THR, color=CN, ls=":", lw=1, label=f"ambang {THR:.2f}"); ax[1].legend()
ax[1].set(xlabel="rasio pengembalian (diset untuk semua baris)", ylabel="rata-rata peluang gagal bayar", title="Partial dependence rasio pengembalian")
save_fig(FIG_DIR, "06_importance")
"""
)

md("**Temuan.** " + T("importance"))

md("### 8.2 Di mana model salah?")

code(
    """
pred = (p_fin >= THR).astype(int)
err = pd.DataFrame({"label": y, "pred": pred, "markup": der.markup, "status": train.status_peminjam, "jenis": train.jenis_pinjaman,
                    "total kosong": ~TEST_LIKE})
err["jenis kesalahan"] = np.select([(err.label == 1) & (err.pred == 0), (err.label == 0) & (err.pred == 1)], ["gagal bayar terlewat", "alarm palsu"], "benar")
tab = err[err["jenis kesalahan"] != "benar"].groupby("jenis kesalahan").agg(jumlah=("label", "size"), median_rasio=("markup", "median"),
        porsi_total_kosong=("total kosong", "mean"), porsi_peminjam_baru=("status", lambda s: (s == "Peminjam_Baru").mean()))
fig, ax = plt.subplots(figsize=(10, 4.3))
for k, col, lab in [("benar", CN, "benar"), ("gagal bayar terlewat", C2, "gagal bayar terlewat"), ("alarm palsu", C1, "alarm palsu")]:
    s = err.markup[err["jenis kesalahan"] == k].dropna().clip(0.95, 1.6)
    ax.hist(s, bins=60, alpha=0.6, density=True, color=col, label=f"{lab} (n={len(s):,})")
ax.set(yscale="log", xlabel="rasio pengembalian", ylabel="densitas (log)", title="Rasio pengembalian menurut hasil prediksi"); ax.legend()
save_fig(FIG_DIR, "07_errors")
tab.style.format({"median_rasio": "{:.3f}", "porsi_total_kosong": "{:.0%}", "porsi_peminjam_baru": "{:.0%}"})
"""
)

md("**Temuan.** " + T("errors"))

# ════════════════════════════════════════════════════════════════════ Submission
md("## 9. Submission\n\nModel akhir dilatih ulang pada seluruh data latih dengan 5 seed, lalu probabilitas rata-ratanya diubah menjadi label dengan ambang dari bagian 7.")

code(
    """
X_test = build(Xb_te, test, SHARE, KEEP)
p_test = np.mean([lgb_model(s).fit(X_final, y).predict_proba(X_test)[:, 1] for s in SEEDS], axis=0)
sub = sample[[ID]].merge(pd.DataFrame({ID: test[ID], TARGET: (p_test >= THR).astype(int)}), on=ID, how="left")
assert sub[TARGET].notna().all() and len(sub) == len(sample)
sub.to_csv(OUT_DIR / "submission.csv", index=False)
res = pd.DataFrame(RESULTS).T.sort_values("macro F1 (seperti test)")
res.to_csv(OUT_DIR / "cv_results.csv"); abl.to_csv(OUT_DIR / "ablation.csv", index=False)
print(f"Ambang {THR:.2f} | prediksi gagal bayar di test: {sub[TARGET].mean():.2%} (train {y.mean():.2%})")
res.style.format({c: "{:.4f}" for c in res.columns if c != "ambang"} | {"ambang": "{:.2f}"})
"""
)

md("**Temuan.** " + T("submission"))

md("## 10. Kesimpulan\n\n" + T("conclusion") + "\n\n### Jejak lingkungan")

code(
    """
import platform
pkgs = ["numpy", "pandas", "scikit-learn", "lightgbm", "matplotlib", "seaborn"]
print(" | ".join([f"python {platform.python_version()}"] + [f"{p} {version(p)}" for p in pkgs]))
print(f"Waktu eksekusi total: {(time.time() - T0) / 60:.1f} menit")
"""
)

nb["cells"] = cells
nb_path = ROOT / "kaggle-notebook" / NB_NAME
nb_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, nb_path)
print("ditulis:", nb_path)

src_files = {str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT / "src").rglob("*.py"))}
runner = f'''"""Runner Kaggle untuk {NB_NAME} (di-generate oleh _build_notebook.py; jangan diedit manual)."""
import json, shutil, subprocess, sys
from pathlib import Path

SRC_FILES = {src_files!r}
NOTEBOOK = json.loads({nbf.writes(nb)!r})

proj = Path("/kaggle/working/project")
for rel, text in SRC_FILES.items():
    p = proj / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
(proj / "notebooks").mkdir(parents=True, exist_ok=True)
nb_in = proj / "notebooks" / "{NB_NAME}"
nb_in.write_text(json.dumps(NOTEBOOK))
cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", str(nb_in),
       "--output", "{NB_NAME}", "--output-dir", "/kaggle/working", "--ExecutePreprocessor.timeout=-1"]
ret = subprocess.run(cmd, cwd=proj / "notebooks")
shutil.rmtree(proj, ignore_errors=True)
sys.exit(ret.returncode)
'''
(ROOT / "kaggle").mkdir(exist_ok=True)
(ROOT / "kaggle" / "run_on_kaggle.py").write_text(runner)
print("ditulis:", ROOT / "kaggle" / "run_on_kaggle.py")
