#!/usr/bin/env python3
"""
Pipeline base de focalización territorial sin variable objetivo (unsupervised MCDA).

Uso:
  python pipeline_focalizacion.py \
    --input "/ruta/archivo.xlsx" \
    --sheet "MFT_CReg_TotMpios" \
    --outdir "./salidas"

Requiere:
  pip install pandas numpy scikit-learn openpyxl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import spearmanr


ID_COLS = [
    "Cod Depto.", "Depto.", "Codmunicipio", "NUM-Codmunicipio", "Municipio",
    "Direcciones Regionales PS ", "Distribución Regiones ComitéPS \n(Res. 3490/25 art2.2.4)"
]

# Variables Z observadas en hoja 2
Z_COLS = [
    "Z_PM_Conv", "Z_MPM", "Z_PAZ", "Z_HOGMON", "Z_IRV", "Z_TCTOT", "Z_AGFAM",
    "Z_TEJEMP", "Z_EILPO", "Z_SUPINFMULMOD", "Z_ICINV", "ZCATRUR", "Z_FIES",
    "Z_PDESN", "Z_ECOS", "Z_INDPS"
]

# Signo esperado (1 mayor = más prioridad, -1 mayor = menos prioridad)
SIGNS = {
    "Z_PM_Conv": 1,
    "Z_MPM": 1,
    "Z_PAZ": 1,
    "Z_HOGMON": 1,
    "Z_IRV": 1,
    "Z_TCTOT": 1,
    "Z_AGFAM": -1,
    "Z_TEJEMP": -1,
    "Z_EILPO": -1,
    "Z_SUPINFMULMOD": -1,
    "Z_ICINV": 1,
    "ZCATRUR": 1,
    "Z_FIES": 1,
    "Z_PDESN": 1,
    "Z_ECOS": 1,
    "Z_INDPS": -1,
}

# Pesos iniciales propuestos (normalizados)
WEIGHTS = {
    "Z_PM_Conv": 0.10,
    "Z_MPM": 0.11,
    "Z_PAZ": 0.07,
    "Z_HOGMON": 0.03,
    "Z_IRV": 0.08,
    "Z_TCTOT": 0.08,
    "Z_AGFAM": 0.05,
    "Z_TEJEMP": 0.03,
    "Z_EILPO": 0.06,
    "Z_SUPINFMULMOD": 0.09,
    "Z_ICINV": 0.08,
    "ZCATRUR": 0.07,
    "Z_FIES": 0.05,
    "Z_PDESN": 0.03,
    "Z_ECOS": 0.08,
    "Z_INDPS": 0.07,
}


def winsorize_series(s: pd.Series, p_low=0.01, p_high=0.99) -> pd.Series:
    lo, hi = s.quantile([p_low, p_high])
    return s.clip(lo, hi)


def robust_scale(s: pd.Series) -> pd.Series:
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if iqr == 0:
        return s - med
    return (s - med) / iqr


def minmax(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def build_score(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    w = normalize_weights(weights)
    work = df.copy()

    # limpieza/transformación robusta
    for c in Z_COLS:
        work[c] = pd.to_numeric(work[c], errors="coerce")
        work[c] = winsorize_series(work[c])
        work[c] = robust_scale(work[c])
        work[c] = work[c] * SIGNS[c]

    # score aditivo
    work["score_raw"] = sum(work[c] * w[c] for c in Z_COLS)
    work["score_index"] = minmax(work["score_raw"])
    work["rank"] = work["score_index"].rank(method="dense", ascending=False).astype(int)

    return work


def add_clusters(df: pd.DataFrame, k=5) -> pd.DataFrame:
    x = df[Z_COLS].fillna(df[Z_COLS].median())
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df = df.copy()
    df["cluster"] = km.fit_predict(x)
    return df


def perturb_weights(base: Dict[str, float], pct=0.2, n=200, seed=42) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    keys = list(base.keys())
    out = []
    for _ in range(n):
        f = rng.uniform(1 - pct, 1 + pct, size=len(keys))
        w = {k: base[k] * f[i] for i, k in enumerate(keys)}
        out.append(normalize_weights(w))
    return out


def topk_overlap(a: pd.Series, b: pd.Series, k=100) -> float:
    sa = set(a.nsmallest(k).index)
    sb = set(b.nsmallest(k).index)
    return len(sa.intersection(sb)) / k


def sensitivity_report(df_base: pd.DataFrame, base_weights: Dict[str, float], n=200) -> dict:
    scenarios = perturb_weights(base_weights, n=n)

    base_rank = df_base["rank"]
    sp_list, ov_list = [], []

    for w in scenarios:
        dfx = build_score(df_base, w)
        sp = spearmanr(base_rank, dfx["rank"]).statistic
        ov = topk_overlap(base_rank, dfx["rank"], k=100)
        sp_list.append(float(sp))
        ov_list.append(float(ov))

    return {
        "n_scenarios": n,
        "spearman_mean": float(np.mean(sp_list)),
        "spearman_p10": float(np.quantile(sp_list, 0.10)),
        "top100_overlap_mean": float(np.mean(ov_list)),
        "top100_overlap_p10": float(np.quantile(ov_list, 0.10)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sheet", default="MFT_CReg_TotMpios")
    ap.add_argument("--outdir", default="salidas_focalizacion")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # header real inicia en fila 13 de hoja 2
    df = pd.read_excel(args.input, sheet_name=args.sheet, header=12)

    # quitar filas vacías sin municipio
    if "Municipio" in df.columns:
        df = df[df["Municipio"].notna()].copy()

    # garantizar columnas
    missing = [c for c in Z_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas Z esperadas: {missing}")

    # modelo principal
    scored = build_score(df, WEIGHTS)
    scored = add_clusters(scored, k=6)

    # reporte sensibilidad
    report = sensitivity_report(scored, WEIGHTS, n=300)

    keep_cols = [c for c in ID_COLS if c in scored.columns] + Z_COLS + [
        "score_raw", "score_index", "rank", "cluster"
    ]
    scored[keep_cols].sort_values("rank").to_csv(outdir / "ranking_focalizacion.csv", index=False)

    # ranking por región
    reg_col = "Distribución Regiones ComitéPS \n(Res. 3490/25 art2.2.4)"
    if reg_col in scored.columns:
        scored[keep_cols].sort_values([reg_col, "rank"]).to_csv(outdir / "ranking_por_region.csv", index=False)

    with open(outdir / "reporte_sensibilidad.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Proceso completado")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
