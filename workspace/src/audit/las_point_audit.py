"""Phase 0 - point-level audit of raw lab LAS files (READ-ONLY, streamed in chunks).

Per file it computes:
  * value counts of classification, user_data, point_source_id, return_number,
    number_of_returns, scan_direction_flag, edge_of_flight_line, synthetic/key_point/withheld
  * intensity histogram (16-bit), bit-depth hints; scan_angle_rank histogram; Z histogram
  * GPS time range + 1-s bins -> points/s and a centroid-displacement speed proxy
  * 1 m x 1 m XY grid: occupied area, points per occupied m^2
  * "ground proxy": points within GROUND_TOL m of the lowest point in their 1 m cell
    (a crude stand-in for road/ground surface; NOT a classification)
  * nearest-neighbour spacing in one 10 m x 10 m window near the median-density ground cell
  * GeoKey directory decoded by hand (EPSG codes), since pyproj is not installed
Outputs: <out>/lab_point_stats.json, <out>/lab_point_stats.csv
Usage:   python las_point_audit.py <raw_root> <out_dir>
"""
import csv
import json
import sys
import time
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

CHUNK = 2_000_000
CELL = 1.0          # m, density grid
GROUND_TOL = 0.20   # m above cell minimum counted as ground proxy
WIN = 10.0          # m, spacing window size


def geokeys(header):
    out = {}
    for v in header.vlrs:
        if v.user_id == "LASF_Projection" and v.record_id == 34735:
            for k in v.geo_keys:
                out[int(k.id)] = int(k.value_offset) if k.tiff_tag_location == 0 else f"tag{k.tiff_tag_location}@{k.value_offset}x{k.count}"
        if v.user_id == "LASF_Projection" and v.record_id == 34737:
            out["ascii_params"] = v.strings if hasattr(v, "strings") else str(v)
    names = {1024: "GTModelType", 1025: "GTRasterType", 2048: "GeographicType", 3072: "ProjectedCSType",
             3076: "ProjLinearUnits", 4096: "VerticalCSType", 4099: "VerticalUnits", 1026: "GTCitation",
             3073: "PCSCitation", 2049: "GeogCitation"}
    return {names.get(k, k) if isinstance(k, int) else k: v for k, v in out.items()}


def vc_update(acc, arr):
    u, c = np.unique(arr, return_counts=True)
    for a, b in zip(u.tolist(), c.tolist()):
        acc[a] = acc.get(a, 0) + b


def pct_from_hist(hist, edges, qs):
    cdf = np.cumsum(hist) / max(hist.sum(), 1)
    return {f"p{q}": float(edges[min(np.searchsorted(cdf, q / 100.0), len(edges) - 1)]) for q in qs}


def audit(path: Path):
    t0 = time.time()
    with laspy.open(str(path), mode="r") as f:
        h = f.header
        mins, maxs = np.array(h.mins), np.array(h.maxs)
        nx = int(np.floor((maxs[0] - mins[0]) / CELL)) + 1
        ny = int(np.floor((maxs[1] - mins[1]) / CELL)) + 1
        n = int(h.point_count)
        gk = geokeys(h)

    cnt = np.zeros(nx * ny, np.int64)
    zmin = np.full(nx * ny, np.inf)
    vcs = {k: {} for k in ["classification", "user_data", "point_source_id", "return_number",
                           "number_of_returns", "scan_direction_flag", "edge_of_flight_line",
                           "synthetic", "key_point", "withheld"]}
    ihist = np.zeros(65536, np.int64)
    ahist = np.zeros(256, np.int64)
    zedges = np.arange(np.floor(mins[2]) - 1, np.ceil(maxs[2]) + 1.1, 0.1)
    zhist = np.zeros(len(zedges) - 1, np.int64)
    tmin, tmax, nonmono = np.inf, -np.inf, 0
    last_t = None
    sec = {}  # int second -> [count, sumx, sumy]
    first_rows = None

    # ---- pass 1
    with laspy.open(str(path), mode="r") as f:
        for pts in f.chunk_iterator(CHUNK):
            x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
            if first_rows is None:
                first_rows = [{"x": float(x[i]), "y": float(y[i]), "z": float(z[i]),
                               "intensity": int(pts.intensity[i]), "return_number": int(pts.return_number[i]),
                               "number_of_returns": int(pts.number_of_returns[i]),
                               "classification": int(pts.classification[i]),
                               "scan_angle_rank": int(pts.scan_angle_rank[i]), "user_data": int(pts.user_data[i]),
                               "point_source_id": int(pts.point_source_id[i]), "gps_time": float(pts.gps_time[i])}
                              for i in range(5)]
            ix = np.clip(((x - mins[0]) / CELL).astype(np.int64), 0, nx - 1)
            iy = np.clip(((y - mins[1]) / CELL).astype(np.int64), 0, ny - 1)
            cid = iy * nx + ix
            cnt += np.bincount(cid, minlength=nx * ny)
            np.minimum.at(zmin, cid, z)
            for k in vcs:
                vc_update(vcs[k], np.asarray(getattr(pts, k)))
            ihist += np.bincount(np.asarray(pts.intensity), minlength=65536)
            ahist += np.bincount(np.asarray(pts.scan_angle_rank).astype(np.int16) + 128, minlength=256)
            zhist += np.histogram(z, zedges)[0]
            t = np.asarray(pts.gps_time)
            tmin, tmax = min(tmin, t.min()), max(tmax, t.max())
            nonmono += int((np.diff(t) < 0).sum()) + (1 if last_t is not None and t[0] < last_t else 0)
            last_t = t[-1]
            s = np.floor(t).astype(np.int64)
            us, inv = np.unique(s, return_inverse=True)
            c = np.bincount(inv)
            sx = np.bincount(inv, weights=x)
            sy = np.bincount(inv, weights=y)
            for a, b, cx, cy in zip(us.tolist(), c.tolist(), sx.tolist(), sy.tolist()):
                e = sec.setdefault(a, [0, 0.0, 0.0])
                e[0] += b; e[1] += cx; e[2] += cy

    occ = cnt > 0
    NEAR_M = 4.0

    # trajectory proxy: polyline of 1-s point-batch centroids (NOT the GNSS trajectory)
    keys = sorted(sec)
    pps = np.array([sec[k][0] for k in keys])
    cx = np.array([sec[k][1] / sec[k][0] for k in keys]); cy = np.array([sec[k][2] / sec[k][0] for k in keys])
    consec = np.diff(keys) == 1
    disp = np.hypot(np.diff(cx), np.diff(cy))[consec]
    seg = []
    for i in range(len(cx) - 1):
        if keys[i + 1] - keys[i] != 1:
            continue
        L = np.hypot(cx[i + 1] - cx[i], cy[i + 1] - cy[i]); k = max(int(L / 0.25), 1)
        tt = np.linspace(0, 1, k, endpoint=False)
        seg.append(np.c_[cx[i] + tt * (cx[i + 1] - cx[i]), cy[i] + tt * (cy[i + 1] - cy[i])])
    poly = np.vstack(seg) if seg else None
    oi = np.nonzero(occ)[0]
    cc = np.c_[mins[0] + (oi % nx + 0.5) * CELL, mins[1] + (oi // nx + 0.5) * CELL]
    near_mask = (cKDTree(poly).query(cc)[0] <= NEAR_M) if poly is not None else np.zeros(len(oi), bool)
    near_cells = oi[near_mask]

    # spacing window: centred on the near-path cell with median point count
    cand = near_cells if len(near_cells) else oi
    med_cell = cand[np.argsort(cnt[cand])[len(cand) // 2]]
    wx0 = mins[0] + (med_cell % nx) * CELL - WIN / 2 + CELL / 2
    wy0 = mins[1] + (med_cell // nx) * CELL - WIN / 2 + CELL / 2

    # ---- pass 2: ground proxy + spacing window points
    gcnt = np.zeros(nx * ny, np.int64)
    wpts, wground = [], []
    with laspy.open(str(path), mode="r") as f:
        for pts in f.chunk_iterator(CHUNK):
            x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
            ix = np.clip(((x - mins[0]) / CELL).astype(np.int64), 0, nx - 1)
            iy = np.clip(((y - mins[1]) / CELL).astype(np.int64), 0, ny - 1)
            cid = iy * nx + ix
            g = z <= zmin[cid] + GROUND_TOL
            gcnt += np.bincount(cid[g], minlength=nx * ny)
            m = (x >= wx0) & (x < wx0 + WIN) & (y >= wy0) & (y < wy0 + WIN)
            if m.any():
                wpts.append(np.c_[x[m] - wx0, y[m] - wy0, z[m]])
                wground.append(g[m])

    def nn_stats(P):
        if len(P) < 10:
            return None
        d, _ = cKDTree(P).query(P, k=2)
        d = d[:, 1]
        return {"n": int(len(P)), "nn_mean_m": float(d.mean()), "nn_median_m": float(np.median(d)),
                "nn_p90_m": float(np.percentile(d, 90))}

    W = np.vstack(wpts) if wpts else np.zeros((0, 3))
    G = np.concatenate(wground) if wground else np.zeros(0, bool)
    occ_c, occ_g = cnt[occ], gcnt[gcnt > 0]

    # point-weighted density: density of the cell a random point falls in
    def wmedian(vals, w, q=50):
        o = np.argsort(vals); c = np.cumsum(w[o]) / w.sum()
        return float(vals[o][np.searchsorted(c, q / 100.0)])
    pw = {f"p{q}": wmedian(occ_c.astype(float), occ_c.astype(float), q) for q in (10, 50, 90)}

    near = None
    sel = gcnt[near_cells]
    sel = sel[sel > 0]
    if len(sel):
        near = {"near_m": NEAR_M, "n_cells": int(len(sel)),
                "ground_proxy_pts_per_m2_median": float(np.median(sel)),
                "ground_proxy_pts_per_m2_p10": float(np.percentile(sel, 10)),
                "ground_proxy_pts_per_m2_p90": float(np.percentile(sel, 90)),
                "all_pts_per_m2_median": float(np.median(cnt[near_cells]))}
    idx_nz = np.nonzero(ihist)[0]
    res = {
        "file": str(path), "point_count_header": n, "point_count_streamed": int(cnt.sum()),
        "geokeys": gk,
        "first_points": first_rows,
        "value_counts": {k: {str(a): b for a, b in sorted(v.items())[:60]} for k, v in vcs.items()},
        "n_unique": {k: len(v) for k, v in vcs.items()},
        "intensity": {"min": int(idx_nz.min()), "max": int(idx_nz.max()), "n_unique": int(len(idx_nz)),
                      "frac_multiple_of_256": float(ihist[::256].sum() / ihist.sum()),
                      "frac_zero": float(ihist[0] / ihist.sum()),
                      **pct_from_hist(ihist, np.arange(65536), [1, 5, 25, 50, 75, 95, 99])},
        "scan_angle_rank": {"min": int(np.nonzero(ahist)[0].min() - 128), "max": int(np.nonzero(ahist)[0].max() - 128)},
        "z": pct_from_hist(zhist, zedges[:-1], [0.1, 1, 50, 99, 99.9]),
        "gps_time": {"min": float(tmin), "max": float(tmax), "duration_s": float(tmax - tmin),
                     "n_backward_steps": nonmono, "n_seconds_with_points": len(keys),
                     "points_per_s_median": float(np.median(pps)) if len(pps) else None,
                     "points_per_s_p10": float(np.percentile(pps, 10)) if len(pps) else None,
                     "points_per_s_p90": float(np.percentile(pps, 90)) if len(pps) else None,
                     "centroid_disp_m_per_s_median": float(np.median(disp)) if len(disp) else None,
                     "centroid_disp_m_per_s_p10": float(np.percentile(disp, 10)) if len(disp) else None,
                     "centroid_disp_m_per_s_p90": float(np.percentile(disp, 90)) if len(disp) else None,
                     "centroid_path_len_m": float(disp.sum()) if len(disp) else None},
        "grid": {"cell_m": CELL, "occupied_m2": int(occ.sum()), "bbox_m2": int(nx * ny),
                 "pts_per_occupied_m2_mean": float(occ_c.mean()),
                 "pts_per_occupied_m2_median": float(np.median(occ_c)),
                 "pts_per_occupied_m2_p10": float(np.percentile(occ_c, 10)),
                 "pts_per_occupied_m2_p90": float(np.percentile(occ_c, 90)),
                 "pts_per_m2_point_weighted": pw,
                 "near_path_proxy": near,
                 "ground_tol_m": GROUND_TOL,
                 "ground_proxy_frac": float(gcnt.sum() / cnt.sum()),
                 "ground_proxy_pts_per_m2_median": float(np.median(occ_g)) if len(occ_g) else None,
                 "ground_proxy_pts_per_m2_p10": float(np.percentile(occ_g, 10)) if len(occ_g) else None,
                 "ground_proxy_pts_per_m2_p90": float(np.percentile(occ_g, 90)) if len(occ_g) else None,
                 "ground_proxy_pts_per_m2_p99": float(np.percentile(occ_g, 99)) if len(occ_g) else None},
        "spacing_window": {"x0": float(wx0), "y0": float(wy0), "size_m": WIN,
                           "all": nn_stats(W), "ground_proxy": nn_stats(W[G]) if len(W) else None},
        "seconds_elapsed": round(time.time() - t0, 1),
    }
    return res


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("*.las"))
    if len(sys.argv) > 3:
        files = [p for p in files if sys.argv[3] in str(p)]
    rows = []
    for p in files:
        r = audit(p)
        rows.append(r)
        print(json.dumps(r)[:4000], flush=True)
    (out / "lab_point_stats.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    flat_keys = ["file", "point_count_streamed"]
    with open(out / "lab_point_stats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        hdr = flat_keys + ["class_values", "intensity_min", "intensity_max", "intensity_p50", "gps_duration_s",
                           "pts_per_s_median", "speed_proxy_mps_median", "occupied_m2",
                           "pts_per_occ_m2_median", "pts_per_m2_point_weighted_median",
                           "ground_proxy_pts_per_m2_median", "near_path_ground_pts_per_m2_median",
                           "nn_spacing_ground_median_m"]
        w.writerow(hdr)
        for r in rows:
            sw = r["spacing_window"]["ground_proxy"] or {}
            w.writerow([r["file"], r["point_count_streamed"], json.dumps(r["value_counts"]["classification"]),
                        r["intensity"]["min"], r["intensity"]["max"], r["intensity"]["p50"],
                        r["gps_time"]["duration_s"], r["gps_time"]["points_per_s_median"],
                        r["gps_time"]["centroid_disp_m_per_s_median"], r["grid"]["occupied_m2"],
                        r["grid"]["pts_per_occupied_m2_median"], r["grid"]["pts_per_m2_point_weighted"]["p50"],
                        r["grid"]["ground_proxy_pts_per_m2_median"],
                        (r["grid"]["near_path_proxy"] or {}).get("ground_proxy_pts_per_m2_median"),
                        sw.get("nn_median_m")])


if __name__ == "__main__":
    main()
