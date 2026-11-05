"""Phase 0 - header-only audit of raw lab LAS files (READ-ONLY).

Reads only LAS headers + VLRs (no point records). Writes JSON + CSV to results/.
Usage: python las_header_audit.py <raw_root> <out_dir>
"""
import csv
import json
import sys
from pathlib import Path

import laspy


def audit(path: Path) -> dict:
    with laspy.open(str(path), mode="r") as f:  # read-only handle
        h = f.header
        vlrs = []
        crs_wkt = None
        for v in list(h.vlrs) + list(getattr(h, "evlrs", None) or []):
            entry = {"user_id": v.user_id, "record_id": v.record_id,
                     "description": v.description}
            if "WktCoordinateSystem" in type(v).__name__:
                crs_wkt = v.string
            vlrs.append(entry)
        try:
            crs = h.parse_crs()
            crs_name = crs.name if crs is not None else None
            epsg = crs.to_epsg() if crs is not None else None
        except Exception as e:  # pyproj may be missing
            crs_name, epsg = f"parse_crs failed: {type(e).__name__}", None
        dims = list(h.point_format.dimension_names)
        std = set(h.point_format.standard_dimension_names)
        extra = [d for d in dims if d not in std]
        return {
            "file": str(path),
            "size_bytes": path.stat().st_size,
            "las_version": f"{h.version.major}.{h.version.minor}",
            "point_format_id": h.point_format.id,
            "point_record_length": h.point_format.size,
            "point_count": int(h.point_count),
            "points_by_return": [int(x) for x in h.number_of_points_by_return],
            "scale": [float(x) for x in h.scales],
            "offset": [float(x) for x in h.offsets],
            "min": [float(x) for x in h.mins],
            "max": [float(x) for x in h.maxs],
            "extent_m": [float(b - a) for a, b in zip(h.mins, h.maxs)],
            "dimensions": dims,
            "extra_dimensions": extra,
            "system_identifier": h.system_identifier,
            "generating_software": h.generating_software,
            "creation_date": str(h.creation_date),
            "global_encoding": int(h.global_encoding.value) if hasattr(h.global_encoding, "value") else str(h.global_encoding),
            "gps_time_type": str(getattr(h.global_encoding, "gps_time_type", "")),
            "vlrs": vlrs,
            "crs_name": crs_name,
            "crs_epsg": epsg,
            "crs_wkt_head": (crs_wkt or "")[:300],
        }


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("*.las")) + sorted(root.rglob("*.laz"))
    rows = [audit(p) for p in files]
    (out / "lab_las_headers.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    keys = ["file", "size_bytes", "las_version", "point_format_id", "point_record_length",
            "point_count", "min", "max", "extent_m", "scale", "offset", "extra_dimensions",
            "crs_name", "crs_epsg", "generating_software", "creation_date"]
    with open(out / "lab_las_headers.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow([json.dumps(r[k]) if isinstance(r[k], list) else r[k] for k in keys])
    for r in rows:
        print(json.dumps({k: r[k] for k in r if k not in ("vlrs", "crs_wkt_head")}))
        print("  VLRs:", [(v["user_id"], v["record_id"], v["description"]) for v in r["vlrs"]])
        print("  WKT:", r["crs_wkt_head"][:200])


if __name__ == "__main__":
    main()
