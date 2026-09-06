"""Build a CaliSenseAQ-ready merged 2025 reference/LCS dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
from xml.sax.saxutils import escape
import math
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed_2025_trends"
MASTER_FILE = DATA_DIR / (
    "ambient-CAC-monitoring_automated_processing_v2_hourly_2025-09-06_"
    "analysis_master_000000000000.csv"
)
REFERENCE_SOURCE_NAME = MASTER_FILE.name
SENSOR_FILES = [
    DATA_DIR / "Jan25_Hourly.csv",
    DATA_DIR / "Feb25_hourly.csv",
    DATA_DIR / "March_25_Hourly.csv",
]

OUTPUT_CSV = OUTPUT_DIR / "CaliSenseAQ_2025_reference_lcs_hourly_merged.csv"
OUTPUT_RECOMMENDED_CSV = OUTPUT_DIR / "CaliSenseAQ_2025_reference_lcs_hourly_merged_recommended.csv"
OUTPUT_XLSX = OUTPUT_DIR / "CaliSenseAQ_2025_reference_lcs_hourly_merged.xlsx"
OUTPUT_MAPPING_CSV = OUTPUT_DIR / "CaliSenseAQ_2025_device_site_mapping.csv"


REFERENCE_RENAME = {
    "site_name": "reference_site_name",
    "pm2_5": "reference_pm25",
    "co": "reference_co",
    "temp_c": "reference_temperature",
    "rh": "reference_humidity",
    "bp": "reference_pressure",
    "ws": "reference_wind_speed",
    "wd": "reference_wind_direction",
}

SENSOR_RENAME = {
    "devId": "sensor_device_id",
    "PM2.5": "sensor_pm25",
    "PM10": "sensor_pm10",
    "PM1": "sensor_pm1",
    "Temperature": "sensor_temperature",
    "Humidity": "sensor_humidity",
    "Pressure": "sensor_pressure",
    "CO2": "sensor_co2",
    "CO": "sensor_co",
    "NO2": "sensor_no2",
    "SO2": "sensor_so2",
    "O3": "sensor_o3",
    "TVOC": "sensor_tvoc",
}


def parse_sensor_timestamp(dataframe: pd.DataFrame) -> pd.Series:
    date_text = dataframe["date"].astype(str).str.strip()
    time_text = dataframe["time"].astype(str).str.strip()
    raw = date_text + " " + time_text
    year_first = date_text.str.match(r"^\d{4}[-/]")

    timestamp = pd.Series(pd.NaT, index=dataframe.index, dtype="datetime64[ns]")
    if year_first.any():
        timestamp.loc[year_first] = pd.to_datetime(raw.loc[year_first], errors="coerce")
    if (~year_first).any():
        timestamp.loc[~year_first] = pd.to_datetime(raw.loc[~year_first], errors="coerce", dayfirst=True)
    return timestamp.dt.floor("h")


def confidence_from_corr(correlation: float) -> str:
    if pd.isna(correlation):
        return "unmapped"
    if correlation >= 0.65:
        return "high"
    if correlation >= 0.50:
        return "medium"
    return "low"


def load_reference_2025() -> pd.DataFrame:
    reference = pd.read_csv(MASTER_FILE)
    reference["timestamp"] = pd.to_datetime(reference["hour_ending_IST"], errors="coerce").dt.floor("h")
    reference = reference[reference["timestamp"].dt.year == 2025].copy()
    keep = ["timestamp", *REFERENCE_RENAME.keys()]
    reference = reference[[column for column in keep if column in reference.columns]].copy()
    for column in reference.columns:
        if column not in {"timestamp", "site_name"}:
            reference[column] = pd.to_numeric(reference[column], errors="coerce")
    reference = (
        reference.groupby(["site_name", "timestamp"], as_index=False)
        .mean(numeric_only=True)
        .rename(columns=REFERENCE_RENAME)
        .sort_values(["reference_site_name", "timestamp"])
    )
    reference["reference_source_file"] = REFERENCE_SOURCE_NAME
    return reference


def load_sensor_2025() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for sensor_file in SENSOR_FILES:
        raw = pd.read_csv(sensor_file)
        raw["timestamp"] = parse_sensor_timestamp(raw)
        raw = raw[raw["timestamp"].dt.year == 2025].copy()
        raw["sensor_source_file"] = sensor_file.name
        keep = ["timestamp", "sensor_source_file", *SENSOR_RENAME.keys()]
        raw = raw[[column for column in keep if column in raw.columns]].copy()
        for column in raw.columns:
            if column not in {"timestamp", "sensor_source_file", "devId"}:
                raw[column] = pd.to_numeric(raw[column], errors="coerce")
        numeric_cols = [
            column
            for column in raw.columns
            if column not in {"timestamp", "sensor_source_file", "devId"}
        ]
        grouped = (
            raw.groupby(["sensor_source_file", "devId", "timestamp"], as_index=False)[numeric_cols]
            .mean(numeric_only=True)
            .rename(columns=SENSOR_RENAME)
            .sort_values(["sensor_device_id", "timestamp"])
        )
        parts.append(grouped)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def infer_device_site_mapping(reference: pd.DataFrame, sensor: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference_pm = reference[["reference_site_name", "timestamp", "reference_pm25"]].copy()
    sensor_pm = sensor[["sensor_device_id", "timestamp", "sensor_pm25"]].copy()
    for device_id, device_df in sensor_pm.groupby("sensor_device_id"):
        for site_name, site_df in reference_pm.groupby("reference_site_name"):
            joined = device_df.merge(site_df, on="timestamp", how="inner").dropna(
                subset=["sensor_pm25", "reference_pm25"]
            )
            corr = joined["sensor_pm25"].corr(joined["reference_pm25"]) if len(joined) >= 10 else float("nan")
            rows.append(
                {
                    "sensor_device_id": device_id,
                    "reference_site_name": site_name,
                    "common_pm25_pairs": len(joined),
                    "pm25_correlation": corr,
                    "sensor_pm25_mean": joined["sensor_pm25"].mean() if len(joined) else None,
                    "reference_pm25_mean": joined["reference_pm25"].mean() if len(joined) else None,
                }
            )

    all_pairs = pd.DataFrame(rows).sort_values(
        ["sensor_device_id", "pm25_correlation"],
        ascending=[True, False],
    )
    best = all_pairs.groupby("sensor_device_id", as_index=False).head(1).copy()
    best["mapping_method"] = "inferred_best_pm25_correlation"
    best["mapping_confidence"] = best["pm25_correlation"].apply(confidence_from_corr)
    best["mapping_note"] = (
        "Verify against the official deployment log before final calibration. "
        "Low-confidence mappings are retained for completeness."
    )
    return best.reset_index(drop=True), all_pairs.reset_index(drop=True)


def build_merged(reference: pd.DataFrame, sensor: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    sensor_mapped = sensor.merge(
        mapping[
            [
                "sensor_device_id",
                "reference_site_name",
                "pm25_correlation",
                "common_pm25_pairs",
                "mapping_method",
                "mapping_confidence",
            ]
        ],
        on="sensor_device_id",
        how="left",
    )
    merged = sensor_mapped.merge(
        reference,
        on=["reference_site_name", "timestamp"],
        how="inner",
    )

    difference_pairs = [
        ("pm25", "sensor_pm25", "reference_pm25"),
        ("co", "sensor_co", "reference_co"),
        ("temperature", "sensor_temperature", "reference_temperature"),
        ("humidity", "sensor_humidity", "reference_humidity"),
        ("pressure", "sensor_pressure", "reference_pressure"),
    ]
    for label, sensor_col, reference_col in difference_pairs:
        if sensor_col in merged.columns and reference_col in merged.columns:
            merged[f"{label}_sensor_minus_reference"] = merged[sensor_col] - merged[reference_col]

    ordered = [
        "timestamp",
        "reference_site_name",
        "sensor_device_id",
        "sensor_source_file",
        "reference_source_file",
        "mapping_method",
        "mapping_confidence",
        "pm25_correlation",
        "common_pm25_pairs",
        "reference_pm25",
        "sensor_pm25",
        "pm25_sensor_minus_reference",
        "reference_co",
        "sensor_co",
        "co_sensor_minus_reference",
        "reference_temperature",
        "sensor_temperature",
        "temperature_sensor_minus_reference",
        "reference_humidity",
        "sensor_humidity",
        "humidity_sensor_minus_reference",
        "reference_pressure",
        "sensor_pressure",
        "pressure_sensor_minus_reference",
        "reference_wind_speed",
        "reference_wind_direction",
        "sensor_pm10",
        "sensor_pm1",
        "sensor_no2",
        "sensor_so2",
        "sensor_o3",
        "sensor_tvoc",
        "sensor_co2",
    ]
    ordered = [column for column in ordered if column in merged.columns]
    extras = [column for column in merged.columns if column not in ordered]
    return merged[ordered + extras].sort_values(["timestamp", "reference_site_name", "sensor_device_id"])


def column_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def cell_xml(row_idx: int, col_idx: int, value: Any) -> str:
    ref = f"{column_name(col_idx)}{row_idx}"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        text = value.strftime("%Y-%m-%d %H:%M:%S")
        return f'<c r="{ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return f'<c r="{ref}"><v>{value}</v></c>'
        return ""
    text = str(value)
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'


def sheet_xml(dataframe: pd.DataFrame) -> str:
    rows: list[str] = []
    headers = list(dataframe.columns)
    rows.append(
        '<row r="1">' + "".join(cell_xml(1, col_idx, header) for col_idx, header in enumerate(headers)) + "</row>"
    )
    for row_idx, (_, row) in enumerate(dataframe.iterrows(), start=2):
        cells = "".join(cell_xml(row_idx, col_idx, row[column]) for col_idx, column in enumerate(headers))
        rows.append(f'<row r="{row_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(rows)}</sheetData>"
        "</worksheet>"
    )


def write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    sheet_items = list(sheets.items())
    workbook_sheets = []
    workbook_rels = []
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for idx, (sheet_name, _) in enumerate(sheet_items, start=1):
        safe_name = sheet_name[:31].replace("[", "(").replace("]", ")")
        workbook_sheets.append(f'<sheet name="{escape(safe_name)}" sheetId="{idx}" r:id="rId{idx}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        )
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    workbook_rels.append(
        f'<Relationship Id="rId{len(sheet_items) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f"{''.join(overrides)}</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{''.join(workbook_sheets)}</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{''.join(workbook_rels)}</Relationships>",
        )
        archive.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<fonts count=\"1\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts>"
            "<fills count=\"1\"><fill><patternFill patternType=\"none\"/></fill></fills>"
            "<borders count=\"1\"><border/></borders>"
            "<cellStyleXfs count=\"1\"><xf/></cellStyleXfs>"
            "<cellXfs count=\"1\"><xf xfId=\"0\"/></cellXfs>"
            "</styleSheet>",
        )
        for idx, (_, dataframe) in enumerate(sheet_items, start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml(dataframe))


def _require_inputs() -> None:
    """Fail with the missing filenames rather than a bare FileNotFoundError.

    These inputs live under a gitignored data/ directory, so anyone but the
    original author starts without them. Say which files are expected and
    where, instead of dying on whichever one pandas reached first.
    """
    expected = [MASTER_FILE, *SENSOR_FILES]
    missing = [path for path in expected if not path.exists()]
    if not missing:
        return

    lines = [
        "This script builds the 2025 co-location dataset from raw exports that are",
        "not in the repository (data/ is gitignored).",
        f"Expected under {DATA_DIR}:",
        *[f"  - {path.name}" for path in expected],
        "Missing:",
        *[f"  - {path.name}" for path in missing],
        "Pass --data-dir to point at another location.",
    ]
    raise SystemExit("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Directory holding the raw reference and sensor exports.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory to write the merged dataset into.",
    )
    return parser.parse_args()


def main() -> None:
    global DATA_DIR, OUTPUT_DIR, MASTER_FILE, SENSOR_FILES
    args = _parse_args()
    if args.data_dir != DATA_DIR:
        DATA_DIR = args.data_dir
        MASTER_FILE = DATA_DIR / MASTER_FILE.name
        SENSOR_FILES = [DATA_DIR / path.name for path in SENSOR_FILES]
    OUTPUT_DIR = args.output_dir
    _require_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = load_reference_2025()
    sensor = load_sensor_2025()
    best_mapping, all_mapping_pairs = infer_device_site_mapping(reference, sensor)
    merged = build_merged(reference, sensor, best_mapping)
    recommended = merged[merged["mapping_confidence"].isin(["high", "medium"])].copy()

    merged.to_csv(OUTPUT_CSV, index=False)
    recommended.to_csv(OUTPUT_RECOMMENDED_CSV, index=False)
    best_mapping.to_csv(OUTPUT_MAPPING_CSV, index=False)

    workbook_summary = pd.DataFrame(
        [
            {
                "item": "merged_rows",
                "value": len(merged),
            },
            {
                "item": "recommended_rows_high_medium_mapping",
                "value": len(recommended),
            },
            {
                "item": "date_range",
                "value": f"{merged['timestamp'].min()} to {merged['timestamp'].max()}",
            },
            {
                "item": "source_files_used",
                "value": ", ".join(path.name for path in SENSOR_FILES),
            },
            {
                "item": "source_files_excluded",
                "value": "All sensors_hourly_Aug-Nov2025.csv, All sensors_hourly_Dec2025.csv",
            },
            {
                "item": "exclusion_reason",
                "value": "Their internal timestamps are 2024, not 2025.",
            },
            {
                "item": "mapping_warning",
                "value": "Device-to-site mapping is inferred from PM2.5 correlation. Verify with official deployment records before final calibration.",
            },
        ]
    )

    write_xlsx(
        OUTPUT_XLSX,
        {
            "merged": merged,
            "recommended": recommended,
            "device_site_mapping": best_mapping,
            "all_mapping_correlations": all_mapping_pairs,
            "summary": workbook_summary,
        },
    )

    print("Done")
    print(f"Merged CSV: {OUTPUT_CSV}")
    print(f"Recommended merged CSV: {OUTPUT_RECOMMENDED_CSV}")
    print(f"Merged Excel: {OUTPUT_XLSX}")
    print(f"Mapping CSV: {OUTPUT_MAPPING_CSV}")
    print(f"Merged rows: {len(merged)}")
    print(f"Recommended rows: {len(recommended)}")
    print(f"Date range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")
    print("Device-site mapping:")
    print(
        best_mapping[
            [
                "sensor_device_id",
                "reference_site_name",
                "common_pm25_pairs",
                "pm25_correlation",
                "mapping_confidence",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
