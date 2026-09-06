"""Reading uploads: repeated reads, and Excel as well as CSV.

Regression cover for a bug that only appeared with uploaded files, never with
the bundled samples. The Upload step reads each upload more than once per
interaction -- to look for repeated timestamps, and again when Load is pressed
-- and pandas leaves a file object at EOF. The second read then saw an empty
stream and reported "No columns to parse from file", which names the symptom
and not the cause. Reading from a path, as the sample data does, never hits it.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from modules.data_loader import (
    EXCEL_SUFFIXES,
    is_excel_source,
    list_excel_sheets,
    load_and_validate_dataset,
    load_csv,
    read_tabular,
    source_name,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="h"),
        "pm25": [10.0, 11, 12, 13, 14],
    })


class _Upload(io.BytesIO):
    """Stands in for Streamlit's UploadedFile: a named, seekable stream."""

    def __init__(self, payload: bytes, name: str):
        super().__init__(payload)
        self.name = name


@pytest.fixture
def csv_upload(frame) -> _Upload:
    return _Upload(frame.to_csv(index=False).encode("utf-8"), "reference.csv")


@pytest.fixture
def excel_upload(frame) -> _Upload:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="measurements", index=False)
        pd.DataFrame({"item": ["rows"], "value": [len(frame)]}).to_excel(
            writer, sheet_name="summary", index=False
        )
    return _Upload(buffer.getvalue(), "reference.xlsx")


class TestRepeatedReads:
    def test_a_csv_upload_survives_being_read_twice(self, csv_upload):
        first = read_tabular(csv_upload)
        second = read_tabular(csv_upload)
        assert first.equals(second)
        assert len(second) == 5

    def test_an_excel_upload_survives_being_read_twice(self, excel_upload):
        first = read_tabular(excel_upload, sheet_name="measurements")
        second = read_tabular(excel_upload, sheet_name="measurements")
        assert first.equals(second)

    def test_listing_sheets_does_not_consume_the_stream(self, excel_upload):
        assert list_excel_sheets(excel_upload) == ["measurements", "summary"]
        assert len(read_tabular(excel_upload, sheet_name="measurements")) == 5

    def test_the_full_loader_can_follow_a_preview_read(self, csv_upload):
        """The exact sequence the Upload step performs."""
        read_tabular(csv_upload)  # duplicate-timestamp preview
        loaded = load_and_validate_dataset(csv_upload, "timestamp", "Reference", "UTC")
        assert len(loaded) == 5

    def test_plain_pandas_would_have_failed(self, csv_upload):
        """Shows the failure the rewind prevents, so the fix cannot quietly regress."""
        pd.read_csv(csv_upload)
        with pytest.raises(pd.errors.EmptyDataError):
            pd.read_csv(csv_upload)


class TestExcelSupport:
    def test_a_workbook_reads(self, excel_upload):
        assert read_tabular(excel_upload, sheet_name="measurements").shape == (5, 2)

    def test_the_named_sheet_is_the_one_returned(self, excel_upload):
        summary = read_tabular(excel_upload, sheet_name="summary")
        assert list(summary.columns) == ["item", "value"]

    def test_without_a_name_the_first_sheet_is_used(self, excel_upload):
        assert list(read_tabular(excel_upload).columns) == ["timestamp", "pm25"]

    def test_a_workbook_goes_through_the_full_loader(self, excel_upload):
        loaded = load_and_validate_dataset(
            excel_upload, "timestamp", "Reference", "UTC", sheet_name="measurements"
        )
        assert len(loaded) == 5

    def test_csv_has_no_sheets_to_choose_from(self, csv_upload):
        assert list_excel_sheets(csv_upload) == []

    @pytest.mark.parametrize("name,expected", [
        ("a.xlsx", True), ("a.XLSX", True), ("a.xlsm", True), ("a.xls", True),
        ("a.csv", False), ("a.txt", False), ("no_extension", False),
    ])
    def test_format_is_decided_by_extension(self, name, expected):
        assert is_excel_source(_Upload(b"", name)) is expected

    def test_every_excel_suffix_is_recognised(self):
        for suffix in EXCEL_SUFFIXES:
            assert is_excel_source(_Upload(b"", f"file{suffix}"))

    def test_source_name_handles_paths_and_uploads(self, tmp_path, csv_upload):
        assert source_name(tmp_path / "x.csv").endswith("x.csv")
        assert source_name(csv_upload) == "reference.csv"
        assert source_name(pd.DataFrame()) == ""


class TestBackwardCompatibility:
    def test_load_csv_still_works(self, csv_upload):
        assert len(load_csv(csv_upload)) == 5

    def test_a_dataframe_is_copied_not_aliased(self, frame):
        out = read_tabular(frame)
        out.loc[0, "pm25"] = 999.0
        assert frame.loc[0, "pm25"] == 10.0

    def test_paths_still_read(self, reference_csv):
        assert not read_tabular(reference_csv).empty
