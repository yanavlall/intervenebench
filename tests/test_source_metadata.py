from __future__ import annotations

from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from intervenebench.source_metadata import (
    require_result_free_workbook,
    xlsx_sheet_names,
)


def _workbook(tmp_path, names: tuple[str, ...]):
    path = tmp_path / "source.xlsx"
    sheets = "".join(
        f'<sheet name="{name}" sheetId="{index}"/>'
        for index, name in enumerate(names, start=1)
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/'
        f'spreadsheetml/2006/main"><sheets>{sheets}</sheets></workbook>'
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", xml)
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            "<worksheet>this cell content must never be read</worksheet>",
        )
    return path


def test_xlsx_sheet_names_reads_only_container_metadata(tmp_path) -> None:
    path = _workbook(tmp_path, ("Codebook", "Treatment Assignment"))
    assert xlsx_sheet_names(path) == ("Codebook", "Treatment Assignment")
    assert require_result_free_workbook(path) == (
        "Codebook",
        "Treatment Assignment",
    )


def test_result_bearing_workbook_fails_before_worksheet_inspection(tmp_path) -> None:
    path = _workbook(tmp_path, ("Codebook", "Unweighted Frequencies"))
    with pytest.raises(ValueError, match="result-bearing"):
        require_result_free_workbook(path)
