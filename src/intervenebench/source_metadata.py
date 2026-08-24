"""Fail-closed source-document metadata inspection.

Some files labelled as codebooks also contain outcome frequencies or result
summaries.  Sheet names are inspected from the XLSX container before any cell
content is eligible to be read.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile


FORBIDDEN_RESULT_SHEET_TOKENS = frozenset(
    {
        "frequency",
        "frequencies",
        "result",
        "results",
        "arm mean",
        "treatment effect",
        "significance",
        "winner",
    }
)


def xlsx_sheet_names(path: Path) -> tuple[str, ...]:
    """Return workbook sheet names without reading worksheet cell XML."""

    try:
        with ZipFile(path) as workbook:
            xml = workbook.read("xl/workbook.xml")
    except (BadZipFile, KeyError) as error:
        raise ValueError("path is not a readable XLSX workbook") from error
    root = ElementTree.fromstring(xml)
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    names = tuple(
        sheet.attrib["name"]
        for sheet in root.findall("main:sheets/main:sheet", namespace)
    )
    if not names or len(names) != len(set(names)):
        raise ValueError("workbook must declare unique sheet names")
    return names


def require_result_free_workbook(path: Path) -> tuple[str, ...]:
    """Reject a workbook container whose sheet names signal outcome content."""

    names = xlsx_sheet_names(path)
    unsafe = tuple(
        name
        for name in names
        if any(token in name.casefold() for token in FORBIDDEN_RESULT_SHEET_TOKENS)
    )
    if unsafe:
        raise ValueError(
            "workbook contains result-bearing sheets and must not be opened in a "
            f"sealed audit: {list(unsafe)}"
        )
    return names
