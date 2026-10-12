"""Create a small sample TXT (and optional DOCX) for local smoke testing."""

from __future__ import annotations

from pathlib import Path

SAMPLE = """# Solar Energy Storage Basics

## Overview
Solar energy storage systems capture electricity produced by photovoltaic panels
and release it later when generation is low. The most common residential option
is a lithium-ion battery bank paired with a hybrid inverter.

## Capacity planning
Battery capacity is measured in kilowatt-hours (kWh). A typical household that
consumes 20 kWh per day may target 10 to 15 kWh of usable storage for evening
coverage. Depth of discharge, round-trip efficiency, and temperature all affect
usable capacity.

## Safety notes
Installations must follow local electrical codes. Battery rooms need ventilation
and thermal monitoring. Never mix incompatible chemistries in the same bank.

## Maintenance
Inspect connections quarterly. Firmware updates for inverters should be applied
only from the manufacturer channel. Replace cells that show abnormal voltage
drift during monthly health checks.
"""


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    txt_path = out_dir / "sample_knowledge.txt"
    txt_path.write_text(SAMPLE.strip() + "\n", encoding="utf-8")
    print(f"Wrote {txt_path}")

    try:
        from docx import Document

        doc = Document()
        for block in SAMPLE.strip().split("\n\n"):
            doc.add_paragraph(block)
        docx_path = out_dir / "sample_knowledge.docx"
        doc.save(docx_path)
        print(f"Wrote {docx_path}")
    except ImportError:
        print("python-docx not installed; skipped DOCX sample.")


if __name__ == "__main__":
    main()
