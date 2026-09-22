from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd

from my_pipeline.nodes.nodes import get_new_base_filename, process_file, run_zone_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = PROJECT_ROOT / "data" / "01_input" / "test_data"


def test_header_fraction_does_not_create_nested_output_path() -> None:
    header = (
        "#  Sample no 5 : EastRiverSPM-1 1/16 142um 1mg; "
        "3x7 dc 0.5M HCl 60C; Pyro 15 HZ C1 1500s C4 600s"
    )

    base_name = get_new_base_filename(header, "fallback")

    assert base_name == "EastRiverSPM-1 1_16 142um"
    assert "/" not in base_name
    assert "\\" not in base_name


def test_thermogram_csv_uses_canonical_schema() -> None:
    sample_file = sorted(TEST_DATA_DIR.glob("*.txt"))[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        result = process_file(str(sample_file), tmpdir)
        base_name = result.get("basefn", sample_file.stem)
        csv_path = Path(tmpdir) / "processed_csv" / f"{base_name}_thermograms.csv"
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        expected_columns = {
            "time_s",
            "temp_raw",
            "Temperature_thermogram",
            "Temperature_Energy",
            "co2_raw",
            "co2_baseline_corrected",
            "co2_shifted",
            "flow_lance_ml_min",
            "zone",
            "co2_normalized",
        }

        assert expected_columns.issubset(set(df.columns))
    assert "temp_smoothed" not in df.columns


def test_rp_input_csv_is_written_from_ramped_slice() -> None:
    sample_file = sorted(TEST_DATA_DIR.glob("*.txt"))[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        result = process_file(str(sample_file), tmpdir)
        base_name = result.get("basefn", sample_file.stem)
        rp_path = Path(tmpdir) / "RampedPyrox_Input" / f"{base_name}_RP-Input.csv"
        assert rp_path.exists()

        df = pd.read_csv(rp_path)
        assert list(df.columns) == ["date_time", "temp", "CO2_scaled"]
        assert not df.empty
        assert df["date_time"].is_monotonic_increasing


def test_batch_summary_uses_public_header_names() -> None:
    sample_file = sorted(TEST_DATA_DIR.glob("*.txt"))[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        shutil.copy(sample_file, input_dir / sample_file.name)

        run_zone_analysis(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "batch_summary_name": "batch_summary",
            }
        )

        batch_path = output_dir / "batch_summary_batch_summary.csv"
        assert batch_path.exists()
        assert (output_dir / "batch_summary_batch_summary.xlsx").exists()
        assert (output_dir / "batch_summary_thermograms.xlsx").exists()

        df = pd.read_csv(batch_path)
        expected_columns = {
            "file",
            "area",
            "ugC/sample",
            "sample weight [mg]",
            "TOC [mgC]",
            "TOC [wt.%]",
            "Heat rate [C]",
            "C1 [s]",
            "C4 [s]",
            "mean temperature pyrolysis [C]",
            "std-div temperature pyrolysis [C]",
            "100-200C [%]",
            "200-300C [%]",
            "300-400C [%]",
            "400-500C [%]",
            "500-600C [%]",
            "600-700C [%]",
            "700-800C [%]",
            "800-900C [%]",
        }

        assert expected_columns.issubset(set(df.columns))
        time_columns = [col for col in df.columns if col.endswith("[s]")]
        assert time_columns

        with pd.ExcelFile(
            output_dir / "batch_summary_thermograms.xlsx"
        ) as thermogram_book:
            assert thermogram_book.sheet_names == [df.iloc[0]["file"]]
