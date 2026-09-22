from __future__ import annotations

from pathlib import Path

from my_pipeline.nodes.nodes import (
    detect_zone_boundaries,
    parse_program_metadata,
    parse_solitoc_txt_to_df,
)

TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "01_input" / "test_data"


def test_zone_boundaries_are_ordered_for_sample_files() -> None:
    for path in sorted(TEST_DATA_DIR.glob("*.txt")):
        _, df = parse_solitoc_txt_to_df(str(path))
        meta = parse_program_metadata(str(path))
        boundaries = detect_zone_boundaries(
            df.iloc[:, 0].to_numpy(),
            df.iloc[:, 1].to_numpy(),
            df.iloc[:, 5].to_numpy(),
            ramp_reference_time=meta.get("b_d3_time_s"),
            c4_seconds=meta.get("c4_s"),
        )["times"]
        ordered = [
            boundaries["Start-Run"],
            boundaries["Start-Ramp"],
            boundaries["Start-Plateau"],
            boundaries["Start-Oxidation"],
            boundaries["End-Run"],
        ]
        assert ordered == sorted(ordered)


def test_zone_boundaries_follow_metadata_rules() -> None:
    for path in sorted(TEST_DATA_DIR.glob("*.txt")):
        _, df = parse_solitoc_txt_to_df(str(path))
        meta = parse_program_metadata(str(path))
        boundaries = detect_zone_boundaries(
            df.iloc[:, 0].to_numpy(),
            df.iloc[:, 1].to_numpy(),
            df.iloc[:, 5].to_numpy(),
            ramp_reference_time=meta.get("b_d3_time_s"),
            c4_seconds=meta.get("c4_s"),
        )["times"]

        if meta.get("b_d3_time_s") is not None:
            assert abs(boundaries["Start-Ramp"] - float(meta["b_d3_time_s"])) <= 1.0
        if meta.get("c4_s") is not None:
            delta = boundaries["Start-Oxidation"] - boundaries["Start-Plateau"]
            assert abs(delta - float(meta["c4_s"])) <= 1.0
        assert boundaries["Start-Oxidation"] > boundaries["Start-Plateau"]
