# my-pipeline

SoliTOC processing pipeline with two execution paths:

1. GUI app for interactive preview and batch processing.
2. Kedro pipeline for repeatable batch runs.

## Current Workflow

1. Parse TXT file data and header metadata.
2. Extract program metadata from header and B-line markers:
	- Pyro heat rate (degC/min)
	- C1 (s)
	- C4 (s)
	- B-line D3-like reference time (s)
3. Compute baseline (manual, fixed interval, auto, or fixed value).
4. Baseline-correct CO2.
5. Apply CO2 lag shift (CO2 earlier by shift points).
6. Build temperature profiles linked to time:
	- Temperature_thermogram
	- Temperature_Energy
7. Detect zone boundaries:
	- Start-Run
	- Start-Ramp (from B-line D3-like marker)
	- Start-Plateau (Start-Oxidation - C4)
	- Start-Oxidation (flow-lance rise)
	- End-Run
8. Integrate CO2 from Start-Ramp onward and compute output metrics.
9. Write per-file thermogram CSV + RP-input CSV + summary text, plus batch summary CSV/XLSX and thermogram workbook.

## Canonical Output Schema

Thermogram CSV columns:

1. time_s
2. temp_raw
3. Temperature_thermogram
4. Temperature_Energy
5. co2_raw
6. co2_baseline_corrected
7. co2_shifted
8. flow_lance_ml_min
9. pyro_heat_rate_c_per_min
10. c1_s
11. c4_s
12. zone
13. co2_normalized

Batch summary CSV fields include at least:

1. file
2. area
3. ugC/sample
4. sample weight [mg]
5. TOC [mgC]
6. TOC [wt.%]
7. Heat rate [C]
8. C1 [s]
9. C4 [s]
10. mean temperature pyrolysis [C]
11. std-div temperature pyrolysis [C]
12. Start-Run [s]
13. Start-Ramp [s]
14. Start-Plateau [s]
15. Start-Oxidation [s]
16. End-Run [s]

## Preview Behavior

Baseline preview shows zone markers in CO2-time visualization space:

1. Start-Run and End-Run are not shifted.
2. Internal markers are shown with +shift points.

## Removed / Deprecated Behavior

1. Temperature-forward shift option in GUI is removed.
2. Temperature-shifted output column is removed.
3. Plateau-duration tuning parameter is removed from active workflow.
4. Legacy temperature-plateau helpers for zone-start detection are removed from active code.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run GUI:

```bash
python app.py
```

Run Kedro pipeline:

```bash
kedro run
```

Run tests:

```bash
pytest
```
