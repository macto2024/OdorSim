# Excel-calibrated MOX sensors

## Runtime array

Sensor parameters are generated into
`odor_sim/config/mox_sensors.yaml`. OdorSim supports 17 nominal sensor
families from the supplied calibration workbooks:

`TGS2620`, `TGS2600`, `TGS2611`, `TGS2610`, `TGS2602`, `101`,
`203`, `302`, `401`, `509`, `104`, `303`, `500-1`,
`43_SINTERED`, `43_UNSINTERED`, `601`, and `701`.

The default OVLA array is fixed, ordered, and eight channels wide:

```text
101, 203, 302, 401, TGS2600, TGS2602, TGS2610, TGS2611
```

Omitting `mox_model` selects those eight sensors. Passing `"all"` selects
all 17 supported models. Episode metadata records `mox_models` in column
order; loaders must validate that order rather than assuming arbitrary file
columns are interchangeable.

## Response model

For each physical channel, gas, and concentration, calibration extracts a late
stable air resistance and a local gas-minimum resistance. It calculates
`Ra/Rg`, takes the median across valid physical channels at each
concentration, then fits one nominal sensor curve:

```text
Ra/Rg = 1 + K * concentration_ppm^n
Rs/Rair = 1 / (1 + K * concentration_ppm^n)
```

For mixtures, excess conductance terms `K * concentration^n` are added before
converting back to resistance. Concentrations are capped at the calibrated
upper limit of 100 ppm.

`R0` is the median late-air reference. The default divider is
`RL = 0.25 * R0`, which produces 1 V in clean air at `Vcc = 5 V`.
Rise and recovery constants are gas-specific effective estimates from workbook
transitions; they include chamber filling and purging effects.

Active calibration uses CROSS for `TGS2610`, `TGS2611`, and `TGS2600`.
Other families use MULTI when available, otherwise CROSS and then PRECOAT.
Rejected red workbook channels do not contribute to nominal fits.

## Supported calibrated gases

Runtime curves are available for methanol, ethanol, acetic acid, ethyl acetate,
toluene, and acetone. The four gases absent from upstream GADEN are added by
`patches/gaden_core_add_excel_gases.patch`; the setup script applies this
patch idempotently before building GADEN.

## Regeneration and validation

```bash
python3 tools/build_mox_registry.py \
  --source-dir /path/to/MOX_calibration_data

python3 tools/audit_mox_ratio_windows.py \
  --source-dir /path/to/MOX_calibration_data

python3 tools/compare_mox_aggregation.py \
  --source-dir /path/to/MOX_calibration_data

python tools/validate_mox_runtime.py
python tools/smoke_test_gaden_gases.py
```

The runtime YAML contains only nominal profiles. Detailed physical-channel
provenance is generated under `analysis/` and is not loaded by OdorSim.
Fit tables and exclusions are summarized in
`docs/mox_calibration_report.md`.

Generated multi-megabyte time-series CSVs, PDFs, and PNGs should normally stay
out of Git. Compact summaries or provenance may be committed deliberately when
