# MOX Calibration Import Report

Physical-channel `Ra/Rg` values are median-aggregated at each concentration, then fitted as `Ra/Rg - 1 = K * concentration^n` once per nominal sensor.

## Summary

- Nominal families: 17
- Unfitted physical-channel records: 113
- Stored gases: methanol, ethanol, acetic_acid, ethyl_acetate, toluene, acetone
- Runtime-enabled gases: acetic_acid, acetone, ethanol, ethyl_acetate, methanol, toluene
- Source priority: MULTI, then CROSS, then PRECOAT
- CROSS overrides: TGS2610, TGS2611, TGS2600
- Calibrated range: 1-100 ppm; conductance law approaches clean air below it and caps above it
- Dynamics: effective tau=(t90-t10)/ln(9), median per physical channel then across channels
- Coarse timing warning: workbook cadence is approximately 9-14 seconds
- Absolute resistance: median adaptive late-air baseline
- Load resistor: RL=0.25*Rair_reference (1 V in air at Vcc=5 V)

## Nominal Fits

| Model | Source | Replicates | R0 ohm | Gas | K | n | Rs/Rair RMSE | Runtime |
|---|---|---:|---:|---|---:|---:|---:|:---:|
| TGS2620 | MULTI2023 | 1 | 11688.70 | methanol | 0.139386 | 0.704023 | 0.01064 | yes |
| TGS2620 | MULTI2023 | 1 | 11688.70 | ethanol | 0.196652 | 0.683629 | 0.01963 | yes |
| TGS2620 | MULTI2023 | 1 | 11688.70 | acetic_acid | 0.0672372 | 0.756029 | 0.01329 | yes |
| TGS2620 | MULTI2023 | 1 | 11688.70 | ethyl_acetate | 0.094478 | 0.752232 | 0.00413 | yes |
| TGS2620 | MULTI2023 | 1 | 11688.70 | toluene | 0.176293 | 0.639574 | 0.01912 | yes |
| TGS2620 | MULTI2023 | 1 | 11688.70 | acetone | 0.0925202 | 0.755204 | 0.00779 | yes |
| TGS2600 | CROSS2024 | 2 | 16048.10 | methanol | 0.133097 | 0.649095 | 0.01087 | yes |
| TGS2600 | CROSS2024 | 2 | 16048.10 | ethanol | 0.151558 | 0.67956 | 0.00807 | yes |
| TGS2600 | CROSS2024 | 2 | 16048.10 | acetic_acid | 0.0794684 | 0.765339 | 0.00756 | yes |
| TGS2600 | CROSS2024 | 2 | 16048.10 | ethyl_acetate | 0.187042 | 0.720985 | 0.00628 | yes |
| TGS2600 | CROSS2024 | 2 | 16048.10 | toluene | 0.400489 | 0.458967 | 0.04597 | yes |
| TGS2600 | CROSS2024 | 2 | 16048.10 | acetone | 0.150536 | 0.742004 | 0.00718 | yes |
| TGS2611 | CROSS2024 | 2 | 26838.38 | methanol | 0.0215963 | 0.662475 | 0.00692 | yes |
| TGS2611 | CROSS2024 | 2 | 26838.38 | ethanol | 0.0237336 | 0.688051 | 0.00536 | yes |
| TGS2611 | CROSS2024 | 2 | 26838.38 | acetic_acid | 0.0144429 | 0.685447 | 0.00415 | yes |
| TGS2611 | CROSS2024 | 2 | 26838.38 | ethyl_acetate | 0.0282719 | 0.676752 | 0.00493 | yes |
| TGS2611 | CROSS2024 | 2 | 26838.38 | toluene | 0.0143916 | 0.601762 | 0.00378 | yes |
| TGS2611 | CROSS2024 | 2 | 26838.38 | acetone | 0.0273503 | 0.64102 | 0.00911 | yes |
| TGS2610 | CROSS2024 | 2 | 42358.86 | methanol | 0.0203432 | 0.684291 | 0.00572 | yes |
| TGS2610 | CROSS2024 | 2 | 42358.86 | ethanol | 0.0220286 | 0.70326 | 0.00296 | yes |
| TGS2610 | CROSS2024 | 2 | 42358.86 | acetic_acid | 0.0146095 | 0.678939 | 0.00454 | yes |
| TGS2610 | CROSS2024 | 2 | 42358.86 | ethyl_acetate | 0.0279257 | 0.680992 | 0.00235 | yes |
| TGS2610 | CROSS2024 | 2 | 42358.86 | toluene | 0.0181407 | 0.587745 | 0.00364 | yes |
| TGS2610 | CROSS2024 | 2 | 42358.86 | acetone | 0.0235391 | 0.667227 | 0.00741 | yes |
| TGS2602 | MULTI2023 | 1 | 35298.22 | methanol | 0.192315 | 0.848422 | 0.01487 | yes |
| TGS2602 | MULTI2023 | 1 | 35298.22 | ethanol | 0.224415 | 0.869714 | 0.01794 | yes |
| TGS2602 | MULTI2023 | 1 | 35298.22 | acetic_acid | 0.126811 | 0.798505 | 0.01556 | yes |
| TGS2602 | MULTI2023 | 1 | 35298.22 | ethyl_acetate | 0.0232016 | 0.862158 | 0.00958 | yes |
| TGS2602 | MULTI2023 | 1 | 35298.22 | toluene | 0.0967318 | 0.681662 | 0.01355 | yes |
| TGS2602 | MULTI2023 | 1 | 35298.22 | acetone | 0.127437 | 0.879342 | 0.01135 | yes |
| 101 | CROSS2024 | 5 | 404260.78 | methanol | 0.125761 | 0.604368 | 0.01582 | yes |
| 101 | CROSS2024 | 5 | 404260.78 | ethanol | 0.133489 | 0.563897 | 0.01786 | yes |
| 101 | CROSS2024 | 5 | 404260.78 | acetic_acid | 0.0488554 | 0.655003 | 0.00468 | yes |
| 101 | CROSS2024 | 5 | 404260.78 | ethyl_acetate | 0.06318 | 0.624591 | 0.00331 | yes |
| 101 | CROSS2024 | 5 | 404260.78 | toluene | 0.0317335 | 0.534361 | 0.00379 | yes |
| 101 | CROSS2024 | 5 | 404260.78 | acetone | 0.0518377 | 0.631622 | 0.00678 | yes |
| 203 | CROSS2024 | 6 | 118697.86 | methanol | 0.36551 | 0.58578 | 0.02000 | yes |
| 203 | CROSS2024 | 6 | 118697.86 | ethanol | 0.380032 | 0.609647 | 0.01877 | yes |
| 203 | CROSS2024 | 6 | 118697.86 | acetic_acid | 0.74494 | 0.388372 | 0.05288 | yes |
| 203 | CROSS2024 | 6 | 118697.86 | ethyl_acetate | 0.295111 | 0.566206 | 0.02331 | yes |
| 203 | CROSS2024 | 6 | 118697.86 | toluene | 0.492647 | 0.378234 | 0.02881 | yes |
| 203 | CROSS2024 | 6 | 118697.86 | acetone | 0.564324 | 0.539288 | 0.02372 | yes |
| 302 | CROSS2024 | 6 | 10791602.29 | methanol | 1.08095 | 0.863464 | 0.03028 | yes |
| 302 | CROSS2024 | 6 | 10791602.29 | ethanol | 1.52939 | 0.898505 | 0.02981 | yes |
| 302 | CROSS2024 | 6 | 10791602.29 | acetic_acid | 1.04176 | 0.770478 | 0.02921 | yes |
| 302 | CROSS2024 | 6 | 10791602.29 | ethyl_acetate | 1.33674 | 0.79459 | 0.03104 | yes |
| 302 | CROSS2024 | 6 | 10791602.29 | toluene | 0.587234 | 0.334808 | 0.00786 | yes |
| 302 | CROSS2024 | 6 | 10791602.29 | acetone | 1.38573 | 0.862661 | 0.01737 | yes |
| 401 | CROSS2024 | 6 | 746033.11 | methanol | 0.360935 | 0.723387 | 0.03307 | yes |
| 401 | CROSS2024 | 6 | 746033.11 | ethanol | 0.323954 | 0.786196 | 0.02040 | yes |
| 401 | CROSS2024 | 6 | 746033.11 | acetic_acid | 0.581547 | 0.639625 | 0.02852 | yes |
| 401 | CROSS2024 | 6 | 746033.11 | ethyl_acetate | 0.0685274 | 0.822329 | 0.03060 | yes |
| 401 | CROSS2024 | 6 | 746033.11 | toluene | 0.0587626 | 1.01147 | 0.01308 | yes |
| 401 | CROSS2024 | 6 | 746033.11 | acetone | 0.170937 | 0.875024 | 0.00411 | yes |
| 509 | CROSS2024 | 6 | 992719.60 | methanol | 1.08456 | 0.697214 | 0.00747 | yes |
| 509 | CROSS2024 | 6 | 992719.60 | ethanol | 1.14698 | 0.726182 | 0.01331 | yes |
| 509 | CROSS2024 | 6 | 992719.60 | acetic_acid | 0.522714 | 0.776955 | 0.00513 | yes |
| 509 | CROSS2024 | 6 | 992719.60 | ethyl_acetate | 1.3467 | 0.732571 | 0.00857 | yes |
| 509 | CROSS2024 | 6 | 992719.60 | toluene | 0.365435 | 0.488448 | 0.04330 | yes |
| 509 | CROSS2024 | 6 | 992719.60 | acetone | 1.60556 | 0.733869 | 0.00539 | yes |
| 104 | PRECOAT2024 | 18 | 43657.83 | methanol | 0.226044 | 0.681804 | 0.03481 | yes |
| 104 | PRECOAT2024 | 18 | 43657.83 | ethanol | 0.468189 | 0.651213 | 0.04131 | yes |
| 104 | PRECOAT2024 | 18 | 43657.83 | acetic_acid | 0.2256 | 0.602106 | 0.01309 | yes |
| 104 | PRECOAT2024 | 18 | 43657.83 | ethyl_acetate | 0.169982 | 0.617613 | 0.01580 | yes |
| 104 | PRECOAT2024 | 18 | 43657.83 | toluene | 0.0839699 | 0.499831 | 0.01518 | yes |
| 104 | PRECOAT2024 | 18 | 43657.83 | acetone | 0.277456 | 0.655524 | 0.01850 | yes |
| 303 | PRECOAT2024 | 18 | 310565.05 | methanol | 0.429755 | 0.961439 | 0.04335 | yes |
| 303 | PRECOAT2024 | 18 | 310565.05 | ethanol | 0.874295 | 0.975825 | 0.05139 | yes |
| 303 | PRECOAT2024 | 18 | 310565.05 | acetic_acid | 0.50926 | 0.930953 | 0.02108 | yes |
| 303 | PRECOAT2024 | 18 | 310565.05 | ethyl_acetate | 0.860976 | 0.827053 | 0.03450 | yes |
| 303 | PRECOAT2024 | 18 | 310565.05 | toluene | 0.184466 | 0.475224 | 0.01831 | yes |
| 303 | PRECOAT2024 | 18 | 310565.05 | acetone | 1.12704 | 0.924757 | 0.03159 | yes |
| 500-1 | MULTI2023 | 10 | 148254.72 | methanol | 1.32501 | 0.634867 | 0.01526 | yes |
| 500-1 | MULTI2023 | 10 | 148254.72 | ethanol | 0.767739 | 0.71982 | 0.00803 | yes |
| 500-1 | MULTI2023 | 10 | 148254.72 | acetic_acid | 0.615252 | 0.746092 | 0.02729 | yes |
| 500-1 | MULTI2023 | 10 | 148254.72 | ethyl_acetate | 0.941285 | 0.750731 | 0.01214 | yes |
| 500-1 | MULTI2023 | 10 | 148254.72 | toluene | 0.162492 | 0.671754 | 0.00379 | yes |
| 500-1 | MULTI2023 | 10 | 148254.72 | acetone | 1.80498 | 0.707011 | 0.00504 | yes |
| 43_SINTERED | MULTI2023 | 3 | 2552.59 | methanol | 0.0118609 | 0.707471 | 0.00584 | yes |
| 43_SINTERED | MULTI2023 | 3 | 2552.59 | ethanol | 0.0305644 | 0.645897 | 0.01423 | yes |
| 43_SINTERED | MULTI2023 | 3 | 2552.59 | acetic_acid | 0.0207414 | 0.57735 | 0.00863 | yes |
| 43_SINTERED | MULTI2023 | 3 | 2552.59 | ethyl_acetate | 0.00180189 | 0.8371 | 0.00063 | yes |
| 43_SINTERED | MULTI2023 | 3 | 2552.59 | toluene | 0.0024141 | 0.686323 | 0.00140 | yes |
| 43_SINTERED | MULTI2023 | 3 | 2552.59 | acetone | 0.0133987 | 0.706971 | 0.00891 | yes |
| 43_UNSINTERED | MULTI2023 | 3 | 925.31 | methanol | 0.009642 | 0.668539 | 0.00397 | yes |
| 43_UNSINTERED | MULTI2023 | 3 | 925.31 | ethanol | 0.0231054 | 0.618266 | 0.01069 | yes |
| 43_UNSINTERED | MULTI2023 | 3 | 925.31 | acetic_acid | 0.0153156 | 0.563889 | 0.00593 | yes |
| 43_UNSINTERED | MULTI2023 | 3 | 925.31 | ethyl_acetate | 0.00348368 | 0.685337 | 0.00061 | yes |
| 43_UNSINTERED | MULTI2023 | 3 | 925.31 | toluene | 0.00353945 | 0.590696 | 0.00133 | yes |
| 43_UNSINTERED | MULTI2023 | 3 | 925.31 | acetone | 0.0116291 | 0.675266 | 0.00582 | yes |
| 601 | MULTI2023 | 5 | 156488.96 | methanol | 0.199572 | 0.740839 | 0.00852 | yes |
| 601 | MULTI2023 | 5 | 156488.96 | ethanol | 0.37364 | 0.747506 | 0.04264 | yes |
| 601 | MULTI2023 | 5 | 156488.96 | acetic_acid | 0.36022 | 0.665663 | 0.03003 | yes |
| 601 | MULTI2023 | 5 | 156488.96 | ethyl_acetate | 0.0128711 | 0.929056 | 0.00706 | yes |
| 601 | MULTI2023 | 5 | 156488.96 | toluene | 0.0462871 | 0.70458 | 0.01785 | yes |
| 601 | MULTI2023 | 5 | 156488.96 | acetone | 0.168926 | 0.788216 | 0.01766 | yes |
| 701 | MULTI2023 | 5 | 401389.11 | methanol | 0.585945 | 0.596475 | 0.00356 | yes |
| 701 | MULTI2023 | 5 | 401389.11 | ethanol | 0.880624 | 0.675405 | 0.01875 | yes |
| 701 | MULTI2023 | 5 | 401389.11 | acetic_acid | 0.444867 | 0.603571 | 0.03088 | yes |
| 701 | MULTI2023 | 5 | 401389.11 | ethyl_acetate | 0.456053 | 0.615101 | 0.02445 | yes |
| 701 | MULTI2023 | 5 | 401389.11 | toluene | 0.0317911 | 0.552613 | 0.00201 | yes |
| 701 | MULTI2023 | 5 | 401389.11 | acetone | 0.445087 | 0.625609 | 0.01822 | yes |

## Effective Dynamics

These include chamber filling/purging and are not intrinsic sensor-only constants.

| Model | Gas | Rise tau s | Rise channels | Recovery tau s | Recovery channels |
|---|---|---:|---:|---:|---:|
| TGS2620 | methanol | 9.773 | 6 | 33.678 | 3 |
| TGS2620 | ethanol | 9.773 | 6 | 33.678 | 3 |
| TGS2620 | acetic_acid | 9.773 | 6 | 33.678 | 3 |
| TGS2620 | ethyl_acetate | 9.773 | 6 | 33.678 | 3 |
| TGS2620 | toluene | 9.773 | 6 | 33.678 | 3 |
| TGS2620 | acetone | 9.773 | 6 | 33.678 | 3 |
| TGS2600 | methanol | 9.425 | 2 | 79.376 | 2 |
| TGS2600 | ethanol | 13.687 | 2 | 32.906 | 2 |
| TGS2600 | acetic_acid | 15.045 | 2 | 55.694 | 4 |
| TGS2600 | ethyl_acetate | 13.006 | 2 | 55.694 | 4 |
| TGS2600 | toluene | 15.576 | 2 | 55.694 | 4 |
| TGS2600 | acetone | 11.152 | 2 | 55.694 | 4 |
| TGS2611 | methanol | 6.370 | 2 | 12.348 | 2 |
| TGS2611 | ethanol | 6.563 | 2 | 9.372 | 4 |
| TGS2611 | acetic_acid | 7.792 | 2 | 9.372 | 4 |
| TGS2611 | ethyl_acetate | 7.684 | 2 | 9.372 | 4 |
| TGS2611 | toluene | 6.489 | 2 | 9.372 | 4 |
| TGS2611 | acetone | 6.753 | 2 | 9.372 | 4 |
| TGS2610 | methanol | 6.101 | 2 | 18.120 | 2 |
| TGS2610 | ethanol | 7.103 | 2 | 11.309 | 5 |
| TGS2610 | acetic_acid | 8.194 | 2 | 11.309 | 5 |
| TGS2610 | ethyl_acetate | 8.009 | 2 | 11.040 | 2 |
| TGS2610 | toluene | 6.952 | 2 | 11.309 | 5 |
| TGS2610 | acetone | 7.155 | 2 | 11.309 | 5 |
| TGS2602 | methanol | 22.830 | 6 | 15.000 | 0 |
| TGS2602 | ethanol | 22.830 | 6 | 15.000 | 0 |
| TGS2602 | acetic_acid | 22.830 | 6 | 15.000 | 0 |
| TGS2602 | ethyl_acetate | 22.830 | 6 | 15.000 | 0 |
| TGS2602 | toluene | 22.830 | 6 | 15.000 | 0 |
| TGS2602 | acetone | 22.830 | 6 | 15.000 | 0 |
| 101 | methanol | 4.223 | 5 | 24.170 | 2 |
| 101 | ethanol | 7.210 | 5 | 4.198 | 2 |
| 101 | acetic_acid | 7.838 | 5 | 11.685 | 4 |
| 101 | ethyl_acetate | 5.080 | 5 | 11.685 | 4 |
| 101 | toluene | 7.764 | 5 | 11.685 | 4 |
| 101 | acetone | 6.860 | 5 | 11.685 | 4 |
| 203 | methanol | 4.540 | 6 | 6.741 | 7 |
| 203 | ethanol | 4.553 | 6 | 6.741 | 7 |
| 203 | acetic_acid | 4.153 | 6 | 6.741 | 7 |
| 203 | ethyl_acetate | 6.782 | 6 | 6.741 | 7 |
| 203 | toluene | 5.808 | 6 | 6.741 | 7 |
| 203 | acetone | 5.322 | 6 | 6.670 | 6 |
| 302 | methanol | 6.555 | 6 | 29.672 | 6 |
| 302 | ethanol | 6.078 | 6 | 49.323 | 6 |
| 302 | acetic_acid | 4.916 | 6 | 51.945 | 6 |
| 302 | ethyl_acetate | 4.242 | 6 | 61.782 | 5 |
| 302 | toluene | 8.361 | 6 | 61.695 | 5 |
| 302 | acetone | 4.089 | 6 | 29.968 | 6 |
| 401 | methanol | 6.565 | 6 | 105.570 | 6 |
| 401 | ethanol | 6.477 | 6 | 42.341 | 6 |
| 401 | acetic_acid | 7.535 | 6 | 43.912 | 21 |
| 401 | ethyl_acetate | 11.110 | 6 | 43.912 | 21 |
| 401 | toluene | 10.402 | 6 | 28.869 | 2 |
| 401 | acetone | 5.260 | 6 | 32.213 | 5 |
| 509 | methanol | 4.075 | 6 | 11.436 | 4 |
| 509 | ethanol | 3.729 | 6 | 10.471 | 6 |
| 509 | acetic_acid | 7.993 | 6 | 11.436 | 22 |
| 509 | ethyl_acetate | 3.843 | 6 | 11.436 | 22 |
| 509 | toluene | 15.828 | 6 | 67.910 | 6 |
| 509 | acetone | 4.063 | 6 | 11.324 | 6 |
| 104 | methanol | 10.432 | 18 | 36.487 | 18 |
| 104 | ethanol | 9.321 | 18 | 33.787 | 18 |
| 104 | acetic_acid | 11.288 | 18 | 46.571 | 18 |
| 104 | ethyl_acetate | 14.921 | 18 | 41.774 | 18 |
| 104 | toluene | 22.028 | 18 | 21.396 | 5 |
| 104 | acetone | 9.578 | 18 | 32.524 | 18 |
| 303 | methanol | 5.140 | 5 | 65.190 | 5 |
| 303 | ethanol | 5.370 | 5 | 50.317 | 5 |
| 303 | acetic_acid | 8.344 | 5 | 28.412 | 5 |
| 303 | ethyl_acetate | 7.822 | 18 | 51.361 | 18 |
| 303 | toluene | 11.052 | 18 | 21.430 | 15 |
| 303 | acetone | 5.071 | 18 | 35.377 | 18 |
| 500-1 | methanol | 5.439 | 10 | 106.358 | 6 |
| 500-1 | ethanol | 4.236 | 10 | 62.939 | 7 |
| 500-1 | acetic_acid | 19.603 | 10 | 190.453 | 5 |
| 500-1 | ethyl_acetate | 6.512 | 10 | 170.851 | 8 |
| 500-1 | toluene | 12.459 | 10 | 52.419 | 9 |
| 500-1 | acetone | 3.884 | 10 | 15.738 | 9 |
| 43_SINTERED | methanol | 10.800 | 3 | 13.812 | 12 |
| 43_SINTERED | ethanol | 17.606 | 3 | 14.228 | 3 |
| 43_SINTERED | acetic_acid | 24.860 | 3 | 19.656 | 3 |
| 43_SINTERED | ethyl_acetate | 5.929 | 3 | 6.333 | 3 |
| 43_SINTERED | toluene | 12.163 | 3 | 13.812 | 12 |
| 43_SINTERED | acetone | 16.082 | 3 | 18.382 | 3 |
| 43_UNSINTERED | methanol | 15.743 | 3 | 11.035 | 3 |
| 43_UNSINTERED | ethanol | 20.437 | 3 | 21.647 | 3 |
| 43_UNSINTERED | acetic_acid | 23.348 | 3 | 20.602 | 3 |
| 43_UNSINTERED | ethyl_acetate | 10.056 | 3 | 7.566 | 3 |
| 43_UNSINTERED | toluene | 11.682 | 3 | 15.409 | 15 |
| 43_UNSINTERED | acetone | 19.230 | 3 | 21.334 | 3 |
| 601 | methanol | 9.527 | 5 | 54.966 | 14 |
| 601 | ethanol | 7.096 | 5 | 32.978 | 5 |
| 601 | acetic_acid | 8.207 | 5 | 54.966 | 14 |
| 601 | ethyl_acetate | 16.209 | 5 | 54.966 | 14 |
| 601 | toluene | 17.192 | 5 | 71.587 | 4 |
| 601 | acetone | 8.981 | 5 | 52.879 | 4 |
| 701 | methanol | 5.365 | 5 | 42.731 | 2 |
| 701 | ethanol | 5.114 | 5 | 45.271 | 5 |
| 701 | acetic_acid | 10.865 | 5 | 135.314 | 4 |
| 701 | ethyl_acetate | 10.845 | 5 | 47.311 | 5 |
| 701 | toluene | 36.358 | 5 | 125.988 | 3 |
| 701 | acetone | 8.847 | 5 | 93.542 | 4 |

## Quality Exclusions

- CROSS2024/ch10/acetic_acid: red raw-sheet channel excluded
- CROSS2024/ch10/acetone: red raw-sheet channel excluded
- CROSS2024/ch10/ethanol: red raw-sheet channel excluded
- CROSS2024/ch10/ethyl_acetate: red raw-sheet channel excluded
- CROSS2024/ch10/methanol: red raw-sheet channel excluded
- CROSS2024/ch10/toluene: red raw-sheet channel excluded
- PRECOAT2024/ch33/acetic_acid: red raw-sheet channel excluded
- PRECOAT2024/ch33/ethanol: red raw-sheet channel excluded
- PRECOAT2024/ch33/methanol: red raw-sheet channel excluded
- PRECOAT2024/ch34/acetic_acid: red raw-sheet channel excluded
- PRECOAT2024/ch34/ethanol: red raw-sheet channel excluded
- PRECOAT2024/ch34/methanol: red raw-sheet channel excluded
- PRECOAT2024/ch36/acetic_acid: red raw-sheet channel excluded
- PRECOAT2024/ch36/ethanol: red raw-sheet channel excluded
- PRECOAT2024/ch36/methanol: red raw-sheet channel excluded
- PRECOAT2024/ch37/acetic_acid: red raw-sheet channel excluded
- PRECOAT2024/ch37/ethanol: red raw-sheet channel excluded
- PRECOAT2024/ch37/methanol: red raw-sheet channel excluded
- PRECOAT2024/ch39/acetic_acid: red raw-sheet channel excluded
- PRECOAT2024/ch39/ethanol: red raw-sheet channel excluded
- PRECOAT2024/ch39/methanol: red raw-sheet channel excluded
- PRECOAT2024/ch41/acetic_acid: red raw-sheet channel excluded
- PRECOAT2024/ch41/ethanol: red raw-sheet channel excluded
- PRECOAT2024/ch41/methanol: red raw-sheet channel excluded
- PRECOAT2024/ch42/acetic_acid: red raw-sheet channel excluded
- PRECOAT2024/ch42/ethanol: red raw-sheet channel excluded
- PRECOAT2024/ch42/methanol: red raw-sheet channel excluded
- PRECOAT2024/ch43/acetic_acid: red raw-sheet channel excluded
- PRECOAT2024/ch43/ethanol: red raw-sheet channel excluded
- PRECOAT2024/ch43/methanol: red raw-sheet channel excluded
- PRECOAT2024/ch45/acetic_acid: red raw-sheet channel excluded
- PRECOAT2024/ch45/ethanol: red raw-sheet channel excluded
- PRECOAT2024/ch45/methanol: red raw-sheet channel excluded
- PRECOAT2024/ch47/acetic_acid: red raw-sheet channel excluded
- PRECOAT2024/ch47/ethanol: red raw-sheet channel excluded
- PRECOAT2024/ch47/methanol: red raw-sheet channel excluded
- PRECOAT2024/ch48/acetic_acid: red raw-sheet channel excluded
- PRECOAT2024/ch48/ethanol: red raw-sheet channel excluded
- PRECOAT2024/ch48/methanol: red raw-sheet channel excluded
- PRECOAT2024/ch49/acetic_acid: red raw-sheet channel excluded
- PRECOAT2024/ch49/ethanol: red raw-sheet channel excluded
- PRECOAT2024/ch49/methanol: red raw-sheet channel excluded
- PRECOAT2024/ch50/acetic_acid: red raw-sheet channel excluded
- PRECOAT2024/ch50/ethanol: red raw-sheet channel excluded
- PRECOAT2024/ch50/methanol: red raw-sheet channel excluded
- MULTI2023/Sheet6-500-1!R36C10: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R37C10: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R38C10: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R39C10: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R40C10: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R36C11: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R37C11: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R38C11: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R39C11: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R40C11: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R36C12: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R37C12: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R38C12: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R39C12: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R40C12: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R36C13: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R37C13: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R38C13: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R39C13: invalid Ra/Rg=None
- MULTI2023/Sheet6-500-1!R40C13: invalid Ra/Rg=None
