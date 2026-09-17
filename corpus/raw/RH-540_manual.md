# Overview
The RH-540 is a mid-range rotational rheometer designed for precise characterization of fluid, semi-solid, and solid materials. Featuring an integrated Peltier temperature control system, it provides accurate thermal regulation of the sample environment across a wide operational range. The instrument utilizes a high-resolution optical encoder for angular displacement and a brushless DC motor to apply controlled stress or strain. This manual outlines the technical specifications, standard calibration procedures, diagnostic troubleshooting steps, and error code definitions required for safe and effective operation of the RH-540.

# Specifications

| Specification | Value |
| :--- | :--- |
| Torque range | 0.05 to 200 mNm |
| Speed range | 0.001 to 1000 rpm |
| Temperature range | -20 to 200 C |

# Calibration Procedure
1. Mount the selected measuring geometry securely onto the drive shaft.
2. Lower the geometry toward the Peltier plate until contact with the plate surface is detected.
3. Zero the gap reading at the exact point of contact.
4. Remove the geometry and recalibrate the normal force sensor to zero.
5. Remount the geometry and raise it to the designated working gap.
6. Set the Peltier plate to the target temperature via the software and allow it to stabilize.
7. Confirm that both the gap and temperature readings have stabilized within acceptable tolerances before starting the test.

# Troubleshooting

| Symptom | Error Code |
| :--- | :--- |
| Normal force reading does not return to zero when the geometry is lifted clear of the sample | E-602 |
| Measurement start is prevented prior to executing the initial setup routine | E-601 |
| Peltier plate temperature exceeds safety thresholds during operation | E-701 |

# Error Codes

## E-601
* **Cause:** The measuring gap has not been zeroed since the geometry was last changed.
* **Resolution:** Run the gap zeroing procedure from the setup menu before starting a measurement.

## E-602
* **Cause:** The normal force sensor has drifted out of its calibrated zero point.
* **Resolution:** Recalibrate the normal force sensor from the maintenance menu with no geometry mounted.

## E-701
* **Cause:** The Peltier plate exceeded its safe operating temperature.
* **Resolution:** Allow the plate to cool to ambient, check that the cooling water supply is connected, and restart the measurement.