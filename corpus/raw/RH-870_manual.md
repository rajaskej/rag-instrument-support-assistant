# Overview

The RH-870 is a high-end rheometer designed for precise oscillatory and rotational testing of complex fluids, polymers, and semi-solids. Featuring an air-bearing supported drive motor and an advanced optical encoder, the instrument delivers exceptional torque sensitivity and speed control. The RH-870 is equipped with an integrated Peltier temperature control system to ensure accurate thermal regulation during demanding rheological characterization.

# Specifications

| Parameter | Value |
| :--- | :--- |
| Torque range | 0.01 to 300 mNm |
| Speed range | 0.0001 to 1500 rpm |
| Temperature range | -40 to 300 C |

# Calibration Procedure

1. Mount the selected measuring geometry onto the drive shaft.
2. Lower the geometry toward the lower plate until contact is detected by the normal force sensor.
3. Zero the gap reading at the exact point of contact.
4. Remove the geometry and recalibrate the normal force sensor to zero.
5. Remount the geometry and raise it to the designated working gap.
6. Set the Peltier plate to the target temperature and allow it to stabilize completely.
7. Run the oscillation amplitude check using the appropriate reference standard.
8. Confirm the gap, normal force zero, and temperature readings before starting the test.

# Troubleshooting

| Symptom | Error Code |
| :--- | :--- |
| Oscillation test aborts immediately when a high frequency sweep is requested | E-801 |
| Normal force readings drift unpredictably during long-term static testing | E-602 |
| Peltier temperature control shuts down unexpectedly during a 250 C thermal ramp | E-701 |

# Error Codes

## E-602
**Cause:** The normal force sensor has drifted out of its calibrated zero point.
**Resolution:** Recalibrate the normal force sensor from the maintenance menu with no geometry mounted.

## E-701
**Cause:** The Peltier plate exceeded its safe operating temperature.
**Resolution:** Allow the plate to cool to ambient, check that the cooling water supply is connected, and restart the measurement.

## E-801
**Cause:** The requested oscillation frequency exceeds the instrument's calibrated bandwidth.
**Resolution:** Reduce the requested frequency to within the calibrated range shown in the specifications, or run a bandwidth recalibration.