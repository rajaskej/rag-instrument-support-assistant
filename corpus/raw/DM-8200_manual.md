# Overview

The DM-8200 is a high-end automated density meter equipped with an integrated sample changer, designed for precise laboratory-grade density and specific gravity measurements. It utilizes the oscillating U-tube method combined with advanced Peltier temperature control to deliver rapid and accurate results across a wide range of industrial and research applications. The automated sample changer allows for high-throughput sample processing, minimizing operator intervention while maximizing repeatability and sample integrity.

# Specifications

| Parameter | Value |
| :--- | :--- |
| Density range | 0 to 3 g/cm3 |
| Accuracy | ±0.00001 g/cm3 |
| Temperature range | -10 to 150 C |
| Sample volume | 0.7 mL |

# Calibration Procedure

1. Power on the instrument and allow a 20-minute warm-up period to ensure thermal equilibrium.
2. Load the three automated reference standards into designated slots within the carousel.
3. Access the touchscreen menu and initiate the automated 3-point calibration routine.
4. Wait for the carousel to cycle through all three reference standards and complete the measurement cycle.
5. Review the generated calibration report, checking for any deviations exceeding 0.00002 g/cm3.
6. Confirm and save the calibration parameters if all three points pass verification.

# Troubleshooting

| Symptom | Error Code |
| :--- | :--- |
| Instrument stops mid-run with the carousel motor still audible | E-402 |
| U-tube oscillation signal unstable during fluid introduction | E-201 |
| Measurement cell fails to reach setpoint temperature | E-104 |

# Error Codes

## E-104
**Cause:** The thermoelectric module has degraded or lost contact with the heat sink, resulting in a Peltier temperature control fault.
**Resolution:** Power down the instrument, inspect the Peltier module contact, and replace the thermoelectric module if the fault persists after reseating.

## E-201
**Cause:** Thermostat connection loose or ambient temperature outside the specified operating range, causing a temperature stabilization timeout.
**Resolution:** Check the thermostat cable connection and ensure ambient temperature is between 15 and 35 C.

## E-402
**Cause:** A sample vial is misaligned in the carousel, resulting in a sample changer jam.
**Resolution:** Open the carousel cover, clear the jammed vial, and re-home the carousel from the maintenance menu.