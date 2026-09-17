# Overview

The DM-5400 is a benchtop mid-range density meter designed for routine laboratory and industrial quality control measurements. The instrument measures true liquid density utilizing an oscillating U-tube measuring cell combined with integrated viscosity correction to account for damping effects in viscous media. Featuring automated temperature control and internal data processing, the DM-5400 ensures consistent, high-precision results across a broad range of liquid sample types.

# Specifications

| Parameter | Value |
| :--- | :--- |
| Density Range | 0 to 3 g/cm3 |
| Accuracy | ±0.00005 g/cm3 |
| Temperature Range | 0 to 95 C |
| Sample Volume | 1 mL |

# Calibration Procedure

1. Power on the device and wait 20 minutes to allow internal components to reach thermal equilibrium.
2. Flush the measuring cell three successive times with degassed distilled water.
3. Introduce dry air into the measuring cell and log the initial calibration reference point.
4. Fill the cell with the distilled water reference and register the second calibration point.
5. Introduce the certified viscosity correction reference standard fluid into the system.
6. Execute the integrated viscosity correction diagnostic check and verify that the test completes successfully.
7. Ensure that the total measured deviation is within 0.00002 g/cm3, then store the new calibration profile in memory.

# Troubleshooting

| Symptom | Error Code |
| :--- | :--- |
| Reading is unstable and drifts erratically mid-measurement | E-104 |
| Target setpoint temperature fails to stabilize within timeout threshold | E-201 |
| Automatic viscosity compensation inactive or reporting invalid state | E-301 |

# Error Codes

## E-104

* **Description:** Air bubble detected in the density cell.
* **Cause:** Incomplete sample fill or trapped air introduced during fluid injection.
* **Resolution:** Purge the cell using the built-in purge cycle. Refill the measuring cell slowly and steadily to prevent cavitation and fluid turbulence.

## E-201

* **Description:** Temperature stabilization timeout.
* **Cause:** The internal thermostat connection is loose, or ambient room temperature is outside the specified operating limits.
* **Resolution:** Inspect and secure the thermostat cable interface. Verify that the ambient room temperature is maintained between 15 and 35 C.

## E-301

* **Description:** Viscosity correction sensor fault.
* **Cause:** The inline viscosity sensor has lost its calibration parameters, or the sensor interface cable is disconnected.
* **Resolution:** Reseat the sensor connection cable securely. Access the maintenance menu and perform the viscosity sensor self-test routine.