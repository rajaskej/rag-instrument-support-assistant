# Overview
The RH-220 is an entry-level rotational rheometer designed for routine viscosity measurements and flow behavior analysis. It features a compact footprint, a direct-drive DC motor for precise rotational control, and an intuitive front-panel interface. The instrument is engineered for laboratory environments where reliability, ease of use, and quick sample turnaround are paramount.

# Specifications
| Parameter | Value |
| :--- | :--- |
| Torque range | 0.1 to 150 mNm |
| Speed range | 0.01 to 500 rpm |
| Temperature range | ambient only, no active temperature control |

# Calibration Procedure
1. Mount the selected measuring geometry securely onto the drive shaft.
2. Lower the geometry carefully until contact with the opposing plate is detected by the internal sensor.
3. Zero the gap reading at the established point of contact.
4. Raise the geometry to the appropriate working gap specified for the chosen test.
5. Confirm that the gap value indicated on the display matches the target gap.

# Troubleshooting
| Symptom | Error Code |
| :--- | :--- |
| Torque reading pins at maximum immediately on startup | E-501 |
| Drive motor fails to respond to rotational speed commands | E-601 |
| Display interface freezes during sample data acquisition | E-601 |

# Error Codes
## E-501
**Cause:** The sample load exceeds the instrument's maximum torque rating.
**Resolution:** Reduce the sample load or switch to a smaller measuring geometry.

## E-601
**Cause:** The measuring gap has not been zeroed since the geometry was last changed.
**Resolution:** Run the gap zeroing procedure from the setup menu before starting a measurement.