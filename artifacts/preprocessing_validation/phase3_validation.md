# Phase 3 Preprocessing Validation

Patients processed: 100
Failed patients: 0

## Valid Depth
|        |     0 |
|:-------|------:|
| min    |  7    |
| p05    |  8    |
| median | 12    |
| mean   | 11.16 |
| p95    | 13    |
| max    | 14    |

## Z Padding
|        |    0 |
|:-------|-----:|
| min    | 0    |
| p05    | 1    |
| median | 2    |
| mean   | 2.84 |
| p95    | 6    |
| max    | 7    |

## Phase 1 Patient-Level Parity
PASS: Phase 1 patient-level depth/padding parity

## Output Validation
|                              |   0 |
|:-----------------------------|----:|
| correct_shape                | 100 |
| float32                      | 100 |
| contiguous                   | 100 |
| finite                       | 100 |
| target_spacing_expected      | 100 |
| no_xy_padding                | 100 |
| valid_depth_leq_14           | 100 |
| no_z_cropping                | 100 |
| normalization_std_gt_epsilon | 100 |

## Failures
None.
