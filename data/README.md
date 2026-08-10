# Data

This directory is reserved for local medical imaging data used by the project.

## Primary Dataset

The project uses:

**ACDC — Automated Cardiac Diagnosis Challenge**

The dataset must be downloaded manually from the official source.

Automatic dataset downloading is intentionally not part of the project bootstrap.

## Local Data Policy

Raw ACDC data must remain local and must never be committed to GitHub.

Do not commit:

* ACDC archives;
* NIfTI volumes;
* extracted patient folders;
* temporary converted data;
* generated preprocessing outputs.

The project `.gitignore` is configured to exclude raw files stored under this directory while keeping this `README.md` tracked.

## Expected Use

After the dataset is downloaded, the local ACDC location will be referenced through project configuration.

The exact directory layout will be documented after the real dataset is downloaded and inspected during Phase 1.

## Phase 1

The first real-data phase will inspect:

* patient count;
* diagnostic classes;
* ED and ES frames;
* image dimensions;
* slice counts;
* X/Y/Z spacing;
* physical Z coverage;
* orientation;
* affine information;
* intensity and background characteristics.

No preprocessing constants will be frozen before this inspection is complete.
