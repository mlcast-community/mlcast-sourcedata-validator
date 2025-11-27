# mlcast-sourcedata-validator

Source data validator for the MLCast Intake catalog ([mlcast-datasets](https://github.com/mlcast-community/mlcast-datasets)).

## What is this?

This repository contains a validation tool for radar datasets contributed to the [MLCast community](https://github.com/mlcast-community). The validator ensures that 2D radar precipitation/reflectivity composite datasets meet the technical requirements for inclusion in the MLCast data collection.

### Background

During the MLCast community meeting, multiple entities offered to contribute radar datasets. To streamline the contribution process and ensure data quality, we developed this validator to help data providers verify that their Zarr archives are compliant with MLCast requirements before submission.

This tool addresses two key needs identified in the community:
1. **Specification compliance** ([#6](https://github.com/mlcast-community/mlcast-datasets/issues/6)): Validates datasets against the formal MLCast Zarr format specification v1.0 (RFC 2119 keywords)
2. **Tool compatibility** ([#5](https://github.com/mlcast-community/mlcast-datasets/issues/5)): Tests that datasets work correctly with common geospatial tools (xarray, GDAL, cartopy)

### What does it validate?

The validator checks both **specification compliance** and **practical tool compatibility**, for example for radar precipitation datasets it checks:

- **Minimum Requirements for Dataset Acceptance:**
    - 2D radar composite at 1km resolution or finer
    - At least 256×256 pixel valid sensing area
    - Minimum 3 years of temporal coverage
    - Consistent spatial domain across all timesteps
    - Data variable in mm (depth), mm/h (rate), or dBZ (reflectivity)

- **Technical Requirements:**
    - GeoZarr format (Zarr v2/v3 with proper georeferencing)
    - CF-compliant coordinate and variable names
    - Correct dimension ordering (time × H × W)
    - Proper chunking strategy (1 chunk per timestep)
    - ZSTD compression (recommended)
    - NaN values for missing/out-of-range data
    - License metadata (CC-BY, CC-BY-SA, OGL, etc.)

- **Tool Compatibility:**
    - xarray can load and slice the data correctly
    - GDAL can interpret the georeferencing (WKT parsing)
    - cartopy can create CRS objects and transform coordinates
    - Cross-tool CRS consistency checks

## How is the tool implemented organized?

*TDB* based on new spec document structure being discussed in https://github.com/mlcast-community/mlcast-sourcedata-validator/pull/4.


## Example usage

Until `mllam-sourcedata-validator` is published to PyPI, the easiest way to run it is to use [uv](https://docs.astral.sh/uv/getting-started/installation/) to run it directly from the GitHub repository:

```bash
uvx --with "git+https://github.com/mlcast-community/mlcast-sourcedata-validator" python -m mlcast_dataset_validator.specs.source_data.radar_precipitation <dataset-path>
```

I.e. you can validate a local Zarr dataset like this:
```bash
uvx --with "git+https://github.com/mlcast-community/mlcast-sourcedata-validator" python -m mlcast_dataset_validator.specs.source_data.radar_precipitation /path/to/zarr/file.zarr
```

The validator supports also remote zarr hosted in S3 buckets at custom endpoints. We can run it on the Radklim Zarr already available in the intake catalog:

```bash
uvx --with "git+https://github.com/mlcast-community/mlcast-sourcedata-validator" python -m mlcast_dataset_validator.specs.source_data.radar_precipitation --s3-endpoint-url https://object-store.os-api.cci2.ecmwf.int --s3-anon s3://mlcast-source-datasets/radklim/v0.1.0/5_minutes.zarr/
```

Or you can of course clone the repository and run it directly:

```bash
git clone
cd mlcast-sourcedata-validator
python -m mlcast_dataset_validator.specs.source_data.radar_precipitation /path/to/zarr/file.zarr
```
