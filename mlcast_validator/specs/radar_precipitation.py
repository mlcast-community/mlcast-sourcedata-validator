#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
## 1. Introduction

This document specifies the requirements for 2D radar precipitation and reflectivity composite datasets to be included in the MLCast data collection. The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in RFC 2119.

## 2. Scope

This specification applies to 2D radar composite datasets (merged from multiple radar sources) intended for machine learning applications in weather and climate research. Single-radar datasets are explicitly excluded from this specification.
"""

import argparse
from typing import List, Optional

import xarray as xr
from loguru import logger

from ..checks.coords.future_extension import check_future_timestep
from ..checks.coords.names import check_coordinate_names
from ..checks.coords.spatial import check_spatial_requirements
from ..checks.coords.temporal import check_temporal_requirements
from ..checks.coords.variable_timestep import check_variable_timestep
from ..checks.data_vars.chunking import check_chunking_strategy
from ..checks.data_vars.compression import check_compression
from ..checks.data_vars.data_structure import check_data_structure
from ..checks.data_vars.georeferencing import check_georeferencing
from ..checks.global_attributes.attributes import check_global_attributes
from ..checks.global_attributes.licensing import check_license
from ..checks.global_attributes.zarr_format import check_zarr_format
from .base import ValidationReport


# -------------------------
# Core public API
# -------------------------
def validate_dataset(
    path: str, storage_options: Optional[dict] = None
) -> ValidationReport:
    """Validate a radar precipitation dataset against the MLCast specification."""
    report = ValidationReport()

    # Load dataset
    ds = xr.open_zarr(path, storage_options=storage_options)
    logger.info(f"Opened dataset at {path}")
    logger.info(str(ds))

    # --- 3.1 Coordinate Variables ---
    # > "Coordinate variable names MUST follow CF conventions and use the following names: `x`, `y`, `lat`, `lon`, `time`."
    # > "All coordinate variables SHOULD include CF-compliant attributes (`long_name`, `standard_name` and `units`)."
    report += check_coordinate_names(
        ds,
        required_names=["x", "y", "lat", "lon", "time"],
        require_cf_attrs=["long_name", "standard_name", "units"],
    )

    # --- 3.2 Future Timestep Extension ---
    # > "Future timesteps MUST have regular timestepping corresponding to the highest (most recent) frequency present in the data."
    # > "Future timesteps MUST NOT extend beyond the year 2050."
    # > "A global attribute named `last_valid_timestep` MUST be present to indicate the most recent non-NaN filled timestep."
    report += check_future_timestep(
        ds,
        max_year=2050,
    )

    # --- 3.3 Spatial Requirements ---
    # > "The dataset MUST provide 2D radar composites with a spatial resolution of 1 kilometer or finer."
    # > "The valid sensing area MUST support at least one 256×256 pixel square crop that is fully contained within the radar sensing range."
    # > "The spatial domain, including resolution, size, and geographical coverage, MUST remain constant across all timesteps in the archive."
    report += check_spatial_requirements(
        ds,
        max_resolution_km=1.0,
        min_crop_size=(256, 256),
        require_constant_domain=True,
    )

    # --- 3.4 Temporal Requirements ---
    # > "The dataset MUST contain a minimum of 3 years of continuous temporal coverage."
    # > "The timestep MAY be variable throughout the archive."
    report += check_temporal_requirements(
        ds,
        min_years=3,
        allow_variable_timestep=True,
    )

    # --- 3.5 Variable Timestep Handling ---
    # > "If the archive contains variable timesteps, the timesteps SHOULD follow the natural timestepping of the data collection."
    # > "A global attribute named `consistent_timestep_start` MAY be included to indicate the first timestamp where regular timestepping begins."
    report += check_variable_timestep(
        ds,
        allow_variable_timestep=True,
    )

    # --- 4.1 Chunking Strategy ---
    # > "The dataset MUST use a chunking strategy of 1 × height × width (one chunk per timestep)."
    report += check_chunking_strategy(
        ds,
        time_chunksize=1,
    )

    # --- 4.2 Compression ---
    # > "The main data arrays MUST use compression to reduce storage requirements."
    # > "ZSTD compression is RECOMMENDED for optimal performance of the main data arrays."
    # > "Coordinate arrays MAY use different compression algorithms (e.g., lz4) as appropriate."
    report += check_compression(
        ds,
        require_compression=True,
        recommended_compression="zstd",
        allow_coord_algs=["lz4"],
    )

    # --- 4.3 Data Structure ---
    # > "The main data variable MUST be encoded with dimensions in the order: time × height (y, lat) × width (x, lon)."
    # > "The data type MUST be floating-point (float16, float32, or float64)."
    report += check_data_structure(
        ds,
        dim_order=("time", "y", "x"),
        allowed_dtypes=["float16", "float32", "float64"],
    )

    # --- 4.4 Data Variable Naming and Attributes ---
    # > "The data variable name SHOULD be a CF convention standard name or use a sensible name from the ECMWF parameter database."
    # > "The data variable MUST include the `long_name`, `standard_name` and `units` attributes following CF conventions."
    # TODO: there is an inconsistency in the spec document here, where to we say what standard names are allowed?
    # report += check_variable_units(
    #     ds,
    #     allowed_units={
    #         "precip_rate": ["kg m-2 h-1", "mm h-1", "mm/h"],
    #         "reflectivity": ["dBZ"],
    #         "precip_amount": ["kg m-2", "mm"],
    #     },
    # )

    # --- 4.5 Georeferencing ---
    # > "The dataset MUST include proper georeferencing information following the GeoZarr specification."
    # > "The data variable MUST include a `grid_mapping` attribute that references the coordinate reference system (crs) variable."
    # > "The crs variable MUST include both a `spatial_ref` and a `crs_wkt` attribute with a WKT string."
    report += check_georeferencing(
        ds,
        require_geozarr=True,
        require_grid_mapping=True,
        crs_attrs=["spatial_ref", "crs_wkt"],
        require_bbox=True,
    )

    # --- 5.1 Global Attributes ---
    # > "The following global attributes are REQUIRED: `license`."
    # > "The following global attributes are CONDITIONAL: `consistent_timestep_start`, `last_valid_timestep`."
    report += check_global_attributes(
        ds,
        required_attrs=["license"],
        conditional_attrs=["consistent_timestep_start", "last_valid_timestep"],
    )

    # --- 5.2 Licensing Requirements ---
    # > "The dataset MUST include a global `license` attribute containing a valid SPDX identifier."
    # > "The following licenses are RECOMMENDED: `CC-BY`, `CC-BY-SA`, `OGL`."
    # > "Licenses with `NC` or `ND` restrictions SHOULD generate warnings but MAY be accepted after review."
    report += check_license(
        ds,
        require_spdx=True,
        recommended=["CC-BY", "CC-BY-SA", "OGL"],
        warn_on_restricted=["NC", "ND"],
    )

    # --- 5.3 Zarr Format ---
    # > "The dataset MUST use Zarr version 2 or version 3 format."
    # > "If Zarr version 2 is used, the dataset MUST include consolidated metadata."
    report += check_zarr_format(
        ds,
        allowed_versions=[2, 3],
        require_consolidated_if_v2=True,
        storage_options=storage_options,
    )

    # --- 6. Missing Data and Special Values ---
    # > "NaN values MUST be used to indicate: Pixels outside the radar sensing range, Pixels blocked by orography, Partially or totally missing scans."
    # XXX: We don't check for the presense of NaNs because that would require
    # us to check all values in the dataset (i.e. fetch the whole dataset) and
    # as only float types are allowed the use of NaNs is implied.
    # TODO: should we check for other special values here?

    return report


# -------------------------
# CLI
# -------------------------
@logger.catch
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a 2D radar composite Zarr dataset against the MLCast specification."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the Zarr dataset to validate.",
    )
    parser.add_argument(
        "--s3-endpoint-url",
        type=str,
        default=None,
        help="Optional S3 endpoint URL for accessing the Zarr dataset.",
    )
    parser.add_argument(
        "--s3-anon",
        action="store_true",
        help="Use anonymous access for S3 storage.",
    )
    args = parser.parse_args(argv)

    storage_options = {}
    if args.s3_endpoint_url:
        storage_options["endpoint_url"] = args.s3_endpoint_url
    if args.s3_anon:
        storage_options["anon"] = True

    # storage_options must default to None if not set, as some backends
    # (e.g., local filesystem) do not accept an empty dict.
    report = validate_dataset(
        args.dataset_path, storage_options=storage_options or None
    )
    report.console_print()

    if report.has_fails():
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
