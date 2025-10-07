from typing import Sequence

import xarray as xr

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call


@log_function_call
def check_zarr_format(
    ds: xr.Dataset,
    *,
    allowed_versions: Sequence[int],
    require_consolidated_if_v2: bool,
) -> ValidationReport:
    """Check Zarr format requirements."""
    report = ValidationReport()

    zarr_format = getattr(ds, "zarr_format", 2)  # Default to Zarr v2
    if zarr_format in allowed_versions:
        report.add(
            "5.1",
            "Zarr version compatibility",
            "PASS",
            f"Using supported Zarr v{zarr_format} format",
        )
    else:
        report.add(
            "5.1",
            "Zarr version compatibility",
            "FAIL",
            f"Unsupported Zarr version: v{zarr_format}",
        )

    if zarr_format == 2 and require_consolidated_if_v2:
        if hasattr(ds, "consolidated") and ds.consolidated:
            report.add(
                "5.1",
                "Consolidated metadata presence",
                "PASS",
                "Zarr v2 dataset has consolidated metadata",
            )
        else:
            report.add(
                "5.1",
                "Consolidated metadata presence",
                "FAIL",
                "Zarr v2 dataset is missing consolidated metadata",
            )

    return report
