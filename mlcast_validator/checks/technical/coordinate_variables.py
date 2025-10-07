from typing import Sequence

import xarray as xr

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call


@log_function_call
def check_coordinate_variables(
    ds: xr.Dataset,
    *,
    required_names: Sequence[str],
    require_cf_attrs: Sequence[str],
) -> ValidationReport:
    """Check coordinate variable requirements."""
    report = ValidationReport()

    for coord in required_names:
        if coord in ds.coords:
            coord_var = ds.coords[coord]
            report.add(
                "5.5",
                f"Coordinate variable '{coord}' presence",
                "PASS",
                f"Coordinate variable '{coord}' is present",
            )

            # Check CF-compliant attributes
            missing_attrs = [
                attr for attr in require_cf_attrs if attr not in coord_var.attrs
            ]
            if not missing_attrs:
                report.add(
                    "5.5",
                    f"CF attributes for '{coord}'",
                    "PASS",
                    f"Coordinate variable '{coord}' has all required CF attributes",
                )
            else:
                report.add(
                    "5.5",
                    f"CF attributes for '{coord}'",
                    "FAIL",
                    f"Coordinate variable '{coord}' is missing CF attributes: {missing_attrs}",
                )
        else:
            report.add(
                "5.5",
                f"Coordinate variable '{coord}' presence",
                "FAIL",
                f"Coordinate variable '{coord}' is missing",
            )

    return report
