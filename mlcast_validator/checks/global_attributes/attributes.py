from typing import Sequence

import xarray as xr

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.1"


@log_function_call
def check_global_attributes(
    ds: xr.Dataset,
    *,
    required_attrs: Sequence[str] = (),
    conditional_attrs: Sequence[str] = (),
) -> ValidationReport:
    """Check global attributes."""
    report = ValidationReport()

    # Check required attributes
    for attr in required_attrs:
        if attr in ds.attrs:
            report.add(
                SECTION_ID,
                f"Required global attribute '{attr}'",
                "PASS",
                f"Global attribute '{attr}' is present",
            )
        else:
            report.add(
                SECTION_ID,
                f"Required global attribute '{attr}'",
                "FAIL",
                f"Global attribute '{attr}' is missing",
            )

    # Check conditional attributes
    for attr in conditional_attrs:
        if attr in ds.attrs:
            report.add(
                SECTION_ID,
                f"Conditional global attribute '{attr}'",
                "PASS",
                f"Global attribute '{attr}' is present",
            )
        else:
            report.add(
                SECTION_ID,
                f"Conditional global attribute '{attr}'",
                "WARNING",
                f"Global attribute '{attr}' is not present (optional)",
            )

    return report
