import xarray as xr

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.2"


@log_function_call
def check_license(
    ds: xr.Dataset,
    require_spdx: bool,
    recommended: list[str],
    warn_on_restricted: list[str],
) -> ValidationReport:
    """
    Validate the licensing information in the dataset.

    Parameters:
        ds (xr.Dataset): The dataset to validate.
        require_spdx (bool): Whether a valid SPDX identifier is required.
        recommended (list[str]): List of recommended licenses.
        warn_on_restricted (list[str]): List of restricted licenses that should generate warnings.

    Returns:
        ValidationReport: A report containing the results of the licensing validation checks.
    """
    report = ValidationReport()

    if "license" not in ds.attrs:
        report.add(
            SECTION_ID,
            "License metadata",
            "FAIL",
            "Missing required 'license' global attribute",
        )
        return report

    license_id = ds.attrs["license"].strip()
    if license_id in recommended:
        report.add(
            SECTION_ID,
            "License compliance",
            "PASS",
            f"License '{license_id}' is recommended and accepted",
        )
    elif license_id in warn_on_restricted:
        report.add(
            SECTION_ID,
            "License compliance",
            "WARNING",
            f"License '{license_id}' has restrictions (NC/ND)",
        )
    else:
        report.add(
            SECTION_ID,
            "License compliance",
            "WARNING",
            f"License '{license_id}' requires case-by-case review",
        )

    return report
