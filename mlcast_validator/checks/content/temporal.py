import pandas as pd
import xarray as xr

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call


@log_function_call
def check_temporal_requirements(
    ds: xr.Dataset,
    min_years: int,
    allow_variable_timestep: bool,
) -> ValidationReport:
    """
    Validate temporal requirements for the dataset.

    Parameters:
        ds (xr.Dataset): The dataset to validate.
        min_years (int): Minimum required years of continuous temporal coverage.
        allow_variable_timestep (bool): Whether variable timesteps are allowed.

    Returns:
        ValidationReport: A report containing the results of the temporal validation checks.
    """
    report = ValidationReport()

    if "time" not in ds.coords:
        report.add(
            "3.2", "Time coordinate presence", "FAIL", "Missing 'time' coordinate"
        )
        return report

    try:
        time_coord = pd.to_datetime(ds.time.values)
        time_range = time_coord[-1] - time_coord[0]
        years = time_range.days / 365.25
        if years >= min_years:
            report.add(
                "3.2",
                "Minimum 3-year coverage",
                "PASS",
                f"Temporal coverage: {years:.1f} years (≥{min_years} years)",
            )
        else:
            report.add(
                "3.2",
                "Minimum 3-year coverage",
                "FAIL",
                f"Temporal coverage: {years:.1f} years (<{min_years} years required)",
            )
    except Exception as e:
        report.add(
            "3.2",
            "Temporal coverage analysis",
            "FAIL",
            f"Failed to analyze temporal coverage: {e}",
        )

    return report
