import pandas as pd
import xarray as xr

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.5"


@log_function_call
def check_variable_timestep(
    ds: xr.Dataset,
    *,
    allow_variable_timestep: bool,
) -> ValidationReport:
    """Check variable timestep handling."""
    report = ValidationReport()

    if "time" not in ds.coords:
        report.add(
            SECTION_ID, "Time coordinate presence", "FAIL", "Missing 'time' coordinate"
        )
        return report

    try:
        time_coord = pd.to_datetime(ds.time.values)
        time_diffs = pd.Series(time_coord).diff().dropna()
        unique_diffs = time_diffs.value_counts()

        if len(unique_diffs) == 1:
            report.add(
                SECTION_ID,
                "Timestep consistency",
                "PASS",
                "Timestep is consistent throughout the dataset",
            )
        else:
            if allow_variable_timestep:
                report.add(
                    SECTION_ID,
                    "Variable timestep handling",
                    "PASS",
                    f"Variable timesteps detected with {len(unique_diffs)} unique intervals",
                )
            else:
                report.add(
                    SECTION_ID,
                    "Variable timestep handling",
                    "FAIL",
                    "Variable timesteps detected but not allowed",
                )

            # Check for `consistent_timestep_start` attribute
            if "consistent_timestep_start" in ds.attrs:
                report.add(
                    SECTION_ID,
                    "Consistent timestep start metadata",
                    "PASS",
                    "Dataset includes 'consistent_timestep_start' metadata",
                )
            else:
                report.add(
                    SECTION_ID,
                    "Consistent timestep start metadata",
                    "WARNING",
                    "Dataset has variable timesteps but is missing 'consistent_timestep_start' metadata",
                )
    except Exception as e:
        report.add(
            SECTION_ID,
            "Variable timestep analysis",
            "FAIL",
            f"Failed to analyze variable timesteps: {e}",
        )

    return report
