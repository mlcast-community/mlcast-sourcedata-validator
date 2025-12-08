import pandas as pd
import xarray as xr

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.2"


@log_function_call
def check_future_timestep(
    ds: xr.Dataset,
    *,
    max_year: int,
) -> ValidationReport:
    """Check future timestep extension."""
    report = ValidationReport()

    if "time" not in ds.coords:
        report.add(
            SECTION_ID, "Time coordinate presence", "FAIL", "Missing 'time' coordinate"
        )
        return report

    try:
        current_time = pd.Timestamp.now()
        time_coord = pd.to_datetime(ds.time.values)
        max_time = time_coord.max()

        if max_time > current_time:
            report.add(
                SECTION_ID,
                "Future timesteps detected",
                "PASS",
                f"Dataset includes future timesteps up to {max_time}",
            )

            # Check for max_year limit
            if max_time.year > max_year:
                report.add(
                    SECTION_ID,
                    "Future timestep limit",
                    "FAIL",
                    f"Future timesteps extend beyond {max_year}: {max_time.year}",
                )
    except Exception as e:
        report.add(
            SECTION_ID,
            "Future timestep analysis",
            "FAIL",
            f"Failed to analyze future timesteps: {e}",
        )

    return report
