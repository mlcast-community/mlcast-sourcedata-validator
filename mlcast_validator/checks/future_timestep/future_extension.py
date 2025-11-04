import pandas as pd
import xarray as xr

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call


@log_function_call
def check_future_timestep(
    ds: xr.Dataset,
    *,
    max_year: int,
) -> ValidationReport:
    """Check future timestep extension."""
    report = ValidationReport()

    if "time" not in ds.coords:
        report.add("8", "Time coordinate presence", "FAIL", "Missing 'time' coordinate")
        return report

    try:
        current_time = pd.Timestamp.now()
        time_coord = pd.to_datetime(ds.time.values)
        max_time = time_coord.max()

        if max_time > current_time:
            report.add(
                "8",
                "Future timesteps detected",
                "PASS",
                f"Dataset includes future timesteps up to {max_time}",
            )

            # Check for max_year limit
            if max_time.year > max_year:
                report.add(
                    "8",
                    "Future timestep limit",
                    "FAIL",
                    f"Future timesteps extend beyond {max_year}: {max_time.year}",
                )

            # Check for `last_valid_timestep` attribute
            if "last_valid_timestep" in ds.attrs:
                report.add(
                    "8",
                    "Last valid timestep metadata",
                    "PASS",
                    "Dataset includes 'last_valid_timestep' metadata",
                )
            else:
                report.add(
                    "8",
                    "Last valid timestep metadata",
                    "FAIL",
                    "Dataset is missing 'last_valid_timestep' metadata for future timesteps",
                )
    except Exception as e:
        report.add(
            "8",
            "Future timestep analysis",
            "FAIL",
            f"Failed to analyze future timesteps: {e}",
        )

    return report
