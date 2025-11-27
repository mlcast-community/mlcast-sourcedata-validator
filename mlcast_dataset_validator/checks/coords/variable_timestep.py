from typing import Dict, Tuple

import pandas as pd
import xarray as xr

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.5"

_TIMESTEP_CACHE: Dict[int, Tuple[bool, int]] = {}


def analyze_dataset_timesteps(ds: xr.Dataset) -> tuple[bool, int]:
    """
    Analyze the dataset's time coordinate spacing.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing a `time` coordinate whose spacing should be inspected.

    Returns
    -------
    tuple[bool, int]
        Tuple where the first element indicates whether variable intervals exist
        and the second reports how many unique timestep differences were observed.

    Notes
    -----
    Results are cached using the dataset object's Python id to avoid redundant
    computations when the same dataset instance is analyzed multiple times.
    """
    cache_key = id(ds)
    cached = _TIMESTEP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    time_index = pd.to_datetime(ds.time.values)
    try:
        raw_values = time_index.asi8
    except AttributeError:
        raw_values = pd.Series(time_index).astype("int64").to_numpy()

    if len(raw_values) < 2:
        result = (False, 0)
    else:
        unique_diffs = {
            int(raw_values[idx + 1]) - int(raw_values[idx])
            for idx in range(len(raw_values) - 1)
        }
        unique_diff_count = len(unique_diffs)
        result = (unique_diff_count > 1, unique_diff_count)

    _TIMESTEP_CACHE[cache_key] = result
    return result


@log_function_call
def check_variable_timestep(
    ds: xr.Dataset,
    *,
    allow_variable_timestep: bool,
) -> ValidationReport:
    """
    Validate whether the dataset's timestep pattern satisfies the specification.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to evaluate.
    allow_variable_timestep : bool
        Whether multiple unique intervals are allowed by the spec variant.

    Returns
    -------
    ValidationReport
        Populated report describing the outcome of each timestep-related rule.
    """
    report = ValidationReport()

    if "time" not in ds.coords:
        report.add(
            SECTION_ID, "Time coordinate presence", "FAIL", "Missing 'time' coordinate"
        )
        return report

    try:
        has_variable, unique_diff_count = analyze_dataset_timesteps(ds)
    except Exception as e:
        report.add(
            SECTION_ID,
            "Variable timestep analysis",
            "FAIL",
            f"Failed to analyze variable timesteps: {e}",
        )
        return report

    if not has_variable:
        report.add(
            SECTION_ID,
            "Timestep consistency",
            "PASS",
            "Timestep is consistent throughout the dataset",
        )
        return report

    if allow_variable_timestep:
        report.add(
            SECTION_ID,
            "Variable timestep handling",
            "PASS",
            f"Variable timesteps detected with {unique_diff_count} unique intervals",
        )
    else:
        report.add(
            SECTION_ID,
            "Variable timestep handling",
            "FAIL",
            "Variable timesteps detected but not allowed",
        )

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

    return report
