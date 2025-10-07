from typing import Sequence

import xarray as xr

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call


@log_function_call
def check_compression(
    ds: xr.Dataset,
    *,
    require_compression: bool,
    recommended_main: str,
    allow_coord_algs: Sequence[str],
) -> ValidationReport:
    """Check compression requirements."""
    report = ValidationReport()

    for data_var in ds.data_vars:
        data_array = ds[data_var]
        compressor = getattr(data_array.encoding, "compressor", None)
        if compressor:
            report.add(
                "5.2",
                f"Compression for {data_var}",
                "PASS",
                f"Data variable '{data_var}' uses compression: {compressor}",
            )
            if recommended_main in str(compressor).lower():
                report.add(
                    "5.2",
                    f"Recommended compression for {data_var}",
                    "PASS",
                    f"Data variable '{data_var}' uses recommended compression: {recommended_main}",
                )
        else:
            if require_compression:
                report.add(
                    "5.2",
                    f"Compression for {data_var}",
                    "FAIL",
                    f"Data variable '{data_var}' does not use compression",
                )
            else:
                report.add(
                    "5.2",
                    f"Compression for {data_var}",
                    "WARNING",
                    f"Data variable '{data_var}' does not use compression",
                )

    return report
