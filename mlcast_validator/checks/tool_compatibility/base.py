import xarray as xr

from ...specs.base import ValidationReport
from . import cartopy as cartopy_checker
from . import gdal as gdal_checker
from . import xarray as xarray_checker


def check_tool_compatibility(ds: xr.Dataset, tools: list[str]) -> ValidationReport:
    """
    Validate that the dataset is compatible with the specified tools.

    Parameters:
        ds: xarray.Dataset
            The dataset to validate.
        tools: list of str
            List of tools to check compatibility with (e.g., "xarray", "GDAL", "cartopy").

    Returns:
        ValidationReport
            A report indicating the results of the compatibility checks.
    """
    report = ValidationReport()

    for tool in tools:
        if tool == "GDAL":
            report += gdal_checker.check_gdal_compatibility(ds=ds)
        elif tool == "xarray":
            report += xarray_checker.check_xarray_compatibility(ds=ds)
        elif tool == "cartopy":
            report += cartopy_checker.check_cartopy_compatibility(ds=ds)
        else:
            raise NotImplementedError(
                f"Compatibility check for {tool} is not implemented."
            )

    return report
