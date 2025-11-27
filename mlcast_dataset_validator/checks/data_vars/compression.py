from typing import Sequence

import xarray as xr

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.2"


def get_compressor_name(da: xr.DataArray) -> str:
    """
    Return the name of the compressor used for a Zarr-backed xarray.DataArray.

    This function handles both Zarr v2 datasets written directly by xarray/zarr
    (where compression appears in `da.encoding["compressor"]`) and datasets
    converted from NetCDF4/HDF5 sources (where compression is stored in
    `da.encoding["filters"]` as e.g. `Zlib(level=5)`).

    The distinction arises because:
      - Zarr natively stores a single 'compressor' codec (e.g. Blosc, Zstd, etc.).
      - HDF5-based formats (NetCDF4) instead use a *pipeline* of filters,
        where the final step may itself be a compressor (e.g. Shuffle + Zlib).

    Parameters
    ----------
    da : xarray.DataArray
        A DataArray opened from a Zarr or NetCDF source.

    Returns
    -------
    str
        Name of the compressor (e.g. "zstd", "zlib", "blosc") or "none"
        if no compression is applied.
    """
    enc = getattr(da, "encoding", {}) or {}

    # First, check for native Zarr compressor
    comp = enc.get("compressor") or enc.get("compressors")
    if comp:
        name = getattr(comp, "codec_id", comp.__class__.__name__).lower()
        return name

    # Otherwise, fall back to inspecting filters (NetCDF/HDF5-style)
    filters = enc.get("filters") or []
    for f in filters:
        name = getattr(f, "codec_id", f.__class__.__name__).lower()
        if any(
            k in name for k in ("zlib", "gzip", "bz2", "blosc", "zstd", "lz4", "snappy")
        ):
            return name

    # No compression found
    return "none"


@log_function_call
def check_compression(
    ds: xr.Dataset,
    *,
    require_compression: bool,
    recommended_compression: str,
    allow_coord_algs: Sequence[str],
) -> ValidationReport:
    """Check compression requirements."""
    report = ValidationReport()

    # Find all grid_mapping data variables since we don't want to check those
    grid_mapping_vars = set()
    for var in ds.data_vars:
        if "grid_mapping" in ds[var].attrs:
            grid_mapping_vars.add(ds[var].attrs["grid_mapping"])
    data_vars = set(ds.data_vars) - grid_mapping_vars

    for data_var in data_vars:
        da = ds[data_var]
        compressor = get_compressor_name(da)

        if require_compression:
            if compressor is None:
                report.add(
                    SECTION_ID,
                    f"{da.name} array compression for {data_var}",
                    "FAIL",
                    f"{da.name} data array does not use compression",
                )
            elif compressor == recommended_compression:
                report.add(
                    SECTION_ID,
                    f"{da.name} array compression for {data_var}",
                    "PASS",
                    f"{da.name} array uses recommended compression: {recommended_compression}",
                )
            else:
                report.add(
                    SECTION_ID,
                    f"{da.name} array compression for {data_var}",
                    "WARNING",
                    f"{da.name} array uses compression: {compressor}, "
                    f"recommended is {recommended_compression}",
                )
        else:
            if compressor is None:
                if require_compression:
                    report.add(
                        SECTION_ID,
                        f"{da.name} array compression for {data_var}",
                        "FAIL",
                        f"{da.name} data array does not use compression",
                    )
                elif compressor == recommended_compression:
                    report.add(
                        SECTION_ID,
                        f"{da.name} array compression for {data_var}",
                        "PASS",
                        f"{da.name} array uses recommended compression: {recommended_compression}",
                    )
                else:
                    pass

    return report
