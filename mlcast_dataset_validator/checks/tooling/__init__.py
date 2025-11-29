from __future__ import annotations

from typing import Iterator, Optional, Tuple

import xarray as xr

__all__ = [
    "grid_mapping_definitions",
    "iter_data_vars",
    "select_data_var",
]


def grid_mapping_definitions(ds: xr.Dataset) -> set[str]:
    """Return names of variables acting as CF grid_mapping definitions."""
    gm_vars: set[str] = set()
    for data_array in ds.data_vars.values():
        mapping_name = data_array.attrs.get("grid_mapping")
        if mapping_name and mapping_name in ds.variables:
            gm_vars.add(mapping_name)
    return gm_vars


def iter_data_vars(
    ds: xr.Dataset, *, require_grid_mapping: bool = False
) -> Iterator[Tuple[str, xr.DataArray]]:
    """Yield data variables, skipping grid_mapping definition variables."""
    gm_vars = grid_mapping_definitions(ds)
    for name, data_array in ds.data_vars.items():
        if name in gm_vars:
            continue
        if require_grid_mapping and "grid_mapping" not in data_array.attrs:
            continue
        yield name, data_array


def select_data_var(
    ds: xr.Dataset,
    preferred: Optional[str] = None,
    *,
    require_grid_mapping: bool = False,
) -> Optional[str]:
    """Select a representative data variable respecting grid_mapping filters."""
    gm_vars = grid_mapping_definitions(ds)
    if preferred:
        if preferred in ds.data_vars and preferred not in gm_vars:
            if not require_grid_mapping or "grid_mapping" in ds[preferred].attrs:
                return preferred
        return None

    for name, _ in iter_data_vars(ds, require_grid_mapping=require_grid_mapping):
        return name
    return None
