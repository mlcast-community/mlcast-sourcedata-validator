#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
## 1. Introduction

This document specifies the requirements for 2D radar precipitation and reflectivity composite datasets to be included in the MLCast data collection. The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in RFC 2119.

## 2. Scope

This specification applies to 2D radar composite datasets (merged from multiple radar sources) intended for machine learning applications in weather and climate research. Single-radar datasets are explicitly excluded from this specification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Dict


# -------------------------
# Data structures
# -------------------------
@dataclass
class Issue:
    section: str
    requirement: str
    level: str  # "ERROR" | "WARNING" | "INFO"
    detail: str = ""


@dataclass
class ValidationReport:
    ok: bool = True
    issues: List[Issue] = field(default_factory=list)

    def add(self, section: str, requirement: str, level: str, detail: str = "") -> None:
        self.issues.append(Issue(section, requirement, level, detail))

    def summarize(self) -> str:
        errs = sum(1 for i in self.issues if i.level == "ERROR")
        warns = sum(1 for i in self.issues if i.level == "WARNING")
        infos = sum(1 for i in self.issues if i.level == "INFO")
        return f"Summary: {errs} error(s), {warns} warning(s), {infos} info note(s)."

    def __iadd__(self, other: "ValidationReport") -> "ValidationReport":
        self.issues.extend(other.issues)
        self.ok = self.ok and other.ok
        return self

    def __add__(self, other: "ValidationReport") -> "ValidationReport":
        out = ValidationReport(ok=self.ok and other.ok)
        out.issues = [*self.issues, *other.issues]
        return out


# -------------------------
# Core public API
# -------------------------
def validate_dataset(ds_path: Path) -> ValidationReport:
    report = ValidationReport()

    # --- 2. Scope ---
    report += check_scope(
        ds_path,
        require_multi_radar=True,
    )

    # --- 3.1 Spatial Requirements ---
    report += check_spatial_requirements(
        ds_path,
        max_resolution_km=1.0,
        min_crop_size=(256, 256),
        require_constant_domain=True,
    )

    # --- 3.2 Temporal Requirements ---
    report += check_temporal_requirements(
        ds_path,
        min_years=3,
        allow_variable_timestep=True,
    )

    # --- 3.3 Data Variable Requirements ---
    report += check_variable_units(
        ds_path,
        allowed_units={
            "precip_amount": ["mm", "kg m-2"],
            "precip_rate": ["mm/h", "mm h-1", "kg m-2 h-1"],
            "reflectivity": ["dBZ"],
        },
    )

    # --- 4. Licensing Requirements ---
    report += check_license(
        ds_path,
        require_spdx=True,
        recommended=["CC-BY", "CC-BY-SA", "OGL"],
        warn_on_restricted=["NC", "ND"],
    )

    # --- 5.1 Zarr Format ---
    report += check_zarr_format(
        ds_path,
        allowed_versions=[2, 3],
        require_consolidated_if_v2=True,
    )

    # --- 5.2 Compression ---
    report += check_compression(
        ds_path,
        require_compression=True,
        recommended_main="zstd",
        allow_coord_algs=["lz4"],
    )

    # --- 5.3 Georeferencing ---
    report += check_georeferencing(
        ds_path,
        require_geozarr=True,
        require_grid_mapping=True,
        crs_attrs=["spatial_ref", "crs_wkt"],
        require_bbox=True,
    )

    # --- 5.4 Data Structure ---
    report += check_data_structure(
        ds_path,
        dim_order=("time", "y", "x"),
        allowed_dtypes=["float16", "float32", "float64"],
    )

    # --- 5.5 Coordinate Variables ---
    report += check_coordinate_variables(
        ds_path,
        required_names=["x", "y", "lat", "lon", "time"],
        require_cf_attrs=["long_name", "standard_name", "units"],
    )

    # --- 5.6 Data Variable Naming and attributes ---
    report += check_variable_naming_and_attributes(
        ds_path,
        required_attrs=["long_name", "standard_name", "units"],
        allowed_names={
            "precip_rate": ["mmh", "rr", "tprate", "prate", "rain_rate", "rainfall_flux", "rainfall_rate"],
            "reflectivity": ["equivalent_reflectivity_factor", "dbz", "rare"],
            "precip_amount": ["rainfall_amount", "mm", "precipitation_amount", "tp"],
        },
        allowed_units={
            "precip_rate": ["kg m-2 h-1", "mm h-1", "mm/h"],
            "reflectivity": ["dBZ"],
            "precip_amount": ["kg m-2", "mm"],
        },
    )

    # --- 5.7 Chunking Strategy ---
    report += check_chunking_strategy(
        ds_path,
        time_chunksize=1,
    )

    report.ok = not any(i.level == "ERROR" for i in report.issues)
    return report


# -------------------------
# Check stubs with kwargs
# -------------------------
def check_scope(ds_path: Path, *, require_multi_radar: bool) -> ValidationReport:
    report = ValidationReport()
    # TODO: enforce multi-radar if require_multi_radar
    return report


def check_spatial_requirements(
    ds_path: Path,
    *,
    max_resolution_km: float,
    min_crop_size: tuple[int, int],
    require_constant_domain: bool,
) -> ValidationReport:
    report = ValidationReport()
    # TODO: use kwargs to check spatial resolution, crop size, domain constancy
    return report


def check_temporal_requirements(
    ds_path: Path,
    *,
    min_years: int,
    allow_variable_timestep: bool,
) -> ValidationReport:
    report = ValidationReport()
    # TODO: use kwargs to check years of coverage, timestep variability
    return report


def check_variable_units(
    ds_path: Path,
    *,
    allowed_units: Dict[str, Sequence[str]],
) -> ValidationReport:
    report = ValidationReport()
    # TODO: validate units against allowed_units per variable class
    return report


def check_license(
    ds_path: Path,
    *,
    require_spdx: bool,
    recommended: Sequence[str],
    warn_on_restricted: Sequence[str],
) -> ValidationReport:
    report = ValidationReport()
    # TODO: use kwargs for license checks
    return report


def check_zarr_format(
    ds_path: Path,
    *,
    allowed_versions: Sequence[int],
    require_consolidated_if_v2: bool,
) -> ValidationReport:
    report = ValidationReport()
    # TODO: enforce Zarr version and consolidated metadata
    return report


def check_compression(
    ds_path: Path,
    *,
    require_compression: bool,
    recommended_main: str,
    allow_coord_algs: Sequence[str],
) -> ValidationReport:
    report = ValidationReport()
    # TODO: compression validation per kwargs
    return report


def check_georeferencing(
    ds_path: Path,
    *,
    require_geozarr: bool,
    require_grid_mapping: bool,
    crs_attrs: Sequence[str],
    require_bbox: bool,
) -> ValidationReport:
    report = ValidationReport()
    # TODO: georeferencing checks per kwargs
    return report


def check_data_structure(
    ds_path: Path,
    *,
    dim_order: Sequence[str],
    allowed_dtypes: Sequence[str],
) -> ValidationReport:
    report = ValidationReport()
    # TODO: data structure checks
    return report


def check_coordinate_variables(
    ds_path: Path,
    *,
    required_names: Sequence[str],
    require_cf_attrs: Sequence[str],
) -> ValidationReport:
    report = ValidationReport()
    # TODO: coordinate variable checks
    return report


def check_variable_naming_and_attributes(
    ds_path: Path,
    *,
    required_attrs: Sequence[str],
    allowed_names: Dict[str, Sequence[str]],
    allowed_units: Dict[str, Sequence[str]],
) -> ValidationReport:
    report = ValidationReport()
    # TODO: variable naming and attribute checks
    return report


def check_chunking_strategy(
    ds_path: Path,
    *,
    time_chunksize: int,
) -> ValidationReport:
    report = ValidationReport()
    # TODO: chunking strategy checks
    return report


# -------------------------
# CLI
# -------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a 2D radar composite Zarr dataset against the MLCast specification."
    )
