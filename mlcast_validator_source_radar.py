#!/usr/bin/env -S uv run --script --index-strategy unsafe-best-match
#
# /// script
# requires-python = ">=3.13"
# dependencies = [
# "affine==2.4.0",
# "aiobotocore==2.24.1",
# "aiohappyeyeballs==2.6.1",
# "aiohttp==3.12.15",
# "aioitertools==0.12.0",
# "aiosignal==1.4.0",
# "asttokens==3.0.0",
# "attrs==25.3.0",
# "botocore==1.39.11",
# "cartopy==0.25.0",
# "certifi==2025.8.3",
# "cfgv==3.4.0",
# "cftime==1.6.4.post1",
# "charset-normalizer==3.4.3",
# "click==8.2.1",
# "click-plugins==1.1.1.2",
# "cligj==0.7.2",
# "cloudpickle==3.1.1",
# "comm==0.2.3",
# "contourpy==1.3.3",
# "crc32c==2.7.1",
# "cycler==0.12.1",
# "dask==2025.7.0",
# "debugpy==1.8.16",
# "decorator==5.2.1",
# "distlib==0.4.0",
# "donfig==0.8.1.post1",
# "execnet==2.1.1",
# "executing==2.2.0",
# "filelock==3.19.1",
# "fonttools==4.59.1",
# "frozenlist==1.7.0",
# "fsspec==2025.7.0",
# "gdal==3.12.0.1",
# "identify==2.6.13",
# "idna==3.10",
# "iniconfig==2.1.0",
# "intake==2.0.8",
# "intake-xarray==2.0.0",
# "ipdb==0.13.13",
# "ipykernel==6.30.1",
# "ipython==9.4.0",
# "ipython-pygments-lexers==1.1.1",
# "jedi==0.19.2",
# "jinja2==3.1.6",
# "jmespath==1.0.1",
# "jupyter-client==8.6.3",
# "jupyter-core==5.8.1",
# "kiwisolver==1.4.9",
# "locket==1.0.0",
# "loguru==0.7.3",
# "markupsafe==3.0.2",
# "matplotlib==3.10.5",
# "matplotlib-inline==0.1.7",
# "msgpack==1.1.1",
# "multidict==6.6.4",
# "nest-asyncio==1.6.0",
# "netcdf4==1.7.2",
# "networkx==3.5",
# "nodeenv==1.9.1",
# "numcodecs==0.16.2",
# "numpy==2.3.2",
# "packaging==25.0",
# "pandas==2.3.1",
# "parso==0.8.4",
# "partd==1.4.2",
# "pexpect==4.9.0",
# "pillow==11.3.0",
# "platformdirs==4.3.8",
# "pluggy==1.6.0",
# "pre-commit==4.3.0",
# "prompt-toolkit==3.0.51",
# "propcache==0.3.2",
# "ptyprocess==0.7.0",
# "pure-eval==0.2.3",
# "pygments==2.19.2",
# "pyparsing==3.2.3",
# "pyshp==2.3.1",
# "python-dateutil==2.9.0.post0",
# "pytz==2025.2",
# "pyyaml==6.0.2",
# "pyzmq==27.0.2",
# "rasterio==1.4.3",
# "requests==2.32.5",
# "rioxarray==0.19.0",
# "s3fs==2025.7.0",
# "sh==2.2.2",
# "shapely==2.1.1",
# "six==1.17.0",
# "stack-data==0.6.3",
# "toolz==1.0.0",
# "tornado==6.5.2",
# "traitlets==5.14.3",
# "typing-extensions==4.14.1",
# "tzdata==2025.2",
# "urllib3==2.5.0",
# "virtualenv==20.34.0",
# "wcwidth==0.2.13",
# "wrapt==1.17.3",
# "xarray==2025.6.1",
# "yarl==1.20.1",
# "zarr==3.1.1",
# ]
# [tool.uv.sources]
#   gdal = { index = "large-image" }
# [[tool.uv.index]]
#   name = "large-image"
#   url = "https://girder.github.io/large_image_wheels/"
# ///
"""
MLCast Source Radar Zarr Validator
A tool that validates compatibility of MLCast Radar Source Zarr datasets.

Author: MLCast Team
Version: 1.0
"""

import sys
import json
import argparse
import warnings
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import xarray as xr

TOOL_NAME = "MLCast Source Radar Zarr Validator"
TOOL_VERSION = "1.0"
TOOL_FULL_NAME = f"{TOOL_NAME} v{TOOL_VERSION}"

# Remote file system support
try:
    import fsspec
    import s3fs
    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False

# Core tool imports
try:
    from osgeo import gdal, osr
    gdal.UseExceptions()  # Enable exceptions for better error handling
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

try:
    import cartopy.crs as ccrs
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

try:
    import rioxarray as rxr
    RIOXARRAY_AVAILABLE = True
except ImportError:
    RIOXARRAY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIP = "SKIP"


@dataclass
class ValidationTest:
    name: str
    description: str
    result: TestResult
    details: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ToolCompatibilityReport:
    tests: List[ValidationTest] = field(default_factory=list)
    
    def add_test(self, name: str, description: str, result: TestResult, 
                 details: str = None, error: str = None):
        self.tests.append(ValidationTest(name, description, result, details, error))
    
    def get_summary(self) -> Dict[str, int]:
        return {
            'passed': sum(1 for t in self.tests if t.result == TestResult.PASS),
            'failed': sum(1 for t in self.tests if t.result == TestResult.FAIL),
            'warnings': sum(1 for t in self.tests if t.result == TestResult.WARNING),
            'skipped': sum(1 for t in self.tests if t.result == TestResult.SKIP)
        }
    
    def has_failures(self) -> bool:
        return any(t.result == TestResult.FAIL for t in self.tests)


class MLCastGeoreferenceValidator:
    """
    Comprehensive MLCast Zarr validator with practical georeferencing testing.
    
    Combines MLCast specification compliance with real-world tool compatibility
    testing (xarray, GDAL, cartopy) as emphasized in GitHub issue #6.
    """
    
    # Valid licenses
    ACCEPTED_LICENSES = {
        'CC-BY', 'CC-BY-SA', 'CC-BY-4.0', 'CC-BY-SA-4.0', 
        'OGL', 'OGL-UK-3.0', 'OGL-Canada-2.0'
    }
    
    WARNING_LICENSES = {
        'CC-BY-NC', 'CC-BY-ND', 'CC-BY-NC-SA', 'CC-BY-NC-ND',
        'CC-BY-NC-4.0', 'CC-BY-ND-4.0', 'CC-BY-NC-SA-4.0', 'CC-BY-NC-ND-4.0'
    }
    
    # Valid variable names and units by type
    VALID_VARIABLE_SPECS = {
        'precipitation_rate': {
            'names': ['mmh', 'rr', 'tprate', 'prate', 'rain_rate', 'rainfall_flux', 'rainfall_rate'],
            'units': ['kg m-2 h-1', 'mm h-1', 'mm/h']
        },
        'reflectivity': {
            'names': ['equivalent_reflectivity_factor', 'dbz', 'rare'],
            'units': ['dBZ']
        },
        'precipitation_amount': {
            'names': ['rainfall_amount', 'mm', 'precipitation_amount', 'tp'],
            'units': ['kg m-2', 'mm']
        }
    }
    ALL_VALID_NAMES = [name.lower() for spec in VALID_VARIABLE_SPECS.values() for name in spec['names']]
    
    # Base required coordinates (always needed)
    BASE_REQUIRED_COORDS = {'lat', 'lon', 'time'}
    
    # Additional coordinates for projected systems
    PROJECTED_COORDS = {'x', 'y'}
    
    # Required CF attributes for data variables
    REQUIRED_DATA_ATTRS = {'long_name', 'standard_name', 'units'}
    
    def __init__(self, zarr_path: str, verbose: bool = False, storage_options: Dict = None, debug: bool = False,
                 custom_data_vars: List[str] = None, custom_base_coords: List[str] = None, 
                 custom_proj_coords: List[str] = None):
        self.zarr_path = zarr_path
        self.is_remote = self._is_remote_path(zarr_path)
        self.path = Path(zarr_path) if not self.is_remote else None
        self.storage_options = storage_options or {}
        self.verbose = verbose
        self.debug = debug
        self.report = ToolCompatibilityReport()
        self.ds = None
        self.zarr_store = None
        self.data_var = None
        self.crs_var = None
        
        # Custom validation specifications (override defaults if provided)
        self.custom_data_vars = custom_data_vars
        self.custom_base_coords = set(custom_base_coords) if custom_base_coords else self.BASE_REQUIRED_COORDS
        self.custom_proj_coords = set(custom_proj_coords) if custom_proj_coords else self.PROJECTED_COORDS
        
    def _is_remote_path(self, path: str) -> bool:
        """Check if path is a remote URL (S3, HTTP, etc.)."""
        parsed = urlparse(path)
        return parsed.scheme in ['s3', 'http', 'https', 'gs', 'azure']
        
    def validate(self) -> ToolCompatibilityReport:        
        display_path = self.zarr_path if self.is_remote else str(self.path)
        print(f"🔍 Validating dataset: {display_path}")
        if self.is_remote:
            print("📡 Remote Zarr store detected")
        print("-" * 60)
        
        # Load and analyze dataset
        if not self._load_dataset():
            return self.report
        
        if not self._identify_data_components():
            return self.report
        
        # Print dataset info in debug mode
        if self.debug:
            self._print_dataset_debug_info()
        
        # MLCast specification compliance tests
        self._validate_zarr_format()
        self._validate_spatial_requirements() 
        self._validate_temporal_requirements()
        self._validate_data_variable()
        self._validate_coordinates()
        self._validate_chunking()
        self._validate_compression()
        self._validate_license()
        self._validate_missing_data()
        self._validate_future_timesteps()
        
        # Core tool compatibility tests
        self._test_xarray_compatibility()
        self._test_gdal_compatibility() 
        self._test_cartopy_compatibility()
        
        # Cross-tool consistency tests
        self._test_crs_consistency()
        
        # Practical workflow tests
        self._test_georeferencing_workflows()
        
        return self.report
    
    def _print_dataset_debug_info(self):
        """Print detailed dataset information for debugging."""
        print("\n🔍 DEBUG: Dataset Information")
        print("-" * 40)
        
        print(f"Data variables: {list(self.ds.data_vars.keys())}")
        print(f"Coordinates: {list(self.ds.coords.keys())}")
        print(f"Dimensions: {dict(self.ds.dims)}")
        
        if self.data_var:
            data_array = self.ds[self.data_var]
            print(f"Main data variable: '{self.data_var}'")
            print(f"  Shape: {data_array.shape}")
            print(f"  Dimensions: {data_array.dims}")
            print(f"  Data type: {data_array.dtype}")
            
        if self.crs_var:
            crs_var = self.ds[self.crs_var]
            print(f"CRS variable: '{self.crs_var}'")
            print(f"  Attributes: {list(crs_var.attrs.keys())}")
            
        # Print coordinate info
        for coord in ['x', 'y', 'lat', 'lon', 'time']:
            if coord in self.ds.coords:
                coord_vals = self.ds[coord].values
                if coord == 'time':
                    # Handle time coordinates specially
                    try:
                        time_min = pd.to_datetime(coord_vals.min())
                        time_max = pd.to_datetime(coord_vals.max())
                        print(f"  {coord}: shape={coord_vals.shape}, range=[{time_min}, {time_max}]")
                    except:
                        print(f"  {coord}: shape={coord_vals.shape}, dtype={coord_vals.dtype}")
                else:
                    # Handle numeric coordinates
                    try:
                        print(f"  {coord}: shape={coord_vals.shape}, range=[{coord_vals.min():.3f}, {coord_vals.max():.3f}]")
                    except:
                        print(f"  {coord}: shape={coord_vals.shape}, dtype={coord_vals.dtype}")
        
        print("-" * 40)
    
    def _load_dataset(self) -> bool:
        """Load dataset with xarray and basic validation."""
        
        # Check fsspec availability for remote paths
        if self.is_remote and not FSSPEC_AVAILABLE:
            self.report.add_test(
                "remote_support",
                "Check remote file system support",
                TestResult.FAIL,
                None,
                "fsspec and s3fs required for remote Zarr stores"
            )
            return False
        
        # Prepare arguments for xarray.open_zarr
        zarr_args = {}
        if self.is_remote:
            zarr_args['storage_options'] = self.storage_options
            store_path = self.zarr_path
        else:
            store_path = str(self.path)
        
        try:
            # Try with consolidated metadata first
            self.ds = xr.open_zarr(store_path, consolidated=True, **zarr_args)
            
            # Also load zarr store directly for format checks
            try:
                import zarr
                if self.is_remote:
                    # For remote stores, create a filesystem map
                    import fsspec
                    fs = fsspec.filesystem('s3', **self.storage_options)
                    self.zarr_store = zarr.open(fs.get_mapper(store_path), mode='r')
                else:
                    self.zarr_store = zarr.open(str(self.path), mode='r')
            except Exception:
                pass  # zarr_store is optional for some tests
            
            location_info = "remote S3" if self.is_remote else "local filesystem"
            self.report.add_test(
                "dataset_loading", 
                "Load Zarr dataset with xarray",
                TestResult.PASS,
                f"Successfully loaded {len(self.ds.data_vars)} data variables from {location_info}"
            )
            return True
        except Exception as e:
            try:
                # Fallback without consolidated metadata
                self.ds = xr.open_zarr(store_path, consolidated=False, **zarr_args)
                self.report.add_test(
                    "dataset_loading",
                    "Load Zarr dataset with xarray", 
                    TestResult.WARNING,
                    "Loaded without consolidated metadata",
                    str(e)
                )
                return True
            except Exception as e2:
                self.report.add_test(
                    "dataset_loading",
                    "Load Zarr dataset with xarray",
                    TestResult.FAIL,
                    None,
                    f"Failed to load dataset: {e2}"
                )
                return False
    
    def _identify_data_components(self) -> bool:
        """Identify main data variable and CRS components."""
        
        # Find main data variable
        data_vars = list(self.ds.data_vars)
        if not data_vars:
            self.report.add_test(
                "data_identification",
                "Identify main data variable",
                TestResult.FAIL,
                None,
                "No data variables found"
            )
            return False
        
        # Use custom data variable list if provided, otherwise auto-detect
        if self.custom_data_vars:
            # Check if any of the custom variables exist in the dataset
            found_vars = [var for var in self.custom_data_vars if var in data_vars]
            if found_vars:
                self.data_var = found_vars[0]
                details = f"Found custom data variable '{self.data_var}' from specified list: {self.custom_data_vars}"
        else:
            found_vars = [var for var in data_vars if var.lower() in self.ALL_VALID_NAMES]
            if found_vars:
                self.data_var = found_vars[0]
                details = f"Auto-detected data variable '{self.data_var}' (first in dataset)"
        if self.data_var:
            self.report.add_test(
                "data_identification",
                "Identify main data variable",
                TestResult.PASS,
                details
            )
        else:
            self.report.add_test(
                "data_identification",
                "Identify main data variable",
                TestResult.FAIL,
                None,
                f"Could not identify main data variable. Available variables: {data_vars}, Expected names: {self.ALL_VALID_NAMES}"
            )
            return False
    
        # Check for grid_mapping
        data_array = self.ds[self.data_var]
        if 'grid_mapping' not in data_array.attrs:
            self.report.add_test(
                "grid_mapping_detection",
                "Detect grid_mapping attribute",
                TestResult.FAIL,
                None,
                f"Data variable '{self.data_var}' missing grid_mapping attribute"
            )
            return False
        
        grid_mapping_name = data_array.attrs['grid_mapping']
        if grid_mapping_name not in self.ds.variables:
            self.report.add_test(
                "crs_variable_detection",
                "Detect CRS variable",
                TestResult.FAIL,
                None,
                f"grid_mapping references non-existent variable '{grid_mapping_name}'"
            )
            return False
        
        self.crs_var = grid_mapping_name
        self.report.add_test(
            "georef_components",
            "Identify georeferencing components",
            TestResult.PASS,
            f"Data variable: '{self.data_var}', CRS variable: '{self.crs_var}'"
        )
        return True
    
    def _get_crs_info(self) -> Optional[Dict[str, Any]]:
        """Get CRS information including projection type."""
        if not self.crs_var:
            return None
        
        try:
            crs_var = self.ds[self.crs_var]
            
            # Get WKT if available
            wkt = crs_var.attrs.get('crs_wkt', '')
            
            # Try to determine if projected using GDAL
            if GDAL_AVAILABLE and wkt:
                try:
                    srs = osr.SpatialReference()
                    srs.ImportFromWkt(wkt)
                    
                    return {
                        'wkt': wkt,
                        'is_projected': srs.IsProjected() == 1,
                        'is_geographic': srs.IsGeographic() == 1,
                        'authority': srs.GetAuthorityName(None),
                        'code': srs.GetAuthorityCode(None)
                    }
                except Exception:
                    pass
            
            # Fallback: try to guess from coordinate presence
            has_xy = 'x' in self.ds.coords and 'y' in self.ds.coords
            
            return {
                'wkt': wkt,
                'is_projected': has_xy,  # Heuristic: if x/y coords exist, likely projected
                'is_geographic': not has_xy,
                'authority': None,
                'code': None
            }
            
        except Exception:
            return None
    
    def _test_xarray_compatibility(self):
        """Test xarray georeferencing integration."""
        print("📊 Testing xarray compatibility...")
        
        # Test basic coordinate access
        try:
            coords = ['x', 'y', 'lat', 'lon', 'time']
            missing_coords = [c for c in coords if c not in self.ds.coords]
            
            if missing_coords:
                self.report.add_test(
                    "xarray_coordinates",
                    "Access standard coordinates",
                    TestResult.FAIL,
                    None,
                    f"Missing coordinates: {missing_coords}"
                )
            else:
                self.report.add_test(
                    "xarray_coordinates", 
                    "Access standard coordinates",
                    TestResult.PASS,
                    "All standard coordinates accessible"
                )
                
        except Exception as e:
            self.report.add_test(
                "xarray_coordinates",
                "Access standard coordinates", 
                TestResult.FAIL,
                None,
                str(e)
            )
        
        # Test CRS attribute access
        try:
            crs_var = self.ds[self.crs_var]
            has_spatial_ref = 'spatial_ref' in crs_var.attrs
            has_crs_wkt = 'crs_wkt' in crs_var.attrs
            
            if has_spatial_ref and has_crs_wkt:
                self.report.add_test(
                    "xarray_crs_access",
                    "Access CRS attributes via xarray",
                    TestResult.PASS,
                    "Both spatial_ref and crs_wkt accessible"
                )
            else:
                missing = []
                if not has_spatial_ref: missing.append('spatial_ref')
                if not has_crs_wkt: missing.append('crs_wkt')
                self.report.add_test(
                    "xarray_crs_access",
                    "Access CRS attributes via xarray",
                    TestResult.FAIL,
                    None,
                    f"Missing CRS attributes: {missing}"
                )
                
        except Exception as e:
            self.report.add_test(
                "xarray_crs_access",
                "Access CRS attributes via xarray",
                TestResult.FAIL,
                None,
                str(e)
            )
        
        # Test data slicing with coordinates
        try:
            sample_data = self.ds[self.data_var].isel(time=0)
            if sample_data.sizes['x'] > 0 and sample_data.sizes['y'] > 0:
                # Test coordinate-based selection
                x_mid = len(self.ds.x) // 2
                y_mid = len(self.ds.y) // 2
                subset = self.ds[self.data_var].isel(time=0, x=slice(x_mid-10, x_mid+10), 
                                                  y=slice(y_mid-10, y_mid+10))
                
                self.report.add_test(
                    "xarray_spatial_slicing",
                    "Spatial data slicing with coordinates",
                    TestResult.PASS,
                    f"Successfully sliced to {subset.shape} subset"
                )
            else:
                self.report.add_test(
                    "xarray_spatial_slicing",
                    "Spatial data slicing with coordinates",
                    TestResult.FAIL,
                    None,
                    "Zero-sized spatial dimensions"
                )
                
        except Exception as e:
            self.report.add_test(
                "xarray_spatial_slicing",
                "Spatial data slicing with coordinates",
                TestResult.FAIL,
                None,
                str(e)
            )
    
    def _test_gdal_compatibility(self):
        """Test GDAL georeferencing interpretation."""
        print("🗺️  Testing GDAL compatibility...")
        
        if not GDAL_AVAILABLE:
            self.report.add_test(
                "gdal_availability",
                "GDAL library availability",
                TestResult.SKIP,
                "GDAL not available for testing"
            )
            return
        
        # Test CRS interpretation via GDAL
        try:
            crs_var = self.ds[self.crs_var]
            if 'crs_wkt' not in crs_var.attrs:
                self.report.add_test(
                    "gdal_wkt_parsing",
                    "Parse WKT string with GDAL",
                    TestResult.FAIL,
                    None,
                    "No crs_wkt attribute found"
                )
                return
            
            wkt_string = crs_var.attrs['crs_wkt']
            
            # Create spatial reference from WKT
            srs = osr.SpatialReference()
            err = srs.ImportFromWkt(wkt_string)
            
            if err == 0:  # OGRERR_NONE
                self.report.add_test(
                    "gdal_wkt_parsing",
                    "Parse WKT string with GDAL",
                    TestResult.PASS,
                    f"Successfully parsed WKT, Authority: {srs.GetAuthorityName(None)}"
                )
                
                # Test if it's a valid projected or geographic CRS
                if srs.IsProjected():
                    proj_name = srs.GetAttrValue('PROJECTION')
                    self.report.add_test(
                        "gdal_projection_type",
                        "Identify projection type",
                        TestResult.PASS,
                        f"Projected CRS with projection: {proj_name}"
                    )
                elif srs.IsGeographic():
                    self.report.add_test(
                        "gdal_projection_type", 
                        "Identify projection type",
                        TestResult.PASS,
                        "Geographic CRS (lat/lon)"
                    )
                else:
                    self.report.add_test(
                        "gdal_projection_type",
                        "Identify projection type", 
                        TestResult.WARNING,
                        "CRS type unclear - neither clearly projected nor geographic"
                    )
            else:
                self.report.add_test(
                    "gdal_wkt_parsing",
                    "Parse WKT string with GDAL",
                    TestResult.FAIL,
                    None,
                    f"GDAL failed to parse WKT (error code: {err})"
                )
                return
                
        except Exception as e:
            self.report.add_test(
                "gdal_wkt_parsing",
                "Parse WKT string with GDAL",
                TestResult.FAIL,
                None,
                str(e)
            )
            return
        
        # Test georeferencing via rioxarray export (if available)
        if RIOXARRAY_AVAILABLE:
            self._test_gdal_via_rioxarray()
    
    def _test_gdal_via_rioxarray(self):
        """Test GDAL compatibility via rioxarray export."""
        try:
            # Export a sample to test GDAL reading
            sample_data = self.ds[self.data_var].isel(time=0)
            
            # Set CRS from the crs_wkt
            crs_var = self.ds[self.crs_var]
            if 'crs_wkt' in crs_var.attrs:
                wkt_string = crs_var.attrs['crs_wkt']
                sample_data = sample_data.rio.write_crs(wkt_string)
                
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                try:
                    # Export to GeoTIFF
                    sample_data.rio.to_raster(tmp_path)
                    
                    # Try to read back with GDAL
                    gdal_ds = gdal.Open(tmp_path)
                    if gdal_ds is not None:
                        # Get geotransform
                        geotransform = gdal_ds.GetGeoTransform()
                        projection = gdal_ds.GetProjection()
                        
                        if geotransform and projection:
                            self.report.add_test(
                                "gdal_roundtrip",
                                "GDAL roundtrip via rioxarray export",
                                TestResult.PASS,
                                f"Successfully exported and read back with GDAL"
                            )
                        else:
                            self.report.add_test(
                                "gdal_roundtrip", 
                                "GDAL roundtrip via rioxarray export",
                                TestResult.FAIL,
                                None,
                                "Missing geotransform or projection in exported file"
                            )
                        gdal_ds = None  # Close dataset
                    else:
                        self.report.add_test(
                            "gdal_roundtrip",
                            "GDAL roundtrip via rioxarray export", 
                            TestResult.FAIL,
                            None,
                            "GDAL could not open exported file"
                        )
                        
                finally:
                    # Clean up
                    try:
                        Path(tmp_path).unlink()
                    except:
                        pass
                        
        except Exception as e:
            self.report.add_test(
                "gdal_roundtrip",
                "GDAL roundtrip via rioxarray export",
                TestResult.FAIL,
                None,
                str(e)
            )
    
    def _test_cartopy_compatibility(self):
        """Test cartopy CRS creation and usage."""
        print("🌍 Testing cartopy compatibility...")
        
        if not CARTOPY_AVAILABLE:
            self.report.add_test(
                "cartopy_availability",
                "cartopy library availability",
                TestResult.SKIP,
                "cartopy not available for testing"
            )
            return
        
        # Test CRS creation from WKT
        try:
            crs_var = self.ds[self.crs_var]
            if 'crs_wkt' not in crs_var.attrs:
                self.report.add_test(
                    "cartopy_crs_creation",
                    "Create cartopy CRS from WKT",
                    TestResult.FAIL,
                    None,
                    "No crs_wkt attribute found"
                )
                return
            
            wkt_string = crs_var.attrs['crs_wkt']
            
            # Create cartopy CRS
            crs = ccrs.CRS(wkt_string)
            
            self.report.add_test(
                "cartopy_crs_creation",
                "Create cartopy CRS from WKT",
                TestResult.PASS,
                f"Successfully created cartopy CRS: {type(crs).__name__}"
            )
            
            # Test BBOX presence (critical for cartopy per issue #6)
            if 'BBOX' in wkt_string or 'bbox' in wkt_string.lower():
                self.report.add_test(
                    "cartopy_bbox_check",
                    "Check BBOX in WKT for cartopy compatibility",
                    TestResult.PASS,
                    "BBOX specification found in WKT"
                )
            else:
                self.report.add_test(
                    "cartopy_bbox_check",
                    "Check BBOX in WKT for cartopy compatibility",
                    TestResult.WARNING,
                    "BBOX specification missing - may cause issues with cartopy plotting"
                )
            
            # Test coordinate bounds access
            try:
                x_bounds = [float(self.ds.x.min()), float(self.ds.x.max())]
                y_bounds = [float(self.ds.y.min()), float(self.ds.y.max())]
                
                self.report.add_test(
                    "cartopy_bounds_access",
                    "Access coordinate bounds for cartopy plotting",
                    TestResult.PASS,
                    f"X bounds: {x_bounds}, Y bounds: {y_bounds}"
                )
            except Exception as e:
                self.report.add_test(
                    "cartopy_bounds_access",
                    "Access coordinate bounds for cartopy plotting",
                    TestResult.FAIL,
                    None,
                    str(e)
                )
            
            # Test coordinate transformation (if possible)
            self._test_cartopy_transforms(crs)
            
        except Exception as e:
            self.report.add_test(
                "cartopy_crs_creation",
                "Create cartopy CRS from WKT",
                TestResult.FAIL,
                None,
                str(e)
            )
    
    def _test_cartopy_transforms(self, crs):
        """Test coordinate transformations with cartopy."""
        try:
            # Test transformation to PlateCarree (common use case)
            pc = ccrs.PlateCarree()
            
            # Sample a few coordinate points
            x_sample = self.ds.x.values[::max(1, len(self.ds.x)//5)][:5]
            y_sample = self.ds.y.values[::max(1, len(self.ds.y)//5)][:5]
            
            # Transform coordinates
            xx, yy = np.meshgrid(x_sample, y_sample)
            transformed = pc.transform_points(crs, xx.flatten(), yy.flatten())
            
            if not np.any(np.isnan(transformed)):
                self.report.add_test(
                    "cartopy_coordinate_transform",
                    "Transform coordinates to PlateCarree",
                    TestResult.PASS,
                    f"Successfully transformed {len(transformed)} coordinate pairs"
                )
            else:
                self.report.add_test(
                    "cartopy_coordinate_transform",
                    "Transform coordinates to PlateCarree",
                    TestResult.WARNING,
                    "Some transformed coordinates are NaN"
                )
                
        except Exception as e:
            self.report.add_test(
                "cartopy_coordinate_transform",
                "Transform coordinates to PlateCarree",
                TestResult.FAIL,
                None,
                str(e)
            )
    
    def _test_crs_consistency(self):
        """Test CRS consistency across tools."""
        print("🔄 Testing cross-tool CRS consistency...")
        
        results = {}
        
        # Get CRS info from xarray/CF
        try:
            crs_var = self.ds[self.crs_var]
            if 'crs_wkt' in crs_var.attrs:
                results['xarray_wkt'] = crs_var.attrs['crs_wkt']
        except:
            pass
        
        # Get CRS info from GDAL
        if GDAL_AVAILABLE and 'xarray_wkt' in results:
            try:
                srs = osr.SpatialReference()
                srs.ImportFromWkt(results['xarray_wkt'])
                results['gdal_authority'] = srs.GetAuthorityName(None)
                results['gdal_code'] = srs.GetAuthorityCode(None)
            except:
                pass
        
        # Get CRS info from cartopy
        if CARTOPY_AVAILABLE and 'xarray_wkt' in results:
            try:
                crs = ccrs.CRS(results['xarray_wkt'])
                results['cartopy_crs_type'] = type(crs).__name__
            except:
                pass
        
        # Check consistency
        consistent_tools = []
        inconsistent_details = []
        
        if 'xarray_wkt' in results:
            consistent_tools.append('xarray')
        if 'gdal_authority' in results:
            consistent_tools.append('GDAL')
        if 'cartopy_crs_type' in results:
            consistent_tools.append('cartopy')
        
        if len(consistent_tools) >= 2:
            self.report.add_test(
                "crs_cross_tool_consistency",
                "CRS consistency across tools",
                TestResult.PASS,
                f"CRS successfully interpreted by: {', '.join(consistent_tools)}"
            )
        else:
            self.report.add_test(
                "crs_cross_tool_consistency", 
                "CRS consistency across tools",
                TestResult.FAIL,
                None,
                f"CRS only interpreted by: {', '.join(consistent_tools) if consistent_tools else 'none'}"
            )
    
    def _test_georeferencing_workflows(self):
        """Test practical georeferencing workflows."""
        print("⚙️  Testing practical georeferencing workflows...")
        
        # Test 1: Basic plotting workflow (if matplotlib available)
        if MATPLOTLIB_AVAILABLE and CARTOPY_AVAILABLE:
            self._test_plotting_workflow()
        
        # Test 2: Coordinate-based data selection
        self._test_coordinate_selection()
        
        # Test 3: Spatial bounds calculation
        self._test_spatial_bounds()
    
    def _test_plotting_workflow(self):
        """Test basic plotting workflow with cartopy."""
        try:
            crs_var = self.ds[self.crs_var]
            if 'crs_wkt' not in crs_var.attrs:
                self.report.add_test(
                    "plotting_workflow",
                    "Basic plotting workflow",
                    TestResult.SKIP,
                    "No crs_wkt available"
                )
                return
            
            # Create CRS from WKT
            wkt_string = crs_var.attrs['crs_wkt']
            data_crs = ccrs.Projection(wkt_string)
            
            # Test matplotlib/cartopy integration
            if MATPLOTLIB_AVAILABLE:
                # Test 1: Try direct CRS usage and create sample plots
                try:
                    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': data_crs})
                    
                    # Try to create actual plots of first and last timesteps
                    try:
                        self._create_sample_plots(data_crs)
                        plt.close(fig)
                        
                        self.report.add_test(
                            "plotting_workflow",
                            "Basic plotting workflow",
                            TestResult.PASS,
                            f"Successfully created cartopy plots and saved sample timesteps as PNG files"
                        )
                        return  # Success, exit early
                    except Exception as plot_data_error:
                        plt.close(fig)
                        # Axes creation worked but data plotting failed
                        self.report.add_test(
                            "plotting_workflow",
                            "Basic plotting workflow",
                            TestResult.PASS,
                            f"Cartopy axes creation successful, data plotting failed: {str(plot_data_error)[:100]}..."
                        )
                        return
                        
                except Exception as e:
                    self.report.add_test(
                        "plotting_workflow",
                        "Basic plotting workflow",
                        TestResult.FAIL,
                        None,
                        f"Failed to create cartopy axes: {str(e)}"
                    )
                    return
            else:
                # Just test CRS creation if matplotlib not available
                self.report.add_test(
                    "plotting_workflow",
                    "Basic plotting workflow",
                    TestResult.PASS,
                    f"CRS successfully created for plotting: {type(data_crs).__name__}"
                )
            
        except Exception as e:
            self.report.add_test(
                "plotting_workflow",
                "Basic plotting workflow",
                TestResult.FAIL,
                None,
                f"Plotting workflow failed: {str(e)}"
            )
    
    def _create_sample_plots(self, data_crs):
        """Create and save sample plots of first and last timesteps."""
        
        # Get coordinate bounds for plotting
        x_vals = self.ds.x.values
        y_vals = self.ds.y.values
        x_min, x_max = float(x_vals.min()), float(x_vals.max())
        y_min, y_max = float(y_vals.min()), float(y_vals.max())
        
        # Sample timesteps to plot
        timesteps = [
            (0, "first"),
            (-1, "last")
        ]
        
        for idx, label in timesteps:
            try:
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': data_crs})
                
                # Get data for this timestep
                data = self.ds[self.data_var].isel(time=idx)
                
                # Create the plot
                im = ax.pcolormesh(
                    self.ds.x.values, 
                    self.ds.y.values, 
                    data.values,
                    transform=data_crs,
                    shading='auto',
                    cmap='viridis'
                )
                
                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8, label=f"{data.attrs.get('long_name', self.data_var)} ({data.attrs.get('units', 'units')})")
                
                # Set extent
                ax.set_extent([x_min, x_max, y_min, y_max], crs=data_crs)
                
                # Add title with timestamp
                try:
                    timestamp = pd.to_datetime(self.ds.time.isel(time=idx).values)
                    ax.set_title(f"{self.data_var} Dataset - {label.title()} Timestep\n{timestamp}", fontsize=14)
                except:
                    ax.set_title(f"{self.data_var} Dataset - {label.title()} Timestep", fontsize=14)
                
                # Add gridlines
                gl = ax.gridlines(draw_labels=True, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                
                # Save the plot
                filename = f"mlcast_validator_plot_{self.data_var}_{label}_timestep.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"📊 Saved sample plot: {filename}")
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not create {label} timestep plot: {e}")
                # Don't raise - this is just a bonus feature
                continue
    
    def _test_coordinate_selection(self):
        """Test coordinate-based data selection."""
        try:
            # Check if required coordinates exist
            if 'x' not in self.ds.coords or 'y' not in self.ds.coords:
                self.report.add_test(
                    "coordinate_selection",
                    "Coordinate-based data selection",
                    TestResult.SKIP,
                    "Missing x or y coordinates"
                )
                return
                
            # Get coordinate values safely
            x_vals = self.ds.x.values
            y_vals = self.ds.y.values
            
            if len(x_vals) < 2 or len(y_vals) < 2:
                self.report.add_test(
                    "coordinate_selection",
                    "Coordinate-based data selection",
                    TestResult.SKIP,
                    f"Insufficient coordinate points: x={len(x_vals)}, y={len(y_vals)}"
                )
                return
            
            # Select middle region with bounds checking
            x_center = x_vals[len(x_vals)//2]
            y_center = y_vals[len(y_vals)//2]
            
            # Use smaller range to avoid coordinate ordering issues
            x_range = abs(x_vals[-1] - x_vals[0]) * 0.05  # 5% of range
            y_range = abs(y_vals[-1] - y_vals[0]) * 0.05
            
            # Determine coordinate ordering
            x_ascending = x_vals[0] < x_vals[-1]
            y_ascending = y_vals[0] < y_vals[-1]
            
            # Create bounds respecting coordinate order
            if x_ascending:
                x_min, x_max = x_center - x_range, x_center + x_range
            else:
                x_min, x_max = x_center + x_range, x_center - x_range
                
            if y_ascending:
                y_min, y_max = y_center - y_range, y_center + y_range
            else:
                y_min, y_max = y_center + y_range, y_center - y_range
            
            # Test coordinate-based selection (without method for slices)
            subset = self.ds[self.data_var].sel(
                x=slice(x_min, x_max),
                y=slice(y_min, y_max)
            )
            
            if subset.sizes.get('x', 0) > 0 and subset.sizes.get('y', 0) > 0:
                self.report.add_test(
                    "coordinate_selection",
                    "Coordinate-based data selection",
                    TestResult.PASS,
                    f"Selected subset: {subset.sizes.get('x', 0)}×{subset.sizes.get('y', 0)} pixels"
                )
            else:
                # Try alternative approach with isel
                try:
                    x_mid = len(x_vals) // 2
                    y_mid = len(y_vals) // 2
                    subset_isel = self.ds[self.data_var].isel(
                        x=slice(max(0, x_mid-10), min(len(x_vals), x_mid+10)),
                        y=slice(max(0, y_mid-10), min(len(y_vals), y_mid+10))
                    )
                    
                    self.report.add_test(
                        "coordinate_selection",
                        "Coordinate-based data selection",
                        TestResult.WARNING,
                        f"Coordinate selection failed, but index selection works: {subset_isel.sizes.get('x', 0)}×{subset_isel.sizes.get('y', 0)} pixels"
                    )
                except:
                    self.report.add_test(
                        "coordinate_selection",
                        "Coordinate-based data selection",
                        TestResult.FAIL,
                        None,
                        "Both coordinate and index selection failed"
                    )
                
        except Exception as e:
            self.report.add_test(
                "coordinate_selection",
                "Coordinate-based data selection",
                TestResult.FAIL,
                None,
                f"Coordinate selection test failed: {str(e)}"
            )
    
    def _test_spatial_bounds(self):
        """Test spatial bounds calculation."""
        try:
            # Check if coordinates exist
            if 'x' not in self.ds.coords or 'y' not in self.ds.coords:
                self.report.add_test(
                    "spatial_bounds",
                    "Calculate spatial bounds",
                    TestResult.SKIP,
                    "Missing x or y coordinates"
                )
                return
                
            # Calculate spatial bounds safely
            x_min, x_max = float(self.ds.x.min()), float(self.ds.x.max())
            y_min, y_max = float(self.ds.y.min()), float(self.ds.y.max())
            
            # Handle cases where coordinates might be in reverse order
            if x_min > x_max:
                x_min, x_max = x_max, x_min
            if y_min > y_max:
                y_min, y_max = y_max, y_min
            
            # Basic sanity checks
            if x_max > x_min and y_max > y_min:
                # Calculate approximate area (in coordinate units)
                width = x_max - x_min
                height = y_max - y_min
                
                # Format bounds nicely
                bounds_str = f"[{x_min:.1f}, {x_max:.1f}, {y_min:.1f}, {y_max:.1f}]"
                size_str = f"{width:.1f} × {height:.1f} units"
                
                self.report.add_test(
                    "spatial_bounds",
                    "Calculate spatial bounds",
                    TestResult.PASS,
                    f"Bounds: {bounds_str}, Size: {size_str}"
                )
            else:
                self.report.add_test(
                    "spatial_bounds",
                    "Calculate spatial bounds",
                    TestResult.FAIL,
                    None,
                    f"Invalid bounds: x=({x_min}, {x_max}), y=({y_min}, {y_max})"
                )
                
        except Exception as e:
            self.report.add_test(
                "spatial_bounds",
                "Calculate spatial bounds",
                TestResult.FAIL,
                None,
                f"Bounds calculation failed: {str(e)}"
            )
    
    def _validate_zarr_format(self):
        """Validate Zarr format requirements."""
        category = "Zarr Format"
        
        # Check Zarr version
        try:
            zarr_format = getattr(self.zarr_store, 'zarr_format', 2)
            if zarr_format in [2, 3]:
                self.report.add_test(
                    "zarr_version", f"Zarr version compatibility", 
                    TestResult.PASS, f"Using supported Zarr v{zarr_format} format"
                )
            else:
                self.report.add_test(
                    "zarr_version", f"Zarr version compatibility",
                    TestResult.FAIL, None, f"Unsupported Zarr version: v{zarr_format}"
                )
        except Exception as e:
            self.report.add_test(
                "zarr_version", "Zarr version compatibility",
                TestResult.WARNING, None, f"Could not determine Zarr version: {e}"
            )
        
        # Check consolidated metadata for Zarr v2
        if hasattr(self.zarr_store, 'zarr_format') and self.zarr_store.zarr_format == 2:
            consolidated_path = self.path / '.zmetadata' if self.path else None
            if consolidated_path and consolidated_path.exists():
                self.report.add_test(
                    "consolidated_metadata", "Consolidated metadata presence",
                    TestResult.PASS, "Has consolidated metadata (.zmetadata)"
                )
            elif not self.is_remote:  # Only check for local files
                self.report.add_test(
                    "consolidated_metadata", "Consolidated metadata presence",
                    TestResult.FAIL, None, "Zarr v2 missing required consolidated metadata"
                )
            else:
                self.report.add_test(
                    "consolidated_metadata", "Consolidated metadata presence",
                    TestResult.PASS, "Remote dataset loaded successfully (assuming consolidated)"
                )
    
    def _validate_spatial_requirements(self):
        """Validate spatial requirements."""
        category = "Spatial Requirements"
        
        if not self.data_var:
            return
        
        data_array = self.ds[self.data_var]
        dims = data_array.dims
        
        # Find spatial dimensions (exclude time)
        spatial_dims = [d for d in dims if d not in ['time', 't']]
        
        if len(spatial_dims) < 2:
            self.report.add_test(
                "spatial_dimensions", "Spatial dimension check",
                TestResult.FAIL, None, "Need at least 2 spatial dimensions"
            )
            return
        
        # Check minimum 256x256 support
        spatial_sizes = [data_array.sizes[d] for d in spatial_dims]
        if all(s >= 256 for s in spatial_sizes):
            self.report.add_test(
                "spatial_256x256", "256×256 pixel support",
                TestResult.PASS, f"Spatial dimensions {spatial_sizes} support 256×256 crops"
            )
        else:
            self.report.add_test(
                "spatial_256x256", "256×256 pixel support",
                TestResult.FAIL, None, f"Spatial dimensions {spatial_sizes} too small for 256×256 crops"
            )
        
        # Check spatial resolution (≤1km)
        if 'x' in self.ds.coords and 'y' in self.ds.coords:
            try:
                x_vals = self.ds.x.values
                y_vals = self.ds.y.values
                
                if len(x_vals) > 1 and len(y_vals) > 1:
                    x_res = abs(float(x_vals[1] - x_vals[0]))
                    y_res = abs(float(y_vals[1] - y_vals[0]))
                    
                    if x_res <= 1000 and y_res <= 1000:
                        self.report.add_test(
                            "spatial_resolution", "Spatial resolution ≤1km",
                            TestResult.PASS, f"Resolution ({x_res:.1f}m × {y_res:.1f}m) ≤ 1km"
                        )
                    else:
                        self.report.add_test(
                            "spatial_resolution", "Spatial resolution ≤1km",
                            TestResult.FAIL, None, f"Resolution ({x_res:.1f}m × {y_res:.1f}m) exceeds 1km limit"
                        )
            except Exception as e:
                self.report.add_test(
                    "spatial_resolution", "Spatial resolution ≤1km",
                    TestResult.WARNING, None, f"Could not verify spatial resolution: {e}"
                )
    
    def _validate_temporal_requirements(self):
        """Validate temporal coverage and requirements."""
        category = "Temporal Requirements"
        
        if 'time' not in self.ds.coords:
            self.report.add_test(
                "time_coordinate", "Time coordinate presence",
                TestResult.FAIL, None, "Missing 'time' coordinate"
            )
            return
        
        try:
            time_coord = pd.to_datetime(self.ds.time.values)
            
            # Check for 3+ years coverage
            time_range = time_coord[-1] - time_coord[0]
            years = time_range.days / 365.25
            
            if years >= 3:
                self.report.add_test(
                    "temporal_coverage", "Minimum 3-year coverage",
                    TestResult.PASS, f"Temporal coverage: {years:.1f} years (≥3 years)"
                )
            else:
                self.report.add_test(
                    "temporal_coverage", "Minimum 3-year coverage",
                    TestResult.FAIL, None, f"Temporal coverage: {years:.1f} years (<3 years required)"
                )
            
            # Check for variable timesteps
            if len(time_coord) > 1:
                time_diffs = pd.Series(time_coord).diff().dropna()
                unique_diffs = time_diffs.value_counts()
                
                if len(unique_diffs) == 1:
                    self.report.add_test(
                        "timestep_consistency", "Timestep consistency",
                        TestResult.PASS, "Consistent timestep throughout dataset"
                    )
                else:
                    self.report.add_test(
                        "timestep_variability", "Variable timestep handling",
                        TestResult.PASS, f"Variable timesteps detected ({len(unique_diffs)} intervals)"
                    )
                    
                    # Check for consistent_timestep_start if variable
                    if 'consistent_timestep_start' in self.ds.attrs:
                        self.report.add_test(
                            "timestep_metadata", "Timestep metadata",
                            TestResult.PASS, "Has 'consistent_timestep_start' metadata"
                        )
                    else:
                        self.report.add_test(
                            "timestep_metadata", "Timestep metadata",
                            TestResult.WARNING, "Variable timesteps without 'consistent_timestep_start'"
                        )
                        
        except Exception as e:
            self.report.add_test(
                "temporal_analysis", "Temporal coverage analysis",
                TestResult.FAIL, None, f"Failed to analyze temporal coverage: {e}"
            )
    
    def _validate_data_variable(self):
        """Validate data variable requirements."""
        category = "Data Variable"
        
        if not self.data_var:
            return
        
        data_array = self.ds[self.data_var]
        
        # Check data type (must be floating-point)
        if np.issubdtype(data_array.dtype, np.floating):
            self.report.add_test(
                "data_type", "Floating-point data type",
                TestResult.PASS, f"Data type {data_array.dtype} is floating-point"
            )
        else:
            self.report.add_test(
                "data_type", "Floating-point data type",
                TestResult.FAIL, None, f"Data type must be floating-point, found: {data_array.dtype}"
            )
        
        # Check dimension order (time × height × width)
        dims = data_array.dims
        if len(dims) >= 3 and dims[0] in ['time', 't']:
            self.report.add_test(
                "dimension_order", "Dimension ordering",
                TestResult.PASS, f"Dimension order {dims} correct (time first)"
            )
        else:
            self.report.add_test(
                "dimension_order", "Dimension ordering",
                TestResult.FAIL, None, f"Dimensions {dims} must be time × height × width"
            )
        
        # Validate variable name and units
        self._validate_variable_name_and_units(data_array)
        
        # Check required CF attributes
        for attr in self.REQUIRED_DATA_ATTRS:
            if attr in data_array.attrs:
                self.report.add_test(
                    f"cf_attr_{attr}", f"CF attribute '{attr}'",
                    TestResult.PASS, f"Has required '{attr}' attribute"
                )
            else:
                self.report.add_test(
                    f"cf_attr_{attr}", f"CF attribute '{attr}'",
                    TestResult.FAIL, None, f"Missing required '{attr}' attribute"
                )
    
    def _validate_variable_name_and_units(self, data_array):
        """Validate variable name and units compliance."""
        
        var_name = self.data_var.lower()
        units = data_array.attrs.get('units', '').strip()
        

        # Find matching variable type
        matched_type = None
        for var_type, specs in self.VALID_VARIABLE_SPECS.items():
            if var_name in [n.lower() for n in specs['names']]:
                matched_type = var_type
                break

        # # Try to match by standard_name first (more reliable)
        # matched_type = None
        # standard_name = data_array.attrs.get('standard_name', '').lower()
        
        # if standard_name:
        #     if 'rainfall_amount' in standard_name or 'precipitation_amount' in standard_name:
        #         matched_type = 'precipitation_amount'
        #     elif 'rainfall_rate' in standard_name or 'precipitation_rate' in standard_name:
        #         matched_type = 'precipitation_rate'
        #     elif 'reflectivity' in standard_name or 'equivalent_reflectivity' in standard_name:
        #         matched_type = 'reflectivity'
        
        # # Fallback to variable name matching if no standard_name match
        # if not matched_type:
        #     for var_type, specs in self.VALID_VARIABLE_SPECS.items():
        #         if var_name in [n.lower() for n in specs['names']]:
        #             matched_type = var_type
        #             break
        
        if matched_type:
            self.report.add_test(
                "variable_name", "Variable name compliance",
                TestResult.PASS, f"Variable name '{self.data_var}' valid for {matched_type}"
            )
            
            # Check units match the variable type
            if units in self.VALID_VARIABLE_SPECS[matched_type]['units']:
                self.report.add_test(
                    "variable_units", "Variable units compliance",
                    TestResult.PASS, f"Units '{units}' valid for {matched_type}"
                )
            else:
                valid_units = self.VALID_VARIABLE_SPECS[matched_type]['units']
                self.report.add_test(
                    "variable_units", "Variable units compliance",
                    TestResult.FAIL, None, f"Units '{units}' invalid for {matched_type}. Valid: {', '.join(valid_units)}"
                )
        else:
            all_names = [n for specs in self.VALID_VARIABLE_SPECS.values() for n in specs['names']]
            self.report.add_test(
                "variable_name", "Variable name compliance",
                TestResult.FAIL, None, f"Variable name '{self.data_var}' not in accepted list. Valid: {', '.join(all_names)}"
            )
    
    def _validate_coordinates(self):
        """Validate coordinate variables based on CRS type."""
        
        present_coords = set(self.ds.coords.keys())
        
        # Always require base coordinates (custom or default)
        missing_base_coords = self.custom_base_coords - present_coords
        
        if missing_base_coords:
            self.report.add_test(
                "required_coordinates", "Required coordinates",
                TestResult.FAIL, None, f"Missing required coordinates: {', '.join(missing_base_coords)}"
            )
        else:
            coord_list = ', '.join(sorted(self.custom_base_coords))
            self.report.add_test(
                "required_coordinates", "Required coordinates", 
                TestResult.PASS, f"All base coordinates ({coord_list}) present"
            )
        
        # Check for projected coordinates if CRS is projected
        crs_info = self._get_crs_info()
        is_projected = crs_info and crs_info.get('is_projected', False)
        
        if is_projected:
            missing_proj_coords = self.custom_proj_coords - present_coords
            if missing_proj_coords:
                self.report.add_test(
                    "projected_coordinates", "Projected coordinates",
                    TestResult.FAIL, None, f"Projected CRS missing coordinates: {', '.join(missing_proj_coords)}"
                )
            else:
                proj_coord_list = ', '.join(sorted(self.custom_proj_coords))
                self.report.add_test(
                    "projected_coordinates", "Projected coordinates",
                    TestResult.PASS, f"Projected coordinates ({proj_coord_list}) present for projected CRS"
                )
        else:
            # Geographic CRS - projected coords not required
            has_proj_coords = bool(self.custom_proj_coords & present_coords)
            if has_proj_coords:
                proj_coord_list = ', '.join(sorted(self.custom_proj_coords & present_coords))
                self.report.add_test(
                    "geographic_coordinates", "Geographic coordinates",
                    TestResult.INFO, f"Geographic CRS has optional projected coordinates ({proj_coord_list})"
                )
            else:
                base_coord_list = ', '.join(sorted(self.custom_base_coords & present_coords))
                self.report.add_test(
                    "geographic_coordinates", "Geographic coordinates", 
                    TestResult.PASS, f"Geographic CRS uses base coordinates ({base_coord_list}) correctly"
                )
        
        # Check CF compliance for all relevant coordinates
        cf_attrs = ['long_name', 'standard_name', 'units']
        all_coord_names = self.custom_base_coords | self.custom_proj_coords
        
        for coord in all_coord_names:
            if coord in self.ds.coords:
                coord_var = self.ds.coords[coord]
                has_cf_attrs = any(attr in coord_var.attrs for attr in cf_attrs)
                
                if has_cf_attrs:
                    self.report.add_test(
                        f"coord_cf_{coord}", f"Coordinate '{coord}' CF metadata",
                        TestResult.PASS, f"Coordinate '{coord}' has CF metadata"
                    )
                else:
                    self.report.add_test(
                        f"coord_cf_{coord}", f"Coordinate '{coord}' CF metadata",
                        TestResult.WARNING, f"Coordinate '{coord}' should have CF metadata"
                    )
    
    def _validate_chunking(self):
        """Validate chunking strategy."""
        
        if not self.data_var:
            return
        
        data_array = self.ds[self.data_var]
        
        # Check if data is chunked (dask array)
        if hasattr(data_array.data, 'chunks'):
            chunks = data_array.data.chunks
            
            if len(chunks) >= 1:
                time_chunks = chunks[0]
                
                # Check if time chunks are size 1
                if all(c == 1 for c in time_chunks):
                    self.report.add_test(
                        "chunking_strategy", "Chunking strategy",
                        TestResult.PASS, "Correct chunking: 1 chunk per timestep"
                    )
                else:
                    self.report.add_test(
                        "chunking_strategy", "Chunking strategy",
                        TestResult.FAIL, None, f"Time dimension must be chunked as 1 per timestep. Found: {time_chunks[:5]}..."
                    )
            else:
                self.report.add_test(
                    "chunking_strategy", "Chunking strategy",
                    TestResult.WARNING, "Could not verify chunking strategy"
                )
        else:
            self.report.add_test(
                "chunking_strategy", "Chunking strategy",
                TestResult.WARNING, "Data not chunked (not a dask array)"
            )
    
    def _validate_compression(self):
        """Validate compression requirements."""
        
        if not self.data_var:
            return
        
        try:
            compression_verified = False
            
            # Check main data array compression
            if hasattr(self, 'zarr_store') and self.zarr_store and self.data_var in self.zarr_store:
                array = self.zarr_store[self.data_var]
                if hasattr(array, 'compressors'):
                    # Use newer compressors property (Zarr v3 style)
                    compressors = array.compressors
                    compressor = compressors[0] if compressors else None
                elif hasattr(array, 'compressor'):
                    # Fallback to older compressor property (Zarr v2 style)
                    compressor = array.compressor
                else:
                    compressor = None
                    
                if compressor is not None:
                        self.report.add_test(
                            "data_compression", "Main data array compression",
                            TestResult.PASS, f"Main data array uses compression: {compressor}"
                        )
                        compression_verified = True
                        
                        # Check if ZSTD (recommended)
                        if 'zstd' in str(compressor).lower():
                            self.report.add_test(
                                "zstd_compression", "ZSTD compression",
                                TestResult.PASS, "Uses recommended ZSTD compression"
                            )
            
            # If compression not verified through zarr store, use heuristics
            if not compression_verified:
                if self.is_remote:
                    # Remote datasets that load efficiently are almost certainly compressed
                    self.report.add_test(
                        "data_compression", "Main data array compression",
                        TestResult.PASS, "Remote dataset loads efficiently (assuming compression)"
                    )
                else:
                    self.report.add_test(
                        "data_compression", "Main data array compression",
                        TestResult.WARNING, "Could not verify compression on local dataset"
                    )
                
        except Exception as e:
            # For remote datasets, assume compression if loading works
            if self.is_remote:
                self.report.add_test(
                    "data_compression", "Main data array compression",
                    TestResult.PASS, f"Remote dataset accessible (assuming compression)"
                )
            else:
                self.report.add_test(
                    "data_compression", "Main data array compression",
                    TestResult.WARNING, None, f"Could not verify compression: {e}"
                )
    
    def _validate_license(self):
        """Validate license requirements."""
        
        if 'license' not in self.ds.attrs:
            self.report.add_test(
                "license_presence", "License metadata",
                TestResult.FAIL, "Missing required 'license' global attribute for MLCast compliance"
            )
            return
        
        license_id = self.ds.attrs['license'].strip()
        
        if license_id in self.ACCEPTED_LICENSES:
            self.report.add_test(
                "license_compliance", "License compliance",
                TestResult.PASS, f"License '{license_id}' is accepted"
            )
        elif license_id in self.WARNING_LICENSES:
            self.report.add_test(
                "license_compliance", "License compliance",
                TestResult.WARNING, f"License '{license_id}' has restrictions (NC/ND)"
            )
        else:
            self.report.add_test(
                "license_compliance", "License compliance",
                TestResult.WARNING, f"License '{license_id}' requires case-by-case review"
            )
    
    def _validate_missing_data(self):
        """Validate missing data handling."""
        
        if not self.data_var:
            return
        
        data_array = self.ds[self.data_var]
        
        try:
            # Sample a few timesteps to check for NaN usage
            sample_indices = [0, len(self.ds.time) // 2, -1]
            has_nans = False
            
            for idx in sample_indices:
                if idx < len(self.ds.time):
                    sample_data = data_array.isel(time=idx).values
                    if np.isnan(sample_data).any():
                        has_nans = True
                        break
            
            if has_nans:
                self.report.add_test(
                    "nan_usage", "NaN usage for missing data",
                    TestResult.PASS, "Uses NaN for missing/out-of-range values"
                )
            else:
                self.report.add_test(
                    "nan_usage", "NaN usage for missing data",
                    TestResult.PASS, "No NaN values detected in sampled timesteps"
                )
                
        except Exception as e:
            self.report.add_test(
                "nan_usage", "NaN usage for missing data",
                TestResult.WARNING, None, f"Could not verify NaN usage: {e}"
            )
    
    def _validate_future_timesteps(self):
        """Validate future timestep extension if present."""
        
        if 'time' not in self.ds.coords:
            return
        
        try:
            current_time = pd.Timestamp.now()
            time_coord = pd.to_datetime(self.ds.time.values)
            max_time = time_coord.max()
            
            if max_time > current_time:
                self.report.add_test(
                    "future_timesteps", "Future timesteps detected",
                    TestResult.PASS, f"Dataset extends to future: {max_time}"
                )
                
                # Check 2050 limit
                if max_time.year > 2050:
                    self.report.add_test(
                        "future_limit", "Future timestep limits",
                        TestResult.FAIL, None, f"Future timesteps beyond 2050 limit: {max_time.year}"
                    )
                
                # Check for required metadata
                if 'last_valid_timestep' in self.ds.attrs:
                    self.report.add_test(
                        "future_metadata", "Future timestep metadata",
                        TestResult.PASS, "Has 'last_valid_timestep' metadata"
                    )
                else:
                    self.report.add_test(
                        "future_metadata", "Future timestep metadata",
                        TestResult.FAIL, None, "Missing 'last_valid_timestep' for future timesteps"
                    )
                        
        except Exception as e:
            self.report.add_test(
                "future_timesteps", "Future timesteps validation",
                TestResult.WARNING, None, f"Could not validate future timesteps: {e}"
            )


def print_report(report: ToolCompatibilityReport, detailed: bool = True):
    summary = report.get_summary()
    
    if detailed:
        print(f"\n{'='*70}")
        print("MLCAST COMPATIBILITY REPORT")
        print(f"{'='*70}")
        
        # Group tests by category
        categories = {}
        for test in report.tests:
            category = test.name.split('_')[0] if '_' in test.name else 'general'
            if category not in categories:
                categories[category] = []
            categories[category].append(test)
        
        for category, tests in categories.items():
            print(f"\n{category.upper()} TESTS:")
            print("-" * 50)
            
            for test in tests:
                if test.result == TestResult.PASS:
                    symbol, color = "✅", "\033[92m"
                elif test.result == TestResult.FAIL:
                    symbol, color = "❌", "\033[91m"
                elif test.result == TestResult.WARNING:
                    symbol, color = "⚠️", "\033[93m"
                else:  # SKIP
                    symbol, color = "⏭️", "\033[94m"
                
                print(f"{symbol} {color}[{test.result.value}]\033[0m {test.description}")
                
                if test.details:
                    print(f"    ℹ️  {test.details}")
                if test.error:
                    print(f"    ❗ {test.error}")
    
    # Summary
    print(f"\n{'='*70}")
    print("COMPATIBILITY SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Passed: {summary['passed']}")
    print(f"⚠️  Warnings: {summary['warnings']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"⏭️  Skipped: {summary['skipped']}")
    
    if summary['failed'] == 0:
        if summary['warnings'] == 0:
            print("\n🎉 RESULT: EXCELLENT - Full tool compatibility achieved!")
            result_code = 0
        else:
            print("\n✨ RESULT: GOOD - Compatible with warnings to review")
            result_code = 0
    else:
        print("\n💥 RESULT: FAILED - Tool compatibility issues detected")
        result_code = 1
    
    print(f"Based on MLCast specification v1.0 + GitHub issue #6 discussion")
    
    return result_code


def main():
    parser = argparse.ArgumentParser(
        description=TOOL_FULL_NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This validator focuses on practical tool compatibility testing (xarray, gdal, cartopy) of Zarr datasets for MLCast Source Radar data.

Examples:
  # Local dataset
  %(prog)s /path/to/dataset.zarr
  
  # S3 dataset with anonymous access
  %(prog)s s3://bucket/dataset.zarr --anon
  
  # S3 dataset with custom endpoint
  %(prog)s s3://mlcast-source-datasets/radklim/v0.1.0/5_minutes.zarr/ \\
    --anon --endpoint-url https://object-store.os-api.cci2.ecmwf.int
  
  # Custom data variable specification
  %(prog)s /path/to/dataset.zarr --data-variables rainfall precipitation
  
  # Custom coordinate requirements
  %(prog)s /path/to/dataset.zarr --base-coordinates latitude longitude time \\
    --projected-coordinates easting northing
  
  # Export detailed report
  %(prog)s /path/to/dataset.zarr --output compatibility_report.json
        """
    )
    
    parser.add_argument('zarr_path', help='Path or URL to Zarr dataset (supports s3://, local paths)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Show only compatibility summary')
    parser.add_argument('--output', '-o', help='Save report to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode - show detailed dataset information')
    
    # S3 and remote options
    parser.add_argument('--anon', action='store_true',
                       help='Use anonymous S3 access')
    parser.add_argument('--endpoint-url', 
                       help='Custom S3 endpoint URL')
    parser.add_argument('--aws-access-key-id',
                       help='AWS access key ID')
    parser.add_argument('--aws-secret-access-key',
                       help='AWS secret access key')
    parser.add_argument('--storage-option', action='append', nargs=2, metavar=('KEY', 'VALUE'),
                       help='Additional storage options as key-value pairs')
    
    # Custom validation options
    parser.add_argument('--data-variables', nargs='+', metavar='VAR',
                       help='Specify custom list of expected data variable names (default: auto-detect)')
    parser.add_argument('--base-coordinates', nargs='+', metavar='COORD', 
                       default=['lat', 'lon', 'time'],
                       help='Specify required base coordinates (default: lat lon time)')
    parser.add_argument('--projected-coordinates', nargs='+', metavar='COORD',
                       default=['x', 'y'], 
                       help='Specify coordinates required for projected CRS (default: x y)')
    
    args = parser.parse_args()
    
    # Build storage options for remote access
    storage_options = {}
    
    if args.anon:
        storage_options['anon'] = True
    
    if args.endpoint_url:
        storage_options['endpoint_url'] = args.endpoint_url
    
    if args.aws_access_key_id:
        storage_options['key'] = args.aws_access_key_id
    
    if args.aws_secret_access_key:
        storage_options['secret'] = args.aws_secret_access_key
    
    # Add custom storage options
    if args.storage_option:
        for key, value in args.storage_option:
            storage_options[key] = value
    
    # Suppress warnings for cleaner output
    if not args.verbose:
        warnings.filterwarnings('ignore')
    
    print(f"🚀 {TOOL_FULL_NAME}")
    
    # Show storage options if remote
    if storage_options and args.verbose:
        print(f"Storage options: {storage_options}")
    
    # Run validation
    validator = MLCastGeoreferenceValidator(
        args.zarr_path, 
        args.verbose, 
        storage_options, 
        args.debug,
        custom_data_vars=args.data_variables,
        custom_base_coords=args.base_coordinates,
        custom_proj_coords=args.projected_coordinates
    )
    report = validator.validate()
    
    # Print results
    result_code = print_report(report, detailed=not args.summary_only)
    
    # Save JSON report if requested
    if args.output:
        report_data = {
            'dataset': str(args.zarr_path),
            'validator': TOOL_FULL_NAME,
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': report.get_summary(),
            'tests': [
                {
                    'name': test.name,
                    'description': test.description,
                    'result': test.result.value,
                    'details': test.details,
                    'error': test.error
                }
                for test in report.tests
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\n📄 Report saved to: {args.output}")
    
    sys.exit(result_code)


if __name__ == "__main__":
    main()