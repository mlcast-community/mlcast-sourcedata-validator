# MLCast Radar Data Archive Specification
**Version 1.0**  
**Status: Stable**

## 1. Introduction

This document specifies the requirements for 2D radar precipitation and reflectivity composite datasets to be included in the MLCast data collection. The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in RFC 2119.

## 2. Scope

This specification applies to 2D radar composite datasets (merged from multiple radar sources) intended for machine learning applications in weather and climate research. Single-radar datasets are explicitly excluded from this specification.

## 3. Dataset Content Requirements

### 3.1 Spatial Requirements

The dataset MUST provide 2D radar composites with a spatial resolution of 1 kilometer or finer.

The valid sensing area MUST support at least one 256×256 pixel square crop that is fully contained within the radar sensing range (i.e., no missing or out-of-range pixels within the crop area).

The spatial domain, including resolution, size, and geographical coverage, MUST remain constant across all timesteps in the archive.

### 3.2 Temporal Requirements

The dataset MUST contain a minimum of 3 years of continuous temporal coverage.

The timestep MAY be variable throughout the archive (for example, older data collected at 10-15 minute intervals and recent data at 5-6 minute intervals).

### 3.3 Data Variable Requirements

The data variable MUST be expressed in one of the following units:
- millimeters (mm) for precipitation depth
- millimeters per hour (mm/h) for precipitation rate
- decibels of reflectivity factor (dBZ) for radar reflectivity

## 4. Licensing Requirements

The dataset MUST include a global `license` attribute containing a valid SPDX identifier.

The following licenses are RECOMMENDED and will be automatically accepted:
- `CC-BY` (Creative Commons Attribution)
- `CC-BY-SA` (Creative Commons Attribution-ShareAlike)
- `OGL` (Open Government License) and equivalent national variants

Licenses with `NC` (NonCommercial) or `ND` (NoDerivatives) restrictions SHOULD generate warnings during validation but MAY be accepted after review.

Any license not explicitly listed above MUST be reviewed on a case-by-case basis by the MLCast community before acceptance.

## 5. Technical Format Requirements

### 5.1 Zarr Format

The dataset MUST use Zarr version 2 or version 3 format.

If Zarr version 2 is used, the dataset MUST include consolidated metadata.

### 5.2 Compression

The main data arrays MUST use compression to reduce storage requirements.

ZSTD compression is RECOMMENDED for optimal performance of the main data arrays.

Coordinate arrays MAY use different compression algorithms (e.g., lz4) as appropriate.

Compression MUST be applied at the array level using any numcodecs compression algorithms supported by Zarr.

### 5.3 Georeferencing

The dataset MUST include proper georeferencing information following the GeoZarr specification.

The georeferencing information MUST be correctly intepreted by GDAL and cartopy.

The data variable MUST include a `grid_mapping` attribute that references the coordinate reference system (crs) variable.

The crs variable MUST include both a `spatial_ref` and a `crs_wkt` attribute with a WKT (Well-Known Text) string following CF conventions, including the BBOX (bounding box) specification for compatibility with cartopy and global projections.


### 5.4 Data Structure

The main data variable MUST be encoded with dimensions in the order: time × height (y, lat) × width (x, lon).

The data type MUST be floating-point (float16, float32, or float64) represented in original units values. This requirement ensures native NaN representation for missing data, out-of-domain pixels, and orographic blocking without requiring scale/offset/nodata value interpretation across different tools.

### 5.5 Coordinate Variables

Coordinate variable names MUST follow CF conventions and use the following names:
- `x` for projected x-coordinate
- `y` for projected y-coordinate
- `lat` for latitude
- `lon` for longitude
- `time` for the temporal coordinate

All coordinate variables SHOULD include CF-compliant attributes (`long_name`, `standard_name` and `units`).

### 5.6 Data Variable Naming and attributes

The data variable name SHOULD be a CF convention stadard name or use a sensible name from the ECMWF parameter database. the data variable MUST inlude the `long_name`, `standard_name` and `units` attributes following CF conventions.

- For precipitation rate (mm/h):
    - the variable name MUST be one of: `mmh`, `rr`, `tprate`, `prate`, `rain_rate`, `rainfall_flux` or `rainfall_rate`
    - `units` MUST be one of: `kg m-2 h-1` or `mm h-1` or `mm/h`

- For radar reflectivity (dBZ):
    - the variable name MUST be one of: `equivalent_reflectivity_factor`, `dbz`, or `rare`
    - `units` MUST be one of: `dBZ`

- For precipitation amount (mm):
    - the variable name MUST be one of: `rainfall_amount`, `mm`, `precipitation_amount`, or `tp`.
    - `units` MUST be one of: `kg m-2` or `mm`

Variable names MAY use lowercase, mixed case, or uppercase format.

### 5.7 Chunking Strategy

The dataset MUST use a chunking strategy of 1 × height × width (one chunk per timestep).

## 6. Missing Data and Special Values

NaN values MUST be used to indicate:
- Pixels outside the radar sensing range
- Pixels blocked by orography
- Partially or totally missing scans

Missing timesteps that should have been recorded but were not available MUST be represented as arrays filled with NaN values.

## 7. Variable Timestep Handling

If the archive contains variable timesteps due to changing acquisition frequencies, the timesteps SHOULD follow the natural timestepping of the data collection.

If the archive does not have a single regular timestep throughout, a global attribute named `consistent_timestep_start` MAY be included to indicate the first timestamp where regular timestepping begins.

## 8. Future Timestep Extension

The dataset MAY extend timesteps into the future to allow for near-realtime updates without regenerating the entire archive.

If future timesteps are included:
- Future timesteps MUST have regular timestepping corresponding to the highest (most recent) frequency present in the data
- Future timesteps MUST be filled entirely with NaN values to maximize compression efficiency
- Future timesteps MUST NOT extend beyond the year 2050
- A global attribute named `last_valid_timestep` MUST be present to indicate the most recent non-NaN filled timestep

## 9. Global Attributes

The following global attributes are REQUIRED:
- `license`: A valid SPDX identifier string

The following global attributes are CONDITIONAL:
- `consistent_timestep_start`: An ISO 8601 timestamp MAY be present if the dataset has variable timestepping
- `last_valid_timestep`: An ISO 8601 timestamp is REQUIRED if the dataset includes future timesteps

## 10. Validation

A compliant dataset MUST pass validation checks for all REQUIRED elements specified in this document.

### 10.1 Tool Compatibility Testing

The most reccent version of the following tools MUST successfully open and interpret (including proper georeferencing) the zarr:
- xarray
- GDAL
- cartopy

### 10.2 Validation Reporting

Validation tools SHOULD report:
- Failures for any MUST/REQUIRED violations
- Warnings for any SHOULD/RECOMMENDED violations
- Information about OPTIONAL elements
- Results of practical tool compatibility tests

## References

- CF Conventions: https://cfconventions.org/
- CF Conventions standard names: https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
- SPDX License List: https://spdx.org/licenses/
- GeoZarr Specification: https://github.com/zarr-developers/geozarr-spec
- ECMWF Parameter Database: https://apps.ecmwf.int/codes/grib/param-db/
- RFC 2119: https://www.ietf.org/rfc/rfc2119.txt
- WKT in CF Conventions: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.9/cf-conventions.html#appendix-grid-mappings