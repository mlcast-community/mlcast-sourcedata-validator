from ...specs.base import ValidationReport, ValidationResult

try:
    from osgeo import osr

    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False


def check_gdal_compatibility(ds) -> ValidationReport:
    """
    Check compatibility of the dataset with GDAL.

    Args:
        ds: The xarray Dataset to check.

    Returns:
        ValidationReport: Report of the compatibility checks.
    """
    report = ValidationReport()

    if not GDAL_AVAILABLE:
        report += ValidationResult(
            section="10.1",
            level="WARNING",
            requirement="GDAL compatibility check",
            detail="GDAL compatibility check skipped: GDAL is not installed.",
        )
    else:
        try:
            crs_var = ds["crs"]
            if "crs_wkt" not in crs_var.attrs:
                report += ValidationResult(
                    section="10.1",
                    level="FAIL",
                    requirement="GDAL compatibility check",
                    detail="No crs_wkt attribute found in the dataset.",
                )
                return report

            wkt_string = crs_var.attrs["crs_wkt"]
            srs = osr.SpatialReference()
            err = srs.ImportFromWkt(wkt_string)

            if err == 0:  # OGRERR_NONE
                report += ValidationResult(
                    section="10.1",
                    level="PASS",
                    requirement="GDAL compatibility check",
                    detail=f"Successfully parsed WKT, Authority: {srs.GetAuthorityName(None)}",
                )

                if srs.IsProjected():
                    proj_name = srs.GetAttrValue("PROJECTION")
                    report += ValidationResult(
                        section="10.1",
                        level="PASS",
                        requirement="Projection type check",
                        detail=f"Projected CRS with projection: {proj_name}",
                    )
                elif srs.IsGeographic():
                    report += ValidationResult(
                        section="10.1",
                        level="PASS",
                        requirement="Projection type check",
                        detail="Geographic CRS (lat/lon)",
                    )
                else:
                    report += ValidationResult(
                        section="10.1",
                        level="WARNING",
                        requirement="Projection type check",
                        detail="CRS type unclear - neither clearly projected nor geographic.",
                    )
            else:
                report += ValidationResult(
                    section="10.1",
                    level="FAIL",
                    requirement="GDAL compatibility check",
                    detail=f"GDAL failed to parse WKT (error code: {err}).",
                )
        except Exception as e:
            report += ValidationResult(
                section="10.1",
                level="FAIL",
                requirement="GDAL compatibility check",
                detail=str(e),
            )

    return report
