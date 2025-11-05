from ...specs.base import ValidationReport, ValidationResult


def check_cartopy_compatibility(ds) -> ValidationReport:
    """
    Dummy function to check compatibility of the dataset with cartopy.

    Args:
        ds: The xarray Dataset to check.

    Returns:
        ValidationReport: A placeholder report for cartopy compatibility checks.
    """
    report = ValidationReport()
    report += ValidationResult(
        section="10.3",
        level="WARNING",
        requirement="cartopy compatibility check",
        detail="This is a dummy cartopy compatibility check.",
    )
    return report
