from typing import Dict

import requests
import xarray as xr
from isodate import ISO8601Error, parse_datetime
from packaging.version import InvalidVersion
from packaging.version import parse as parse_version
from parse import compile as parse_compile
from requests import RequestException

from ...specs.base import ValidationReport
from ...utils.logging_decorator import log_function_call
from . import SECTION_ID as PARENT_SECTION_ID

SECTION_ID = f"{PARENT_SECTION_ID}.4"
CREATED_BY_FORMAT = "{name} <{email}>"
GITHUB_WITH_VERSION_FORMAT = "https://github.com/{org}/{repo}@{version}"
SOURCE_ORG_ID_FORMAT = "{country}-{institution}"
EXPECTED_GITHUB_ORG = "mlcast-community"
EXPECTED_REPO_PATTERN = "mlcast-dataset-{organisation_id}-{dataset_name}"

_CREATED_BY_PARSER = parse_compile(CREATED_BY_FORMAT)
_GITHUB_WITH_VERSION_PARSER = parse_compile(GITHUB_WITH_VERSION_FORMAT)
_SOURCE_ORG_ID_PARSER = parse_compile(SOURCE_ORG_ID_FORMAT)
_EXPECTED_REPO_PARSER = parse_compile(EXPECTED_REPO_PATTERN)
_GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "mlcast-dataset-validator",
}


def parse_created_by(value: str) -> Dict[str, str]:
    """
    Parse creator info in 'Name <email>' format.

    Parameters
    ----------
    value : str
        Creator string to parse.

    Returns
    -------
    dict
        Dictionary with keys 'name' and 'email'.

    Raises
    ------
    ValueError
        If the value does not match or has invalid components.
    """
    try:
        parsed = _CREATED_BY_PARSER.parse(value.strip())
    except AttributeError:
        raise ValueError("Value is not a string")
    if not parsed:
        raise ValueError("Does not match 'Name <email>'")
    name = parsed["name"].strip()
    email = parsed["email"].strip()
    if not name:
        raise ValueError("Missing name")
    if not email or "@" not in email:
        raise ValueError("Missing or invalid email")
    if any(ch.isspace() for ch in email):
        raise ValueError("Email contains whitespace")
    return {"name": name, "email": email}


def parse_github_url_with_version(value: str) -> Dict[str, str]:
    """
    Parse GitHub URL with version suffix and enforce mlcast conventions.

    Parameters
    ----------
    value : str
        URL string to parse.

    Returns
    -------
    dict
        Dictionary with keys 'org', 'repo', and 'version'.

    Raises
    ------
    ValueError
        If the value does not match or has invalid components.
    """
    try:
        parsed = _GITHUB_WITH_VERSION_PARSER.parse(value.strip())
    except AttributeError:
        raise ValueError("Value is not a string")
    if not parsed:
        raise ValueError("Does not match https://github.com/{org}/{repo}@{version}")
    org = parsed["org"].strip()
    repo = parsed["repo"].strip()
    version = parsed["version"].strip()
    if not (org and repo and version):
        raise ValueError("Missing org, repo, or version")
    if any(any(ch.isspace() for ch in part) for part in (org, repo, version)):
        raise ValueError("Contains whitespace in org/repo/version")
    if org != EXPECTED_GITHUB_ORG:
        raise ValueError(f"GitHub organisation must be '{EXPECTED_GITHUB_ORG}'")
    parsed_repo = _EXPECTED_REPO_PARSER.parse(repo)
    if not parsed_repo:
        raise ValueError(f"Repository must follow '{EXPECTED_REPO_PATTERN}'")
    org_id = parsed_repo["organisation_id"]
    dataset_name = parsed_repo["dataset_name"]
    if not org_id or not dataset_name:
        raise ValueError(
            "Repository must include non-empty organisation_id and dataset_name"
        )
    return {"org": org, "repo": repo, "version": version}


def github_repo_exists(org: str, repo: str) -> bool:
    """Return True if GitHub repository exists, False if 404, else raise on other issues."""
    base = f"https://api.github.com/repos/{org}/{repo}"
    response = requests.get(base, headers=_GITHUB_HEADERS, timeout=5)
    if response.status_code == 404:
        return False
    if not response.ok:
        raise RuntimeError(
            f"GitHub repo check failed with status {response.status_code}"
        )
    return True


def github_ref_exists(org: str, repo: str, ref: str) -> str | None:
    """
    Check if a GitHub revision exists (tag, branch, or commit).

    Returns the matched API path if found, None if not found, otherwise raises.
    """
    base = f"https://api.github.com/repos/{org}/{repo}"
    ref_paths = [
        f"/git/refs/tags/{ref}",
        f"/git/refs/heads/{ref}",
        f"/commits/{ref}",
    ]
    for path in ref_paths:
        response = requests.get(base + path, headers=_GITHUB_HEADERS, timeout=5)
        if response.status_code == 404:
            continue
        if response.ok:
            return path
        raise RuntimeError(
            f"GitHub ref check failed with status {response.status_code} for {path}"
        )
    return None


def parse_source_org_id(value: str) -> Dict[str, str]:
    """
    Parse source organisation id '<ISO-country-code>-<institution-identifier>'.

    Parameters
    ----------
    value : str
        Source organisation identifier to parse.

    Returns
    -------
    dict
        Dictionary with keys 'country' and 'institution'.

    Raises
    ------
    ValueError
        If the value does not match or has invalid components.
    """
    try:
        parsed = _SOURCE_ORG_ID_PARSER.parse(value.strip())
    except AttributeError:
        raise ValueError("Value is not a string")
    if not parsed:
        raise ValueError("Does not match '<ISO-country-code>-<institution-identifier>'")
    country = parsed["country"]
    institution = parsed["institution"].strip()
    if len(country) != 2:
        raise ValueError("Country code must be 2 uppercase letters")
    if not country.isalpha() or not country.isupper():
        raise ValueError("Country code must use uppercase alphabetic characters")
    if not institution:
        raise ValueError("Missing institution identifier")
    if institution[0] in "-_" or institution[-1] in "-_":
        raise ValueError("Institution identifier cannot start/end with '-' or '_'")
    if not all(ch.isalnum() or ch in "-_" for ch in institution):
        raise ValueError("Institution identifier contains invalid characters")
    return {"country": country, "institution": institution}


@log_function_call
def check_mlcast_metadata(ds: xr.Dataset) -> ValidationReport:
    """Validate required MLCast-specific global attributes."""
    report = ValidationReport()
    attrs = ds.attrs

    created_on = attrs.get("mlcast_created_on")
    if created_on is None:
        report.add(
            SECTION_ID,
            "Global attribute 'mlcast_created_on'",
            "FAIL",
            "Missing required creation timestamp in ISO 8601 format",
        )
    else:
        try:
            parsed_dt = parse_datetime(created_on.strip())
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_created_on'",
                "PASS",
                f"Creation timestamp parsed ({parsed_dt})",
            )
        except (ISO8601Error, ValueError, TypeError, AttributeError) as exc:
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_created_on'",
                "FAIL",
                f"Value '{created_on}' is not a valid ISO 8601 datetime string: {exc}",
            )

    created_by = attrs.get("mlcast_created_by")
    if created_by is None:
        report.add(
            SECTION_ID,
            "Global attribute 'mlcast_created_by'",
            "FAIL",
            "Missing required creator contact in 'Name <email>' format",
        )
    else:
        try:
            parsed_by = parse_created_by(created_by)
            detail = (
                f"Creator contact present ({parsed_by['name']} <{parsed_by['email']}>)"
            )
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_created_by'",
                "PASS",
                detail,
            )
        except ValueError as exc:
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_created_by'",
                "FAIL",
                f"Creator contact '{created_by}' is not in 'Name <email>' format: {exc}",
            )

    created_with = attrs.get("mlcast_created_with")
    if created_with is None:
        report.add(
            SECTION_ID,
            "Global attribute 'mlcast_created_with'",
            "FAIL",
            "Missing required creator software GitHub URL with version (e.g. https://github.com/org/repo@v0.1.0)",
        )
    else:
        try:
            parsed_created_with = parse_github_url_with_version(created_with)
            url_detail = (
                f"Creator software GitHub URL parsed "
                f"({parsed_created_with['org']}/{parsed_created_with['repo']}@{parsed_created_with['version']})"
            )
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_created_with'",
                "PASS",
                url_detail,
            )
            try:
                repo_exists = github_repo_exists(
                    parsed_created_with["org"], parsed_created_with["repo"]
                )
                if repo_exists:
                    report.add(
                        SECTION_ID,
                        "Creator software GitHub repository",
                        "PASS",
                        "Repository exists on GitHub",
                    )
                    try:
                        ref_path = github_ref_exists(
                            parsed_created_with["org"],
                            parsed_created_with["repo"],
                            parsed_created_with["version"],
                        )
                        if ref_path:
                            report.add(
                                SECTION_ID,
                                "Creator software GitHub revision",
                                "PASS",
                                f"Revision found at {ref_path}",
                            )
                        else:
                            report.add(
                                SECTION_ID,
                                "Creator software GitHub revision",
                                "FAIL",
                                "Revision not found as tag, branch, or commit",
                            )
                    except RequestException as exc:
                        report.add(
                            SECTION_ID,
                            "Creator software GitHub revision",
                            "WARNING",
                            f"Could not verify revision due to network error: {exc}",
                        )
                    except RuntimeError as exc:
                        report.add(
                            SECTION_ID,
                            "Creator software GitHub revision",
                            "WARNING",
                            f"GitHub revision check returned unexpected response: {exc}",
                        )
                else:
                    report.add(
                        SECTION_ID,
                        "Creator software GitHub repository",
                        "WARNING",
                        "Repository not found on GitHub",
                    )
            except RequestException as exc:
                report.add(
                    SECTION_ID,
                    "Creator software GitHub repository",
                    "WARNING",
                    f"Could not verify repository due to network error: {exc}",
                )
            except RuntimeError as exc:
                report.add(
                    SECTION_ID,
                    "Creator software GitHub repository",
                    "WARNING",
                    f"GitHub repository check returned unexpected response: {exc}",
                )
        except ValueError as exc:
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_created_with'",
                "FAIL",
                f"Value '{created_with}' is not a GitHub URL with an @version suffix: {exc}",
            )

    dataset_version = attrs.get("mlcast_dataset_version")
    if dataset_version is None:
        report.add(
            SECTION_ID,
            "Global attribute 'mlcast_dataset_version'",
            "FAIL",
            "Missing required dataset specification version (semver or calver)",
        )
    else:
        try:
            parsed_version = parse_version(dataset_version.strip())
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_dataset_version'",
                "PASS",
                f"Dataset specification version parsed ({parsed_version})",
            )
        except (InvalidVersion, ValueError, AttributeError) as exc:
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_dataset_version'",
                "FAIL",
                f"Version '{dataset_version}' is not valid semver or calver: {exc}",
            )

    source_org_id = attrs.get("mlcast_source_org_id")
    if source_org_id is None:
        report.add(
            SECTION_ID,
            "Global attribute 'mlcast_source_org_id'",
            "FAIL",
            "Missing required source organisation identifier '<ISO-country-code>-<institution-identifier>'",
        )
    else:
        try:
            parsed_source_org = parse_source_org_id(source_org_id)
            detail = (
                "Source organisation identifier present with expected pattern "
                f"({parsed_source_org['country']}-{parsed_source_org['institution']})"
            )
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_source_org_id'",
                "PASS",
                detail,
            )
        except ValueError as exc:
            report.add(
                SECTION_ID,
                "Global attribute 'mlcast_source_org_id'",
                "FAIL",
                f"Identifier '{source_org_id}' does not match '<ISO-country-code>-<institution-identifier>': {exc}",
            )

    return report
