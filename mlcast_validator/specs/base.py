from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from rich.console import Console
from rich.table import Table


# -------------------------
# Data structures
# -------------------------
@dataclass
class ValidationResult:
    section: str
    requirement: str
    level: str  # "FAIL" | "WARNING" | "PASS"
    detail: str = ""
    module: Optional[str] = None
    function: Optional[str] = None

    def __post_init__(self):
        valid_levels = {"FAIL", "WARNING", "PASS"}
        if self.level not in valid_levels:
            raise ValueError(
                f"Invalid level: {self.level}. Valid levels are: {', '.join(valid_levels)}."
            )


@dataclass
class ValidationReport:
    ok: bool = True
    results: List[ValidationResult] = field(default_factory=list)

    def add(self, section: str, requirement: str, level: str, detail: str = "") -> None:
        """
        Add a result to the validation report.

        Args:
            section (str): The section where the result occurred.
            requirement (str): The specific requirement that was evaluated.
            level (str): The severity level of the result ("FAIL", "WARNING", "INFO", "PASS").
            detail (str, optional): Additional details about the result. Defaults to an empty string.

        Returns:
            None
        """
        self.results.append(ValidationResult(section, requirement, level, detail))

    def summarize(self) -> str:
        """
        Summarize the validation report by counting results of each severity level.

        Returns:
            str: A summary string with counts of fails, warnings, and passes.
        """
        fails = sum(1 for r in self.results if r.level == "FAIL")
        warns = sum(1 for r in self.results if r.level == "WARNING")
        passes = sum(1 for r in self.results if r.level == "PASS")
        return f"Summary: {fails} fail(s), {warns} warning(s), {passes} pass(es)."

    def __iadd__(self, other: "ValidationReport") -> "ValidationReport":
        """
        Merge another ValidationReport into this one (in-place).

        Args:
            other (ValidationReport): The other validation report to merge.

        Returns:
            ValidationReport: The updated validation report (self).
        """
        if isinstance(other, ValidationReport):
            self.results.extend(other.results)
            self.ok = self.ok and other.ok
        elif isinstance(other, ValidationResult):
            self.results.append(other)
            if other.level == "FAIL":
                self.ok = False
        else:
            raise TypeError(
                "Can only add ValidationReport or ValidationResult instances."
            )
        return self

    def console_print(self) -> None:
        """
        Print all results in the validation report as a table using the rich library.

        Returns:
            None
        """
        console = Console()
        table = Table(title="Validation Report")

        table.add_column("Section", style="bold")
        table.add_column("Requirement", style="dim")
        table.add_column("Level", justify="center")
        table.add_column("Detail", style="italic")
        table.add_column(
            "Checking function", style="bold"
        )  # New column for function name

        level_emojis = {"FAIL": "❌", "WARNING": "⚠️", "PASS": "✅"}

        for result in self.results:
            if result.module and result.function:
                fn_fqn = f"{result.module}.{result.function}".removeprefix(
                    "mlcast_validator.checks."
                )
            else:
                fn_fqn = "N/A"
            table.add_row(
                result.section,
                result.requirement,
                level_emojis.get(result.level, result.level),
                result.detail,
                fn_fqn,
            )

        console.print(table)
        console.print(self.summarize())

    def has_fails(self) -> bool:
        """
        Check if the report contains any FAIL results.

        Returns:
            bool: True if there is at least one FAIL result, False otherwise.
        """
        return any(r.level == "FAIL" for r in self.results)
