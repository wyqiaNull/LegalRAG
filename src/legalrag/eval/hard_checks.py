"""Collect deterministic pytest assertions into a machine-readable summary."""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _junit_counts(path: Path) -> tuple[int, int, int, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("./testsuite")
    return tuple(
        sum(int(suite.attrib.get(field, 0)) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    )


def run_hard_checks(output_dir: str | Path, *, include_real_services: bool) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    junit_path = output / "hard-checks.xml"
    targets = [
        "tests/assertions/test_version_management.py",
        "tests/unit/test_versioning.py",
        "tests/unit/test_permission_filter.py",
        "tests/unit/test_query_permissions.py",
        "tests/integration/test_v02_citation_pipeline.py",
    ]
    if include_real_services:
        targets.extend(
            [
                "tests/integration/test_postgres_persistence.py",
                "tests/integration/test_redis_conversation_store.py",
            ]
        )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"--junitxml={junit_path}",
            *targets,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    tests, failures, errors, skipped = _junit_counts(junit_path)
    return {
        "passed": tests - failures - errors - skipped,
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "exit_code": completed.returncode,
        "real_services": include_real_services,
        "unauthorized_candidate_count": 0 if completed.returncode == 0 else None,
        "evidence_scope": {
            "quality": "offline deterministic assertions",
            "governance": (
                "real PostgreSQL and Redis integration"
                if include_real_services
                else "offline stores only"
            ),
        },
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
