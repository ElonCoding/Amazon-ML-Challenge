#!/usr/bin/env python3
"""Create the final challenge ZIP with the required archive layout."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def required_files() -> list[tuple[Path, str]]:
    source_root = ROOT / "business_entity_resolution" / "code"
    entries = [
        (ROOT / "output" / "matching_results.tsv", "output/matching_results.tsv"),
        (ROOT / "output" / "candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        (
            ROOT / "business_entity_resolution" / "Documentation_template.md",
            "Documentation_template.md",
        ),
        (source_root / "README.md", "code/business_entity_resolution/README.md"),
        (source_root / "requirements.txt", "code/business_entity_resolution/requirements.txt"),
    ]
    src = source_root / "src"
    if src.is_dir():
        entries.extend(
            (path, f"code/business_entity_resolution/src/{path.relative_to(src).as_posix()}")
            for path in sorted(src.rglob("*.py"))
            if "__pycache__" not in path.parts
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-name", default="team_innovators")
    parser.add_argument("--output", type=Path, default=None, help="ZIP output path")
    parser.add_argument("--check-ids", action="store_true", help="also validate target IDs against test sources")
    args = parser.parse_args()

    team_name = args.team_name.strip()
    if not team_name or any(ch in team_name for ch in "/\\"):
        parser.error("--team-name must be a non-empty filename component")

    destination = args.output or ROOT / f"{team_name}_submission.zip"
    if not destination.is_absolute():
        destination = ROOT / destination

    entries = required_files()
    missing = [str(source) for source, _ in entries if not source.is_file()]
    if missing:
        parser.error("Required submission files are missing:\n  " + "\n  ".join(missing))
    methodology = ROOT / "business_entity_resolution" / "Documentation_template.md"
    remaining = re.findall(r"\[[^\]]+\]", methodology.read_text(encoding="utf-8"))
    if remaining:
        parser.error(
            "Fill in the measured methodology fields before packaging; remaining markers: "
            + ", ".join(sorted(set(remaining)))
        )
    if not any(name.startswith("code/business_entity_resolution/src/") for _, name in entries):
        parser.error("No Python source files found under business_entity_resolution/code/src")

    validator = ROOT / "dataset" / "student_resource" / "utils" / "validate_submission.py"
    validation_command = [
        sys.executable,
        str(validator),
        "--matching",
        str(ROOT / "output" / "matching_results.tsv"),
        "--candidate",
        str(ROOT / "output" / "candidate_pairs.tsv"),
        "--test-dir",
        str(ROOT / "dataset" / "student_resource" / "dataset" / "test"),
    ]
    if args.check_ids:
        validation_command.append("--check-ids")
    print("Validating submission files before packaging...")
    validation = subprocess.run(validation_command, cwd=ROOT, check=False)
    if validation.returncode:
        parser.error("Submission validation failed; fix the listed issues before packaging.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for source, arcname in entries:
            archive.write(source, arcname)

    print(f"Created {destination}")
    print(f"Included {len(entries)} files with the challenge submission layout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
