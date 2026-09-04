"""Deployment reproducibility and bundled-model asset checks."""

from __future__ import annotations

import re
from pathlib import Path


def test_aarch64_installer_pins_sdk_sources_and_installs_urdf(project_root: Path) -> None:
    installer = (project_root / "deploy" / "install_aarch64.sh").read_text(encoding="utf-8")

    assert re.search(r"^CYCLONEDDS_COMMIT=[0-9a-f]{40}$", installer, re.MULTILINE)
    assert re.search(r"^UNITREE_SDK_COMMIT=[0-9a-f]{40}$", installer, re.MULTILINE)
    assert " pull " not in installer
    assert "configs/*.urdf" in installer
    assert "configs/UNITREE_ROS_LICENSE.txt" in installer
    assert "rev-parse HEAD" in installer
    assert re.search(r"status\s+\\\n\s+--porcelain=v1 --untracked-files=all", installer)
    assert "submodule status --recursive" in installer
    assert "Refusing vendor repository with unpinned submodule content" in installer

    source_assignment = re.search(r'^CYCLONEDDS_SRC="(?P<value>.+)"$', installer, re.MULTILINE)
    build_assignment = re.search(r'^CYCLONEDDS_BUILD="(?P<value>.+)"$', installer, re.MULTILINE)
    prefix_assignment = re.search(r'^CYCLONEDDS_PREFIX="(?P<value>.+)"$', installer, re.MULTILINE)
    assert source_assignment is not None
    assert build_assignment is not None
    assert prefix_assignment is not None
    assert 'BUILD_ROOT="$INSTALL_ROOT/build"' in installer
    assert 'DEPENDENCY_ROOT="$INSTALL_ROOT/dependencies"' in installer
    assert source_assignment.group("value") not in build_assignment.group("value")
    assert source_assignment.group("value") not in prefix_assignment.group("value")
    assert '-B "$CYCLONEDDS_BUILD"' in installer
    assert 'cmake --build "$CYCLONEDDS_BUILD"' in installer
    assert 'UNITREE_SDK_ARCHIVE="$BUILD_ROOT/' in installer
    assert 'git -C "$UNITREE_SDK_SRC" archive --format=tar' in installer
    assert 'pip install "$UNITREE_SDK_ARCHIVE"' in installer
    assert 'pip install "$UNITREE_SDK_SRC"' not in installer
    assert '"${PROJECT_ROOT}[mpc]"' in installer
    assert "import numpy, scipy" in installer


def test_aarch64_installer_preserves_old_yaml_with_migration_candidate(
    project_root: Path,
) -> None:
    installer = (project_root / "deploy" / "install_aarch64.sh").read_text(encoding="utf-8")

    assert '$SUDO cmp -s -- "$source" "$destination"' in installer
    assert 'candidate="$destination.dist"' in installer
    assert "preserving locally edited/older configuration" in installer
    assert "merge each corresponding *.dist candidate" in installer
    assert "CONFIG_MIGRATION_REQUIRED=1" in installer
    assert "configs/*.urdf" in installer
    assert '"$CONFIG_ROOT/$name"' in installer


def test_source_distribution_includes_pinned_urdf_and_license(project_root: Path) -> None:
    manifest = (project_root / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include configs *.yaml *.urdf *.txt" in manifest
    assert (project_root / "configs" / "go2_description.unitree_ros.urdf").is_file()
    assert (project_root / "configs" / "UNITREE_ROS_LICENSE.txt").is_file()
