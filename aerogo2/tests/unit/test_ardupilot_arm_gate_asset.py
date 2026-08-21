from __future__ import annotations

from pathlib import Path


def test_ardupilot_arm_gate_is_fail_closed_and_has_no_force_arm(
    project_root: Path,
) -> None:
    script = (project_root / "ardupilot" / "scripts" / "aerogo2_arm_gate.lua").read_text(
        encoding="utf-8"
    )

    assert "local AEROGO2_AUTH_COMMAND = 31000" in script
    assert "local MAV_CMD_COMPONENT_ARM_DISARM = 400" in script
    assert "mavlink:block_command(AEROGO2_AUTH_COMMAND)" in script
    assert "mavlink:block_command(MAV_CMD_COMPONENT_ARM_DISARM)" in script
    assert "arming:set_aux_auth_failed" in script
    assert "arming:set_aux_auth_passed" in script
    assert "local accepted = arming:arm()" in script
    assert "local accepted = arming:disarm()" in script
    assert "arm_force" not in script
    assert "AUTH_HEARTBEAT_TIMEOUT_MS = 1500" in script
    assert "ARM_SWITCH_CHANNEL = 5" in script
    assert '{"RC5_OPTION", 153}' in script
    assert '{"ARMING_RUDDER", 0}' in script
    assert '{"ARMING_CHECK", 1}' in script
    assert 'param:get("ARMING_SKIPCHK")' in script


def test_arm_gate_install_asset_is_registered_in_wheel(project_root: Path) -> None:
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.setuptools.data-files]" in pyproject
    assert ('"share/aerogo2/ardupilot" = ["ardupilot/scripts/aerogo2_arm_gate.lua"]') in pyproject
