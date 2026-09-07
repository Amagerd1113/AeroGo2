from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

import aerogo2.bridges.go2_control_arbiter as go2_arbiter_module
from aerogo2.bridges.go2_control_arbiter import (
    ControlOwnershipError,
    Go2ControlArbiter,
    InterProcessOwnerLock,
)
from aerogo2.bridges.go2_lowlevel_interface import Go2OwnershipPermit
from aerogo2.bridges.go2_lowlevel_sdk_bridge import (
    Go2SdkBindings,
    UnitreeGo2LowLevelSdkBridge,
    compute_go2_mapping_hash,
)
from aerogo2.common.clock import ManualClock
from aerogo2.common.config import Go2Config, Go2LowLevelConfig
from aerogo2.common.models import LowCmdOwnershipState
from aerogo2.common.results import OperationResult
from aerogo2.landing.impact_aware.integration import Go2JointPositionCommand


@pytest.fixture(autouse=True)
def reset_channel_factory() -> None:
    Go2ControlArbiter._reset_channel_factory_for_tests()


def _low_level_config(**changes: Any) -> Go2LowLevelConfig:
    names = tuple(f"joint_{index}" for index in range(12))
    motor_ids = tuple(range(12))
    directions = tuple(1 if index % 2 == 0 else -1 for index in range(12))
    offsets = tuple(0.1 * (index + 1) for index in range(12))
    mapping_version = "bench-v1"
    values = {
        "enabled": True,
        "low_state_topic": "rt/lowstate",
        "low_command_topic": "rt/lowcmd",
        "send_period_s": 0.02,
        "maximum_jitter_s": 0.019,
        "low_state_max_age_s": 1.0,
        "target_ttl_s": 0.2,
        "acquire_timeout_s": 0.5,
        "release_timeout_s": 0.5,
        "safe_hold_policy": "capture_current",
        "safe_hold_pose_rad": (0.0,) * 12,
        "safe_hold_position_tolerance_rad": (0.02,) * 12,
        "safe_hold_velocity_tolerance_rad_s": (0.05,) * 12,
        "tracking_position_error_limit_rad": (0.2,) * 12,
        "safe_hold_ack_timeout_s": 0.5,
        "restore_mode_form": "0",
        "restore_mode_name": "normal",
        "mapping_version": mapping_version,
        "mapping_hash": compute_go2_mapping_hash(
            mapping_version, names, motor_ids, directions, offsets
        ),
        "joint_names": names,
        "motor_ids": motor_ids,
        "directions": directions,
        "zero_offsets_rad": offsets,
        "q_min_rad": (-2.0,) * 12,
        "q_max_rad": (2.0,) * 12,
        "dq_max_rad_s": (1.0,) * 12,
        "maximum_delta_q_rad": (0.05,) * 12,
        "kp": (10.0,) * 12,
        "kd": (1.0,) * 12,
        "tau_ff_nm": tuple(0.2 if index % 2 == 0 else -0.2 for index in range(12)),
        "tau_limit_nm": (5.0,) * 12,
        "feedback_loss_degraded_kp": (1.0,) * 12,
        "feedback_loss_degraded_kd": (0.1,) * 12,
        "feedback_loss_degraded_tau_ff_nm": (0.0,) * 12,
        "firmware_torque_limit_nm": (4.0,) * 12,
        "firmware_torque_clamp_verified": True,
        "temperature_limit_c": (70.0,) * 12,
    }
    values.update(changes)
    return Go2LowLevelConfig(**values)


def _go2_config(low_level: Go2LowLevelConfig) -> Go2Config:
    return Go2Config(enabled=True, status_timeout_s=0.1, low_level=low_level)


@dataclass
class _MotorState:
    q: float
    dq: float = 0.0
    tau_est: float = 0.0
    temperature: float = 25.0
    lost: int = 0


class _LowState:
    def __init__(self, config: Go2LowLevelConfig) -> None:
        offsets = config.zero_offsets_rad
        assert offsets is not None
        self.motor_state = [_MotorState(0.0) for _ in range(20)]
        for index in range(12):
            self.motor_state[index] = _MotorState(offsets[index])


class _BlockingLowState:
    """Capture callback time before a write, then finish parsing afterwards."""

    def __init__(
        self,
        config: Go2LowLevelConfig,
        entered: threading.Event,
        proceed: threading.Event,
    ) -> None:
        self._motor_state = _LowState(config).motor_state
        self._entered = entered
        self._proceed = proceed

    @property
    def motor_state(self) -> List[_MotorState]:
        self._entered.set()
        if not self._proceed.wait(1.0):
            raise RuntimeError("test did not release the blocked LowState callback")
        return self._motor_state


@dataclass
class _MotorCommand:
    mode: int = 0
    q: float = 0.0
    dq: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    tau: float = 0.0


class _LowCommand:
    def __init__(self) -> None:
        self.head = [0, 0]
        self.level_flag = 0
        self.gpio = 0
        self.motor_cmd = [_MotorCommand() for _ in range(20)]
        self.crc = 0


class _Subscriber:
    def __init__(self, state: _LowState) -> None:
        self.state = state
        self.closed = False
        self.callback: Any = None

    def Init(self, callback: Any, queue_depth: int) -> None:
        assert queue_depth == 10
        self.callback = callback
        callback(self.state)

    def emit(self) -> None:
        if self.callback is not None:
            self.callback(self.state)

    def Close(self) -> None:
        self.closed = True


class _Publisher:
    def __init__(self, *, after_write: Any = None) -> None:
        self.writes: List[_LowCommand] = []
        self.timeouts: List[Any] = []
        self.write_event = threading.Event()
        self.closed = False
        self.close_error = False
        self.init_error = False
        self.init_delay_s = 0.0
        self.after_write = after_write

    def Init(self) -> None:
        if self.init_delay_s > 0.0:
            time.sleep(self.init_delay_s)
        if self.init_error:
            raise RuntimeError("injected publisher init failure after partial creation")
        return None

    def Write(self, message: _LowCommand, timeout: Any = None) -> bool:
        self.writes.append(message)
        self.timeouts.append(timeout)
        self.write_event.set()
        if self.after_write is not None:
            timer = threading.Timer(0.001, self.after_write)
            timer.daemon = True
            timer.start()
        return True

    def Close(self) -> None:
        if self.close_error:
            raise RuntimeError("injected publisher close failure")
        self.closed = True


class _MotionSwitcher:
    def __init__(self) -> None:
        self.mode_form = "0"
        self.mode_name = "normal"
        self.release_calls = 0
        self.timeout = 0.0
        self.check_delay_s = 0.0
        self.fail_check_after_release = False
        self.release_code = 0
        self.select_code = 0
        self.select_activates = True
        self.selected_name_override: Any = None
        self.select_calls: List[str] = []

    def SetTimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def Init(self) -> None:
        return None

    def CheckMode(self) -> Tuple[int, Any]:
        if self.check_delay_s > 0.0:
            time.sleep(self.check_delay_s)
        if self.release_calls and self.fail_check_after_release:
            return 1, None
        return 0, {"form": self.mode_form, "name": self.mode_name}

    def ReleaseMode(self) -> Tuple[int, None]:
        self.release_calls += 1
        if self.release_code == 0:
            self.mode_name = ""
        return self.release_code, None

    def SelectMode(self, name: str) -> Tuple[int, None]:
        self.select_calls.append(name)
        if self.select_code == 0 and self.select_activates:
            self.mode_name = (
                name if self.selected_name_override is None else self.selected_name_override
            )
        return self.select_code, None


class _CRC:
    def Crc(self, message: _LowCommand) -> int:
        assert message.head == [0xFE, 0xEF]
        return 0xA5A5A5A5


@dataclass
class _FakeSdk:
    bindings: Go2SdkBindings
    publisher: _Publisher
    subscriber: _Subscriber
    motion: _MotionSwitcher
    channel_init_calls: List[Tuple[int, str]]


def _fake_sdk(config: Go2LowLevelConfig, *, auto_feedback: bool = True) -> _FakeSdk:
    state = _LowState(config)
    subscriber = _Subscriber(state)
    publisher = _Publisher(after_write=subscriber.emit if auto_feedback else None)
    motion = _MotionSwitcher()
    init_calls: List[Tuple[int, str]] = []

    def initialize(domain_id: int, interface: str) -> None:
        init_calls.append((domain_id, interface))

    return _FakeSdk(
        bindings=Go2SdkBindings(
            channel_factory_initialize=initialize,
            subscriber_factory=lambda topic, message_type: subscriber,
            publisher_factory=lambda topic, message_type: publisher,
            low_state_type=_LowState,
            low_cmd_factory=_LowCommand,
            motion_switcher_factory=lambda: motion,
            crc_factory=_CRC,
            publisher_constructor_deferred_until_init=True,
            publisher_close_retry_idempotency_verified=True,
        ),
        publisher=publisher,
        subscriber=subscriber,
        motion=motion,
        channel_init_calls=init_calls,
    )


def _force_cleanup_fake_owner(
    bridge: UnitreeGo2LowLevelSdkBridge,
    ownership_epoch: int,
) -> None:
    """Prevent a failed fake-transport assertion from stranding a test process."""

    if not bridge.status().ownership_pending:
        return
    bridge._writer_stop.set()
    thread = bridge._writer_thread
    if thread is not None and thread is not threading.current_thread():
        thread.join(1.0)
    with bridge._guard:
        subscriber = bridge._subscriber
        publisher = bridge._publisher
        bridge._subscriber = None
        bridge._publisher = None
        bridge._owner_epoch = 0
    bridge._close_transport(subscriber=subscriber, publisher=publisher)
    if bridge._arbiter.status().low_level_epoch == ownership_epoch:
        bridge._arbiter.release_low_level(ownership_epoch)


def _permit(config: Go2LowLevelConfig, *, ttl: float = 0.2) -> Go2OwnershipPermit:
    now = time.monotonic()
    assert config.mapping_version is not None
    assert config.mapping_hash is not None
    return Go2OwnershipPermit(
        timestamp_s=now - 0.01,
        valid_until_s=now + ttl,
        operator_authorized=True,
        robot_supported=True,
        pixhawk_disarmed=True,
        rotors_stopped=True,
        mapping_version=config.mapping_version,
        mapping_hash=config.mapping_hash,
        reason="bench fixture supports the robot",
    )


def _ground_transfer_ok(transfer: str, permit: Go2OwnershipPermit) -> OperationResult:
    assert transfer in {"acquire", "release"}
    assert isinstance(permit, Go2OwnershipPermit)
    return OperationResult.success("fresh test ground evidence")


def _target(config: Go2LowLevelConfig, sequence: int, value: float) -> Go2JointPositionCommand:
    now = time.monotonic()
    ttl = config.target_ttl_s
    assert ttl is not None
    return Go2JointPositionCommand(
        sequence=sequence,
        timestamp_s=now - 0.001,
        valid_until_s=now + ttl * 0.5,
        joint_positions_rad=(value,) * 12,
        desired_contact_forces_world_n=(0.0,) * 12,
    )


def test_ground_permit_is_short_lived_and_mapping_bound() -> None:
    config = _low_level_config()
    permit = _permit(config)
    assert config.mapping_version is not None
    assert config.mapping_hash is not None
    now = time.monotonic()
    assert permit.authorizes(
        now,
        mapping_version=config.mapping_version,
        mapping_hash=config.mapping_hash,
    )
    assert not permit.authorizes(
        now,
        mapping_version=config.mapping_version,
        mapping_hash="sha256:" + "0" * 64,
    )
    assert not permit.authorizes(
        permit.valid_until_s,
        mapping_version=config.mapping_version,
        mapping_hash=config.mapping_hash,
    )


def test_arbiter_excludes_sport_and_a_second_local_owner(tmp_path: Path) -> None:
    path = tmp_path / "owner.lock"
    first = Go2ControlArbiter(
        lock_path=path,
        epoch_factory=lambda: 41,
        network_exclusivity_verifier=lambda: True,
        network_exclusivity_verifier_name="test-dds-graph-audit",
    )
    second = Go2ControlArbiter(
        lock_path=path,
        epoch_factory=lambda: 42,
        network_exclusivity_verifier=lambda: True,
        network_exclusivity_verifier_name="test-dds-graph-audit",
    )
    with first.sport_lease():
        assert first.acquire_low_level().code == "GO2_SPORT_OPERATION_ACTIVE"
    grant = first.acquire_low_level()
    assert grant.ok and grant.data["ownership_epoch"] == 41
    assert first.status().network_exclusivity_verified
    assert grant.data["continuous_network_monitoring_active"] is False
    assert not first.status().continuous_network_monitoring_active
    assert first.status().network_verifier_name == "test-dds-graph-audit"
    with pytest.raises(ControlOwnershipError):
        with first.sport_lease():
            pass
    assert second.acquire_low_level().code == "GO2_LOW_LEVEL_OWNER_LOCKED"
    assert first.release_low_level(999).code == "GO2_OWNERSHIP_EPOCH_MISMATCH"
    assert first.release_low_level(41).ok
    assert not first.status().network_exclusivity_verified
    assert second.acquire_low_level().ok
    assert second.release_low_level(42).ok


def test_arbiter_default_refuses_to_treat_local_lock_as_network_proof(
    tmp_path: Path,
) -> None:
    arbiter = Go2ControlArbiter(lock_path=tmp_path / "unverified.lock")
    result = arbiter.acquire_low_level()
    assert result.code == "GO2_NETWORK_EXCLUSIVITY_UNVERIFIED"
    status = arbiter.status()
    assert not status.network_exclusivity_verified
    assert status.network_verifier_name == "not_configured"
    assert status.network_verification_timestamp_s > 0.0
    assert not status.local_single_instance_held


def test_arbiter_holds_host_lock_during_network_verification(tmp_path: Path) -> None:
    observations: List[bool] = []
    arbiter: Go2ControlArbiter

    def verify_network() -> bool:
        observations.append(arbiter.status().local_single_instance_held)
        return True

    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "lock-before-network-proof.lock",
        epoch_factory=lambda: 73,
        network_exclusivity_verifier=verify_network,
    )

    grant = arbiter.acquire_low_level()

    assert grant.ok
    assert observations == [True]
    assert arbiter.release_low_level(73).ok


def test_parallel_acquire_cannot_invalidate_inflight_network_proof(tmp_path: Path) -> None:
    verifier_started = threading.Event()
    finish_verifier = threading.Event()
    first_result: List[OperationResult] = []

    def verify_network() -> bool:
        verifier_started.set()
        return finish_verifier.wait(1.0)

    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "parallel-acquire-proof.lock",
        epoch_factory=lambda: 74,
        network_exclusivity_verifier=verify_network,
    )
    first = threading.Thread(target=lambda: first_result.append(arbiter.acquire_low_level(1.0)))
    first.start()
    assert verifier_started.wait(0.5)

    racing = arbiter.acquire_low_level(0.1)
    assert racing.code == "GO2_LOW_LEVEL_ACQUIRE_BUSY"
    finish_verifier.set()
    first.join(1.0)

    assert not first.is_alive()
    assert len(first_result) == 1 and first_result[0].ok
    assert arbiter.status().network_exclusivity_verified
    assert arbiter.release_low_level(74).ok


@pytest.mark.parametrize("bad_epoch", [True, 1.0])
def test_arbiter_rejects_non_exact_integer_epoch_values(
    tmp_path: Path,
    bad_epoch: object,
) -> None:
    invalid = Go2ControlArbiter(
        lock_path=tmp_path / f"invalid-factory-{type(bad_epoch).__name__}.lock",
        epoch_factory=lambda: bad_epoch,  # type: ignore[arg-type,return-value]
        network_exclusivity_verifier=lambda: True,
    )
    assert invalid.acquire_low_level().code == "GO2_INVALID_OWNERSHIP_EPOCH"
    assert not invalid.status().local_single_instance_held

    valid = Go2ControlArbiter(
        lock_path=tmp_path / f"release-type-{type(bad_epoch).__name__}.lock",
        epoch_factory=lambda: 1,
        network_exclusivity_verifier=lambda: True,
    )
    assert valid.acquire_low_level().ok
    assert valid.release_low_level(bad_epoch).code == "GO2_OWNERSHIP_EPOCH_MISMATCH"  # type: ignore[arg-type]
    assert valid.status().local_single_instance_held
    assert valid.status().low_level_epoch == 1
    assert valid.release_low_level(1).ok


def test_arbiter_poisoned_unlock_never_claims_lock_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "poisoned-unlock.lock",
        epoch_factory=lambda: 171,
        network_exclusivity_verifier=lambda: True,
    )
    assert arbiter.acquire_low_level().ok
    original_unlock = arbiter._owner_lock._unlock_fd

    def fail_unlock(fd: int) -> None:
        del fd
        raise OSError("injected unlock ambiguity")

    monkeypatch.setattr(arbiter._owner_lock, "_unlock_fd", fail_unlock)
    failed = arbiter.release_low_level(171)

    assert failed.code == "GO2_OWNER_LOCK_RELEASE_FAILED"
    assert failed.data["owner_lock_retained"] is False
    assert failed.data["ownership_exclusivity_lost"] is True
    status = arbiter.status()
    assert status.low_level_epoch == 171
    assert status.local_single_instance_poisoned
    assert not status.local_single_instance_held

    # Restore the primitive only to close the test descriptor. The poisoned
    # arbiter intentionally remains unusable for all future LowCmd *and*
    # Sport operations, even after the retained epoch is cleaned up.
    monkeypatch.setattr(arbiter._owner_lock, "_unlock_fd", original_unlock)
    assert arbiter.release_low_level(171).ok
    assert arbiter.acquire_low_level().code == "GO2_CONTROL_ARBITER_POISONED"
    assert arbiter.assert_sport_allowed().code == "GO2_CONTROL_ARBITER_POISONED"
    with pytest.raises(ControlOwnershipError, match="ambiguous owner-lock"):
        with arbiter.sport_lease():
            pass


def test_owner_lock_poisoned_if_post_lock_setup_and_close_both_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = InterProcessOwnerLock(tmp_path / "post-lock-close-failure.lock")
    real_os = os

    class _FailingOsProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(real_os, name)

        @staticmethod
        def fsync(fd: int) -> None:
            del fd
            raise OSError("injected metadata failure after lock")

        @staticmethod
        def close(fd: int) -> None:
            del fd
            raise OSError("injected ambiguous close")

    monkeypatch.setattr(go2_arbiter_module, "os", _FailingOsProxy())
    with pytest.raises(OSError, match="ambiguous close"):
        lock.acquire()

    assert lock.poisoned
    assert not lock.held
    assert lock._fd is not None
    # Test-only descriptor cleanup with the real primitive. Poison remains
    # latched, exactly as production would require until process restart.
    real_os.close(lock._fd)
    lock._fd = None
    assert not lock.acquire()


def test_host_lock_excludes_sport_and_lowcmd_across_arbiter_instances(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared-sport-lowcmd.lock"
    sport_process = Go2ControlArbiter(
        lock_path=path,
        epoch_factory=lambda: 181,
        network_exclusivity_verifier=lambda: True,
    )
    lowcmd_process = Go2ControlArbiter(
        lock_path=path,
        epoch_factory=lambda: 182,
        network_exclusivity_verifier=lambda: True,
    )

    with sport_process.sport_lease():
        assert lowcmd_process.acquire_low_level().code == "GO2_LOW_LEVEL_OWNER_LOCKED"

    assert lowcmd_process.acquire_low_level().ok
    assert sport_process.assert_sport_allowed().code == "GO2_HOST_CONTROL_LOCKED"
    with pytest.raises(ControlOwnershipError, match="host-wide Go2 control lock"):
        with sport_process.sport_lease():
            pass
    assert lowcmd_process.release_low_level(182).ok

    with sport_process.sport_lease():
        pass


def test_host_lock_excludes_sport_and_lowcmd_across_os_processes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cross-process-control.lock"
    parent = Go2ControlArbiter(
        lock_path=path,
        epoch_factory=lambda: 183,
        network_exclusivity_verifier=lambda: True,
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path.cwd() / "src")
    acquire_script = (
        "import sys; from pathlib import Path; "
        "from aerogo2.bridges.go2_control_arbiter import Go2ControlArbiter; "
        "a=Go2ControlArbiter(lock_path=Path(sys.argv[1]), "
        "network_exclusivity_verifier=lambda: True); "
        "print(a.acquire_low_level().code)"
    )
    sport_probe_script = (
        "import sys; from pathlib import Path; "
        "from aerogo2.bridges.go2_control_arbiter import Go2ControlArbiter; "
        "a=Go2ControlArbiter(lock_path=Path(sys.argv[1])); "
        "print(a.assert_sport_allowed().code)"
    )

    with parent.sport_lease():
        child = subprocess.run(
            [sys.executable, "-c", acquire_script, str(path)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=5.0,
        )
        assert child.stdout.strip() == "GO2_LOW_LEVEL_OWNER_LOCKED"

    assert parent.acquire_low_level().ok
    child = subprocess.run(
        [sys.executable, "-c", sport_probe_script, str(path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=5.0,
    )
    assert child.stdout.strip() == "GO2_HOST_CONTROL_LOCKED"
    assert parent.release_low_level(183).ok


def test_arbiter_network_verifier_has_hard_timeout(tmp_path: Path) -> None:
    never = threading.Event()
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "slow-audit.lock",
        network_exclusivity_verifier=lambda: never.wait(10.0),
    )
    started = time.monotonic()
    result = arbiter.acquire_low_level(0.02)
    assert result.code == "GO2_NETWORK_EXCLUSIVITY_TIMEOUT"
    assert time.monotonic() - started < 0.25
    assert not arbiter.status().local_single_instance_held


@pytest.mark.parametrize("factory_raises", [False, True])
def test_failed_epoch_grant_reports_poisoned_lock_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_raises: bool,
) -> None:
    def epoch_factory() -> object:
        if factory_raises:
            raise RuntimeError("injected epoch failure")
        return True

    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / f"failed-grant-{factory_raises}.lock",
        epoch_factory=epoch_factory,  # type: ignore[arg-type]
        network_exclusivity_verifier=lambda: True,
    )
    original_unlock = arbiter._owner_lock._unlock_fd

    def fail_unlock(fd: int) -> None:
        del fd
        raise OSError("injected failed-grant unlock ambiguity")

    monkeypatch.setattr(arbiter._owner_lock, "_unlock_fd", fail_unlock)
    result = arbiter.acquire_low_level()
    assert result.code == "GO2_CONTROL_ARBITER_POISONED"
    assert result.data["process_restart_required"] is True
    assert result.data["local_single_instance_poisoned"] is True
    assert arbiter.status().low_level_epoch == 0
    assert arbiter.assert_sport_allowed().code == "GO2_CONTROL_ARBITER_POISONED"

    # Test-only descriptor cleanup; poison deliberately remains latched.
    monkeypatch.setattr(arbiter._owner_lock, "_unlock_fd", original_unlock)
    arbiter._owner_lock.release()


def test_channel_factory_is_initialized_once_and_rejects_conflict() -> None:
    calls: List[Tuple[int, str]] = []

    def initialize(domain: int, interface: str) -> None:
        calls.append((domain, interface))

    assert Go2ControlArbiter.initialize_channel_factory(initialize, 0, "eth0").ok
    assert Go2ControlArbiter.initialize_channel_factory(initialize, 0, "eth0").ok
    assert calls == [(0, "eth0")]
    conflict = Go2ControlArbiter.initialize_channel_factory(initialize, 1, "eth1")
    assert conflict.code == "GO2_CHANNEL_FACTORY_CONFLICT"


@pytest.mark.asyncio
async def test_disabled_bridge_status_is_fail_closed_without_sdk(tmp_path: Path) -> None:
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(Go2LowLevelConfig()),
        arbiter=Go2ControlArbiter(lock_path=tmp_path / "disabled.lock"),
    )
    status = bridge.status()
    assert status.ownership_state is LowCmdOwnershipState.DISABLED
    assert not status.healthy
    assert (await bridge.connect()).code == "GO2_LOW_STATE_OBSERVATION_DISABLED"
    now = time.monotonic()
    disabled_permit = Go2OwnershipPermit(
        timestamp_s=now,
        valid_until_s=now + 0.1,
        operator_authorized=True,
        robot_supported=True,
        pixhawk_disarmed=True,
        rotors_stopped=True,
        mapping_version="not-configured",
        mapping_hash="not-configured",
        reason="shutdown cleanup",
    )
    assert (
        await bridge.release(
            disabled_permit,
            "shutdown cleanup",
            ownership_epoch=0,
        )
    ).ok


@pytest.mark.asyncio
async def test_connect_is_observe_only_and_hardware_gate_blocks_acquire(tmp_path: Path) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(lock_path=tmp_path / "observe.lock"),
        allow_hardware_write=False,
        sdk_bindings=fake.bindings,
    )
    connected = await bridge.connect()
    assert connected.ok
    assert connected.data["low_state_fresh"] is True
    assert connected.data["publisher_active"] is False
    assert connected.data["writer_present"] is False
    status = bridge.status()
    assert status.ownership_state is LowCmdOwnershipState.OBSERVE_ONLY
    assert status.healthy
    assert not status.publisher_active
    assert not status.writer_alive
    assert len(status.motors) == 12
    assert all(motor.q_rad == pytest.approx(0.0) for motor in status.motors)
    assert fake.publisher.writes == []
    rejected = await bridge.acquire(_permit(config))
    assert rejected.code == "GO2_LOW_LEVEL_WRITE_LOCKED"
    assert fake.publisher.writes == []
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_connect_rejects_lowstate_that_is_stale_before_commit(tmp_path: Path) -> None:
    config = _low_level_config(low_state_max_age_s=0.1)
    fake = _fake_sdk(config)
    clock = ManualClock(10.0)
    state = _LowState(config)

    class _StalingSubscriber(_Subscriber):
        def Init(self, callback: Any, queue_depth: int) -> None:
            assert queue_depth == 10
            self.callback = callback
            callback(self.state)
            clock.advance(0.101)

    subscriber = _StalingSubscriber(state)
    bindings = replace(
        fake.bindings,
        subscriber_factory=lambda topic, message_type: subscriber,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(lock_path=tmp_path / "stale-connect.lock"),
        clock=clock,
        sdk_bindings=bindings,
    )

    connected = await bridge.connect()

    assert connected.code == "GO2_LOW_STATE_STALE_AT_CONNECT"
    assert connected.data["low_state_age_s"] == pytest.approx(0.101)
    assert subscriber.closed
    status = bridge.status()
    assert status.ownership_state is LowCmdOwnershipState.DISCONNECTED
    assert not status.connected
    assert not status.healthy
    assert not status.publisher_active
    assert not status.writer_alive
    assert fake.publisher.writes == []


@pytest.mark.asyncio
async def test_connect_rejects_latest_lowstate_feedback_fault(tmp_path: Path) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    state = _LowState(config)

    class _FaultingSubscriber(_Subscriber):
        def Init(self, callback: Any, queue_depth: int) -> None:
            assert queue_depth == 10
            self.callback = callback
            callback(self.state)
            self.state.motor_state[0].lost = 1
            callback(self.state)

    subscriber = _FaultingSubscriber(state)
    bindings = replace(
        fake.bindings,
        subscriber_factory=lambda topic, message_type: subscriber,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(lock_path=tmp_path / "feedback-fault-connect.lock"),
        sdk_bindings=bindings,
    )

    connected = await bridge.connect()

    assert connected.code == "GO2_LOW_STATE_FEEDBACK_FAULT"
    assert "lost" in str(connected.data["feedback_fault"]).lower()
    assert subscriber.closed
    status = bridge.status()
    assert status.ownership_state is LowCmdOwnershipState.DISCONNECTED
    assert not status.connected
    assert not status.healthy
    assert not status.publisher_active
    assert not status.writer_alive
    assert fake.publisher.writes == []


@pytest.mark.asyncio
async def test_observe_only_status_rejects_any_writer_object(tmp_path: Path) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(lock_path=tmp_path / "observe-invariant.lock"),
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok

    with bridge._guard:
        bridge._writer_thread = threading.Thread()
    status = bridge.status()
    assert status.ownership_state is LowCmdOwnershipState.OBSERVE_ONLY
    assert not status.healthy
    assert not status.writer_alive
    assert status.fault_reason is not None
    assert "unexpectedly retains" in status.fault_reason

    with bridge._guard:
        bridge._writer_thread = None
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_observe_only_connect_needs_no_gains_limits_or_control_helpers(
    tmp_path: Path,
) -> None:
    full = _low_level_config()
    observe_only = replace(
        full,
        enabled=False,
        observe_only_enabled=True,
        q_min_rad=None,
        q_max_rad=None,
        dq_max_rad_s=None,
        kp=None,
        kd=None,
        tau_limit_nm=None,
        temperature_limit_c=None,
        firmware_torque_clamp_verified=None,
    )
    fake = _fake_sdk(observe_only)

    def forbidden_control_helper() -> Any:
        raise AssertionError("observe-only connect initialized a control helper")

    bindings = replace(
        fake.bindings,
        publisher_factory=lambda topic, message_type: forbidden_control_helper(),
        motion_switcher_factory=forbidden_control_helper,
        crc_factory=forbidden_control_helper,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(observe_only),
        arbiter=Go2ControlArbiter(lock_path=tmp_path / "read-side-only.lock"),
        allow_hardware_write=True,
        sdk_bindings=bindings,
    )

    connected = await bridge.connect()
    assert connected.ok
    assert connected.data["lowcmd_actuation_ready"] is False
    status = bridge.status()
    assert status.ownership_state is LowCmdOwnershipState.OBSERVE_ONLY
    assert status.healthy
    assert status.mapping_hash_verified
    assert len(status.motors) == 12

    assert (await bridge.acquire(_permit(observe_only))).code == "GO2_LOW_LEVEL_DISABLED"
    submitted = await bridge.submit(
        _target(observe_only, 1, 0.0),
        ownership_epoch=0,
        mapping_hash=observe_only.mapping_hash or "",
    )
    assert submitted.code == "GO2_LOW_LEVEL_NOT_ACTIVE"
    assert fake.publisher.writes == []
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_acquire_requires_deferred_publisher_construction_contract(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    unsafe_bindings = replace(
        fake.bindings,
        publisher_constructor_deferred_until_init=False,
    )
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "publisher-constructor-contract.lock",
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=unsafe_bindings,
    )
    assert (await bridge.connect()).ok

    rejected = await bridge.acquire(_permit(config))

    assert rejected.code == "GO2_PUBLISHER_CONSTRUCTION_CONTRACT_UNVERIFIED"
    assert fake.motion.release_calls == 0
    assert fake.publisher.writes == []
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_malformed_lowstate_does_not_refresh_last_valid_feedback(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "malformed.lock",
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    valid = _LowState(config)
    valid.foot_force = [1, 2, 3, 4]
    valid.foot_force_est = [5, 6, 7, 8]
    valid.tick = 7
    bridge._on_low_state(valid)
    assert bridge.status().foot_force_feedback.source_identity_valid
    valid_timestamp = bridge.status().low_state_timestamp
    bridge._on_low_state(object())
    status = bridge.status()
    assert status.low_state_timestamp == valid_timestamp
    assert status.fault_reason == "LowState.motor_state is missing"
    assert status.motors and all(motor.lost for motor in status.motors)
    assert not status.foot_force_feedback.raw_valid
    assert not status.foot_force_feedback.source_identity_valid
    assert (await bridge.acquire(_permit(config))).code == "GO2_LOW_STATE_UNSAFE"
    assert fake.motion.release_calls == 0
    assert fake.publisher.writes == []
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_lowstate_preserves_raw_and_estimated_foot_force_identity(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(lock_path=tmp_path / "foot-force.lock"),
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    state = _LowState(config)
    state.foot_force = [11, 22, 33, 44]
    state.foot_force_est = [101, 202, 303, 404]
    state.tick = 0xFFFFFFFE

    bridge._on_low_state(state)

    feedback = bridge.status().foot_force_feedback
    assert feedback.raw_sdk_int16 == (11, 22, 33, 44)
    assert feedback.estimated_sdk_int16 == (101, 202, 303, 404)
    assert feedback.raw_valid and feedback.estimated_valid
    assert feedback.source_tick == 0xFFFFFFFE
    assert feedback.source_identity_valid
    assert feedback.receipt_sequence > 0
    assert feedback.subscription_generation > 0
    assert feedback.receipt_timestamp_s > 0.0

    # uint32 wrap is a forward serial-number transition.
    state.tick = 1
    bridge._on_low_state(state)
    assert bridge.status().foot_force_feedback.source_identity_valid
    assert bridge.status().foot_force_feedback.source_tick == 1

    # A duplicate tick never becomes a fresh high-rate source identity, even
    # though its integers remain available for diagnostics.
    bridge._on_low_state(state)
    duplicate = bridge.status().foot_force_feedback
    assert duplicate.raw_valid and duplicate.estimated_valid
    assert not duplicate.source_tick_monotonic
    assert not duplicate.source_identity_valid

    state.tick = 2
    bridge._on_low_state(state)
    assert bridge.status().foot_force_feedback.source_identity_valid
    assert (await bridge.disconnect()).ok
    assert not bridge.status().foot_force_feedback.source_identity_valid


@pytest.mark.asyncio
async def test_malformed_foot_force_is_independently_invalid_and_not_newtons(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(lock_path=tmp_path / "bad-foot-force.lock"),
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    state = _LowState(config)
    state.foot_force = [1, 2, True, 4]
    state.foot_force_est = [1, 2, 3, 40000]
    state.tick = -1

    bridge._on_low_state(state)

    status = bridge.status()
    feedback = status.foot_force_feedback
    assert status.healthy  # joint feedback remains independently usable
    assert not feedback.raw_valid
    assert not feedback.estimated_valid
    assert feedback.raw_sdk_int16 == (0, 0, 0, 0)
    assert feedback.estimated_sdk_int16 == (0, 0, 0, 0)
    assert feedback.source_tick is None
    assert not feedback.source_identity_valid
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_bridge_requires_runtime_network_proof_before_release_mode(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(lock_path=tmp_path / "no-network-proof.lock")
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    rejected = await bridge.acquire(_permit(config))
    assert rejected.code == "GO2_NETWORK_EXCLUSIVITY_UNVERIFIED"
    assert fake.motion.release_calls == 0
    assert fake.publisher.writes == []
    assert not bridge.status().network_exclusivity_verified
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_permit_expiring_during_check_mode_never_calls_release_mode(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    fake.motion.check_delay_s = 0.04
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "permit-check-mode-expiry.lock",
        epoch_factory=lambda: 76,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok

    rejected = await bridge.acquire(_permit(config, ttl=0.025))

    assert rejected.code == "GO2_OWNERSHIP_PERMIT_EXPIRED"
    assert rejected.data["release_rpc_attempted"] is False
    assert fake.motion.release_calls == 0
    assert fake.publisher.closed
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
@pytest.mark.parametrize("violation", ["q", "dq", "tau", "temperature", "lost"])
async def test_acquire_rejects_each_available_motor_feedback_limit(
    tmp_path: Path,
    violation: str,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(lock_path=tmp_path / f"{violation}.lock"),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    bad = _LowState(config)
    motor = bad.motor_state[0]
    if violation == "q":
        assert config.zero_offsets_rad is not None
        motor.q = config.zero_offsets_rad[0] + 2.1
    elif violation == "dq":
        motor.dq = 1.1
    elif violation == "tau":
        motor.tau_est = 5.1
    elif violation == "temperature":
        motor.temperature = 70.0
    else:
        motor.lost = 1
    bridge._on_low_state(bad)
    result = await bridge.acquire(_permit(config))
    assert result.code == "GO2_LOW_STATE_UNSAFE"
    assert fake.publisher.writes == []
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_owner_builds_crc_mapped_lowcmd_and_enforces_epoch_ttl_limits(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    config = _low_level_config(
        # This case tests mapping, CRC, mailbox and safe-hold semantics. Leave
        # scheduler-overrun behavior to its dedicated deterministic tests.
        send_period_s=0.1,
        maximum_jitter_s=0.099,
        target_ttl_s=1.0,
        acquire_timeout_s=1.0,
        release_timeout_s=2.0,
        safe_hold_ack_timeout_s=1.5,
        maximum_delta_q_rad=(0.2,) * 12,
    )
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "active.lock",
        epoch_factory=lambda: 77,
        network_exclusivity_verifier=lambda: True,
        network_exclusivity_verifier_name="test-dds-graph-audit",
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    request.addfinalizer(lambda: _force_cleanup_fake_owner(bridge, 77))
    assert (await bridge.connect()).ok
    acquired = await bridge.acquire(_permit(config, ttl=0.5))
    assert acquired.ok
    assert acquired.data["network_exclusivity_verified"] is True
    assert bridge._writer_thread is not None
    assert not bridge._writer_thread.daemon
    assert acquired.data["actuator_ack_available"] is False
    assert fake.motion.release_calls == 1
    assert fake.publisher.writes
    assert fake.publisher.timeouts[0] == pytest.approx(config.acquire_timeout_s)
    first = fake.publisher.writes[0]
    assert first.head == [0xFE, 0xEF]
    assert first.level_flag == 0xFF
    assert first.crc == 0xA5A5A5A5
    assert first.motor_cmd[12].q == pytest.approx(2.146e9)
    assert first.motor_cmd[12].dq == pytest.approx(16000.0)
    assert first.motor_cmd[0].q == pytest.approx(0.1)
    assert first.motor_cmd[1].q == pytest.approx(0.2)
    assert first.motor_cmd[0].tau == pytest.approx(0.2)
    assert first.motor_cmd[1].tau == pytest.approx(0.2)
    acquired_status = bridge.status()
    assert acquired_status.safe_hold_active
    assert (
        acquired_status.safe_hold_write_generation == acquired_status.safe_hold_request_generation
    )
    assert bridge.status().safe_hold_settled

    assert config.mapping_hash is not None
    stale_epoch = await bridge.submit(
        _target(config, 1, 0.5), ownership_epoch=76, mapping_hash=config.mapping_hash
    )
    assert stale_epoch.code == "GO2_OWNERSHIP_EPOCH_MISMATCH"
    wrong_hash = await bridge.submit(
        _target(config, 1, 0.5), ownership_epoch=77, mapping_hash="sha256:" + "0" * 64
    )
    assert wrong_hash.code == "GO2_MAPPING_HASH_MISMATCH"
    outside_limit = await bridge.submit(
        _target(config, 1, 3.0), ownership_epoch=77, mapping_hash=config.mapping_hash
    )
    assert outside_limit.code == "GO2_JOINT_TARGET_LIMIT"
    accepted = await bridge.submit(
        _target(config, 1, 0.5), ownership_epoch=77, mapping_hash=config.mapping_hash
    )
    assert accepted.ok
    assert accepted.code == "GO2_LOW_CMD_TARGET_STAGED"
    assert accepted.data["mailbox_stage_acknowledged"] is True
    assert accepted.data["writer_enqueue_acknowledged"] is False
    assert accepted.data["actuator_application_acknowledged"] is False
    fake.publisher.write_event.clear()
    replay = await bridge.submit(
        _target(config, 1, 0.4), ownership_epoch=77, mapping_hash=config.mapping_hash
    )
    assert replay.code == "GO2_MPC_TARGET_REPLAY"
    wrote_target_cycle = await asyncio.get_running_loop().run_in_executor(
        None,
        fake.publisher.write_event.wait,
        0.75,
    )
    assert wrote_target_cycle
    active = bridge.status()
    assert active.ownership_state is LowCmdOwnershipState.MPC_ACTIVE
    assert active.target_sequence == 1
    assert active.mailbox_staged_target_sequence == 1
    assert active.writer_enqueued_target_sequence == 1
    assert active.actuator_applied_target_sequence is None
    assert not active.continuous_owner_monitoring_active
    assert not active.independent_watchdog_active
    assert not active.actuator_application_ack_available
    # dq_max * period = 0.1 rad, which is stricter than delta_q=0.2.
    for before, after in zip(fake.publisher.writes, fake.publisher.writes[1:]):
        assert abs(after.motor_cmd[0].q - before.motor_cmd[0].q) <= 0.1000001
        assert abs(after.motor_cmd[1].q - before.motor_cmd[1].q) <= 0.1000001
    assert fake.publisher.writes[-1].motor_cmd[0].q > first.motor_cmd[0].q

    # capture_current freezes the newest measured posture at revocation; it
    # must not pull the legs back toward the pose captured at acquisition.
    measured = _LowState(config)
    assert config.zero_offsets_rad is not None
    assert config.directions is not None
    for index in range(12):
        measured.motor_state[index].q = (
            config.zero_offsets_rad[index] + config.directions[index] * 0.02
        )
    fake.publisher.after_write = lambda: bridge._on_low_state(measured)
    bridge._on_low_state(measured)
    revoked = await bridge.revoke("controller shutdown", ownership_epoch=77)
    assert revoked.ok
    revoked_status = bridge.status()
    assert revoked_status.ownership_state is LowCmdOwnershipState.SAFE_HOLD
    assert revoked_status.safe_hold_active
    assert revoked_status.safe_hold_write_generation == revoked.data["safe_hold_generation"]
    assert revoked.data["actuator_ack_available"] is False
    assert revoked_status.safe_hold_settled
    rejected_reactivation = await bridge.submit(
        _target(config, 2, 0.1),
        ownership_epoch=77,
        mapping_hash=config.mapping_hash,
    )
    assert rejected_reactivation.code == "GO2_LOW_LEVEL_NOT_ACTIVE"
    assert bridge._safe_hold_q == pytest.approx((0.02,) * 12)
    bridge._on_low_state(measured)
    assert bridge.status().safe_hold_settled
    assert (await bridge.release(_permit(config), "bench teardown", ownership_epoch=77)).ok
    assert fake.motion.select_calls == ["normal"]
    assert bridge.status().ownership_state is LowCmdOwnershipState.OBSERVE_ONLY
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_writer_freezes_first_limited_q_until_a_new_mailbox_sequence(
    tmp_path: Path,
) -> None:
    """One sequence must not keep slewing after its first writer ACK."""

    config = _low_level_config(
        # Keep the background writer asleep while deterministic cycles below
        # exercise the same production bridge method under its owner guard.
        send_period_s=1.0,
        maximum_jitter_s=0.999,
        low_state_max_age_s=10.0,
        target_ttl_s=10.0,
        acquire_timeout_s=1.5,
        release_timeout_s=2.5,
        safe_hold_ack_timeout_s=1.5,
        maximum_delta_q_rad=(0.05,) * 12,
    )
    fake = _fake_sdk(config)
    ownership_epoch = 772
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "freeze-one-sequence.lock",
            epoch_factory=lambda: ownership_epoch,
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    try:
        assert (await bridge.connect()).ok
        assert (await bridge.acquire(_permit(config, ttl=1.0))).ok
        assert config.mapping_hash is not None
        first_target = _target(config, 1, 0.5)
        assert (
            await bridge.submit(
                first_target,
                ownership_epoch=ownership_epoch,
                mapping_hash=config.mapping_hash,
            )
        ).ok

        with bridge._guard:
            bridge._writer_cycle(time.monotonic() + 1.0, expected_deadline=None)
            first_ack = bridge.status()
            first_q = first_ack.writer_enqueued_q_rad
            first_generation = first_ack.writer_enqueue_generation
            assert first_ack.writer_enqueued_target_sequence == 1
            assert first_q == pytest.approx((0.05,) * 12)

            bridge._writer_cycle(time.monotonic() + 1.0, expected_deadline=None)
            repeated_once = bridge.status()
            bridge._writer_cycle(time.monotonic() + 1.0, expected_deadline=None)
            repeated_twice = bridge.status()

            assert repeated_once.writer_enqueued_target_sequence == 1
            assert repeated_twice.writer_enqueued_target_sequence == 1
            assert repeated_once.writer_enqueue_generation > first_generation
            assert (
                repeated_twice.writer_enqueue_generation
                > repeated_once.writer_enqueue_generation
            )
            assert repeated_once.writer_enqueued_q_rad == pytest.approx(first_q)
            assert repeated_twice.writer_enqueued_q_rad == pytest.approx(first_q)

        second_target = _target(config, 2, 0.5)
        assert (
            await bridge.submit(
                second_target,
                ownership_epoch=ownership_epoch,
                mapping_hash=config.mapping_hash,
            )
        ).ok
        with bridge._guard:
            bridge._writer_cycle(time.monotonic() + 1.0, expected_deadline=None)
            second_ack = bridge.status()

        assert second_ack.writer_enqueued_target_sequence == 2
        assert second_ack.writer_enqueue_generation > repeated_twice.writer_enqueue_generation
        assert second_ack.writer_enqueued_q_rad == pytest.approx((0.1,) * 12)
    finally:
        _force_cleanup_fake_owner(bridge, ownership_epoch)


@pytest.mark.parametrize("bad_crc", [-1, 0x1_0000_0000, True, 1.0])
def test_lowcmd_rejects_crc_outside_exact_uint32(
    tmp_path: Path,
    bad_crc: Any,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(lock_path=tmp_path / f"bad-crc-{bad_crc}.lock"),
    )

    class _BadCRC:
        def Crc(self, message: _LowCommand) -> Any:
            del message
            return bad_crc

    bridge._sdk = fake.bindings
    bridge._crc = _BadCRC()
    with pytest.raises(RuntimeError, match="unsigned 32-bit integer"):
        bridge._make_low_command(
            (0.0,) * 12,
            previous_q=(0.0,) * 12,
            elapsed_s=0.02,
            now_s=time.monotonic(),
        )


@pytest.mark.asyncio
async def test_revoke_rejects_bool_and_float_epochs_even_when_epoch_is_one(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "revoke-exact-epoch.lock",
            epoch_factory=lambda: 1,
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok

    for bad_epoch in (True, 1.0):
        rejected = await bridge.revoke(
            "reject Python numeric equality",
            ownership_epoch=bad_epoch,  # type: ignore[arg-type]
        )
        assert rejected.code == "GO2_OWNERSHIP_EPOCH_MISMATCH"
        assert config.mapping_hash is not None
        rejected_submit = await bridge.submit(
            _target(config, 1, 0.0),
            ownership_epoch=bad_epoch,  # type: ignore[arg-type]
            mapping_hash=config.mapping_hash,
        )
        assert rejected_submit.code == "GO2_OWNERSHIP_EPOCH_MISMATCH"
        rejected_release = await bridge.release(
            _permit(config),
            "reject non-exact release epoch",
            ownership_epoch=bad_epoch,  # type: ignore[arg-type]
        )
        assert rejected_release.code == "GO2_OWNERSHIP_EPOCH_MISMATCH"
        assert bridge.status().owner_epoch == 1
        assert bridge.status().writer_alive

    bridge._on_low_state(_LowState(config))
    assert (await bridge.release(_permit(config), "exact epoch cleanup", ownership_epoch=1)).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_total_pd_torque_envelope_clips_q_without_breaking_slew(
    tmp_path: Path,
) -> None:
    config = _low_level_config(
        kp=(1000.0,) * 12,
        kd=(0.0,) * 12,
        tau_ff_nm=(0.2,) * 12,
        tau_limit_nm=(1.0,) * 12,
        feedback_loss_degraded_kd=(0.0,) * 12,
        firmware_torque_limit_nm=(0.8,) * 12,
    )
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "torque-envelope.lock",
            epoch_factory=lambda: 701,
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    assert config.mapping_hash is not None
    assert (
        await bridge.submit(
            _target(config, 1, 0.5),
            ownership_epoch=701,
            mapping_hash=config.mapping_hash,
        )
    ).ok
    await asyncio.sleep(0.03)
    latest = fake.publisher.writes[-1]
    assert config.zero_offsets_rad is not None
    # 1000*q_error + 0.2 <= 1.0, hence q_error <= 0.0008 rad.
    commanded_q = latest.motor_cmd[0].q - config.zero_offsets_rad[0]
    assert 0.0 <= commanded_q <= 0.0008001
    assert abs(1000.0 * commanded_q + latest.motor_cmd[0].tau) <= 1.0001
    assert (await bridge.revoke("torque test hold", ownership_epoch=701)).ok
    bridge._on_low_state(_LowState(config))
    assert bridge.status().safe_hold_settled
    fake.motion.mode_name = "normal"
    assert (await bridge.release(_permit(config), "torque test", ownership_epoch=701)).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_infeasible_pd_envelope_derates_and_keeps_fault_stream_alive(
    tmp_path: Path,
) -> None:
    config = _low_level_config(
        safe_hold_ack_timeout_s=0.6,
        release_timeout_s=0.7,
    )
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "derated-fault-hold.lock",
            epoch_factory=lambda: 703,
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok

    moving_feedback = _LowState(config)
    assert config.directions is not None
    for index in range(12):
        moving_feedback.motor_state[index].dq = config.directions[index] * 0.4
    fake.subscriber.state = moving_feedback
    bridge._on_low_state(moving_feedback)

    # Simulate a discontinuity between the last enqueued position and current
    # feedback.  The normal Kp envelope then has no intersection with the
    # actual-time slew interval.  This must cause a bounded FAULT hold write,
    # not an exception or a silent gap in the sole command stream.
    with bridge._guard:
        bridge._last_commanded_q = (1.0,) * 12
    writes_before_fault = len(fake.publisher.writes)
    for _ in range(30):
        if (
            bridge.status().ownership_state is LowCmdOwnershipState.FAULT
            and len(fake.publisher.writes) > writes_before_fault
        ):
            break
        await asyncio.sleep(0.005)

    status = bridge.status()
    assert status.ownership_state is LowCmdOwnershipState.FAULT
    assert status.writer_alive
    first_derated = fake.publisher.writes[writes_before_fault]
    assert config.kp is not None
    assert config.kd is not None
    assert config.tau_ff_nm is not None
    assert config.tau_limit_nm is not None
    assert config.zero_offsets_rad is not None
    assert first_derated.motor_cmd[0].kp < config.kp[0]
    assert first_derated.motor_cmd[1].kp < config.kp[1]
    for joint_index, motor_id in enumerate(range(12)):
        motor = first_derated.motor_cmd[motor_id]
        direction = config.directions[joint_index]
        measured_q = 0.0
        algorithm_q = direction * (motor.q - config.zero_offsets_rad[joint_index])
        algorithm_tau_ff = direction * motor.tau
        kp_scale = motor.kp / config.kp[joint_index]
        kd_scale = motor.kd / config.kd[joint_index]
        tau_scale = algorithm_tau_ff / config.tau_ff_nm[joint_index]
        assert 0.0 <= kp_scale < 1.0
        assert kd_scale == pytest.approx(kp_scale)
        assert tau_scale == pytest.approx(kp_scale)
        conservative_envelope = (
            abs(motor.kp * (algorithm_q - measured_q)) + abs(motor.kd * 0.4) + abs(algorithm_tau_ff)
        )
        signed_total = motor.kp * (algorithm_q - measured_q) - motor.kd * 0.4 + algorithm_tau_ff
        assert conservative_envelope <= config.tau_limit_nm[joint_index] + 1.0e-8
        assert abs(signed_total) <= config.tau_limit_nm[joint_index] + 1.0e-8

    writes_after_fault = len(fake.publisher.writes)
    for _ in range(50):
        if len(fake.publisher.writes) > writes_after_fault:
            break
        await asyncio.sleep(0.005)
    assert len(fake.publisher.writes) > writes_after_fault
    assert bridge.status().writer_alive

    # Restore a reachable previous command only to keep test teardown short;
    # the production path would continue its bounded slew toward safe-hold.
    stationary_feedback = _LowState(config)
    fake.subscriber.state = stationary_feedback
    bridge._on_low_state(stationary_feedback)
    with bridge._guard:
        bridge._last_commanded_q = (0.0,) * 12
    for _ in range(30):
        if bridge.status().safe_hold_settled:
            break
        await asyncio.sleep(0.005)
    assert bridge.status().safe_hold_settled
    fake.motion.mode_name = "normal"
    assert (await bridge.release(_permit(config), "derating test", ownership_epoch=703)).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_limit_fault_uses_new_finite_qdq_instead_of_old_nominal_frame(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "latest-finite-envelope.lock",
            epoch_factory=lambda: 706,
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    fake.publisher.after_write = None

    limit_fault = _LowState(config)
    assert config.zero_offsets_rad is not None
    assert config.directions is not None
    limit_fault.motor_state[0].q = config.zero_offsets_rad[0] + config.directions[0] * 3.0
    writes_before = len(fake.publisher.writes)
    bridge._on_low_state(limit_fault)
    for _ in range(200):
        if (
            len(fake.publisher.writes) > writes_before
            and fake.publisher.writes[-1].motor_cmd[0].kp < config.kp[0]
        ):
            break
        await asyncio.sleep(0.005)

    assert bridge.status().ownership_state is LowCmdOwnershipState.FAULT
    latest = fake.publisher.writes[-1]
    assert config.kp is not None
    # Reusing the old q=0 frame would leave Kp unchanged.  The latest finite
    # q=3 frame forces an instantaneous absolute-component envelope derating.
    assert 0.0 < latest.motor_cmd[0].kp < config.kp[0]

    with bridge._guard:
        safe_hold_q = bridge._safe_hold_q
    assert safe_hold_q is not None
    assert config.zero_offsets_rad is not None
    assert config.directions is not None
    normal = _LowState(config)
    for joint_index, q_hold in enumerate(safe_hold_q):
        normal.motor_state[joint_index].q = (
            config.zero_offsets_rad[joint_index] + config.directions[joint_index] * q_hold
        )
    for _ in range(40):
        bridge._on_low_state(normal)
        if bridge.status().safe_hold_settled:
            break
        await asyncio.sleep(0.005)
    fake.motion.mode_name = "normal"
    assert (await bridge.release(_permit(config), "finite fault teardown", ownership_epoch=706)).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_lost_qdq_forces_commissioned_degraded_gains_and_latest_command_hold(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "lost-feedback-degraded.lock",
            epoch_factory=lambda: 707,
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    fake.publisher.after_write = None
    assert config.mapping_hash is not None
    assert (
        await bridge.submit(
            _target(config, 1, 0.2),
            ownership_epoch=707,
            mapping_hash=config.mapping_hash,
        )
    ).ok
    await asyncio.sleep(0.03)
    with bridge._guard:
        last_commanded = bridge._last_commanded_q
    assert last_commanded is not None

    lost = _LowState(config)
    lost.motor_state[0].lost = 1
    writes_before = len(fake.publisher.writes)
    bridge._on_low_state(lost)
    with bridge._guard:
        assert bridge._safe_hold_q == pytest.approx(last_commanded)
    for _ in range(40):
        if len(fake.publisher.writes) > writes_before:
            break
        await asyncio.sleep(0.005)

    latest = fake.publisher.writes[-1]
    assert config.feedback_loss_degraded_kp is not None
    assert config.feedback_loss_degraded_kd is not None
    assert config.feedback_loss_degraded_tau_ff_nm is not None
    for joint_index in range(12):
        motor = latest.motor_cmd[joint_index]
        assert motor.kp == pytest.approx(config.feedback_loss_degraded_kp[joint_index])
        assert motor.kd == pytest.approx(config.feedback_loss_degraded_kd[joint_index])
        assert abs(motor.tau) == pytest.approx(
            abs(config.feedback_loss_degraded_tau_ff_nm[joint_index])
        )

    with bridge._guard:
        safe_hold_q = bridge._safe_hold_q
    assert safe_hold_q is not None
    assert config.zero_offsets_rad is not None
    assert config.directions is not None
    normal = _LowState(config)
    for joint_index, q_hold in enumerate(safe_hold_q):
        normal.motor_state[joint_index].q = (
            config.zero_offsets_rad[joint_index] + config.directions[joint_index] * q_hold
        )
    for _ in range(40):
        bridge._on_low_state(normal)
        if bridge.status().safe_hold_settled:
            break
        await asyncio.sleep(0.005)
    fake.motion.mode_name = "normal"
    assert (await bridge.release(_permit(config), "lost feedback teardown", ownership_epoch=707)).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_out_of_order_valid_callback_cannot_roll_feedback_cache_back(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config, auto_feedback=False)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "out-of-order-lowstate.lock",
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert config.zero_offsets_rad is not None
    assert config.directions is not None

    entered = threading.Event()
    proceed = threading.Event()
    older = _BlockingLowState(config, entered, proceed)
    older._motor_state[0].q = config.zero_offsets_rad[0] + config.directions[0] * 0.1
    callback = threading.Thread(target=bridge._on_low_state, args=(older,), daemon=True)
    callback.start()
    assert entered.wait(0.2)

    newer = _LowState(config)
    newer.motor_state[0].q = config.zero_offsets_rad[0] + config.directions[0] * 0.2
    bridge._on_low_state(newer)
    proceed.set()
    callback.join(0.2)
    assert not callback.is_alive()
    status = bridge.status()
    assert status.motors[0].q_rad == pytest.approx(0.2)
    with bridge._guard:
        assert bridge._valid_motors[0].q_rad == pytest.approx(0.2)
        assert bridge._envelope_motors[0].q_rad == pytest.approx(0.2)
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_status_detects_writer_blocked_inside_dds_write(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "blocked-write-health.lock",
            epoch_factory=lambda: 708,
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok

    write_entered = threading.Event()
    write_may_finish = threading.Event()
    original_write = fake.publisher.Write

    def blocked_write(message: _LowCommand, timeout: Any = None) -> bool:
        write_entered.set()
        if not write_may_finish.wait(1.0):
            raise RuntimeError("test did not release blocked DDS Write")
        return original_write(message, timeout)

    fake.publisher.Write = blocked_write  # type: ignore[method-assign]
    assert await asyncio.get_running_loop().run_in_executor(None, write_entered.wait, 0.2)
    feedback_thread = threading.Thread(
        target=bridge._on_low_state,
        args=(_LowState(config),),
        daemon=True,
    )
    feedback_thread.start()

    started = time.monotonic()
    status = bridge.status()
    assert time.monotonic() - started < 0.2
    assert not status.healthy
    assert not status.watchdog_healthy
    assert status.writer_alive
    assert "DDS Write/owner guard" in str(status.fault_reason)

    async def event_loop_heartbeat() -> float:
        await asyncio.sleep(0.01)
        return time.monotonic()

    heartbeat_started_s = time.monotonic()
    heartbeat = asyncio.create_task(event_loop_heartbeat())
    assert config.mapping_hash is not None
    submit_started_s = time.monotonic()
    submit_result = await bridge.submit(
        _target(config, 1, 0.0),
        ownership_epoch=708,
        mapping_hash=config.mapping_hash,
    )
    submit_elapsed_s = time.monotonic() - submit_started_s
    revoke_started_s = time.monotonic()
    revoke_result = await bridge.revoke(
        "blocked writer must not block revoke",
        ownership_epoch=708,
    )
    revoke_elapsed_s = time.monotonic() - revoke_started_s
    release_started_s = time.monotonic()
    release_result = await bridge.release(
        _permit(config, ttl=0.05),
        "blocked writer must not fake handback",
        ownership_epoch=708,
    )
    release_elapsed_s = time.monotonic() - release_started_s
    heartbeat_observed_s = await asyncio.wait_for(heartbeat, timeout=0.05)

    assert submit_result.code == "GO2_OWNER_GUARD_TIMEOUT"
    assert revoke_result.code == "GO2_OWNER_GUARD_TIMEOUT"
    assert release_result.code == "GO2_OWNER_GUARD_TIMEOUT"
    assert submit_elapsed_s < 0.1
    assert revoke_elapsed_s < 0.1
    assert release_elapsed_s < 0.1
    assert heartbeat_observed_s - heartbeat_started_s < 0.25
    assert bridge._owner_epoch == 708
    assert bridge._arbiter.status().low_level_epoch == 708
    assert not fake.publisher.closed
    assert fake.motion.select_calls == []

    write_may_finish.set()
    await asyncio.get_running_loop().run_in_executor(None, feedback_thread.join, 0.3)
    assert not feedback_thread.is_alive()
    fake.publisher.Write = original_write  # type: ignore[method-assign]
    for _ in range(40):
        if bridge.status().safe_hold_settled:
            break
        await asyncio.sleep(0.005)
    fake.motion.mode_name = "normal"
    assert (await bridge.release(_permit(config), "blocked write teardown", ownership_epoch=708)).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_pre_write_lowstate_cannot_acknowledge_safe_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _low_level_config(
        acquire_timeout_s=0.6,
        safe_hold_ack_timeout_s=0.4,
        release_timeout_s=0.6,
    )
    fake = _fake_sdk(config, auto_feedback=False)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "causal-lowstate.lock",
            epoch_factory=lambda: 704,
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok

    callback_entered = threading.Event()
    callback_may_finish = threading.Event()
    original_begin = bridge._begin_low_state_callback
    stale_ingress_token = 0

    def pause_after_ingress_registration(
        subscription_generation: int,
    ) -> Tuple[int, int, float]:
        nonlocal stale_ingress_token
        ingress = original_begin(subscription_generation)
        assert ingress is not None
        token, transaction, _receipt_time = ingress
        stale_ingress_token = token
        callback_entered.set()
        if not callback_may_finish.wait(1.0):
            raise RuntimeError("test did not release the registered callback")
        # Recreate the historical race: timestamp capture happens only after
        # the safe-hold Write.  The ingress-token fence must reject the frame
        # even though this deliberately late timestamp would otherwise pass.
        return token, transaction, time.monotonic()

    monkeypatch.setattr(
        bridge,
        "_begin_low_state_callback",
        pause_after_ingress_registration,
    )
    stale_callback = threading.Thread(
        target=bridge._on_low_state,
        args=(_LowState(config),),
        daemon=True,
    )
    stale_callback.start()
    assert callback_entered.wait(0.2)

    acquire_task = asyncio.create_task(bridge.acquire(_permit(config, ttl=0.5)))
    for _ in range(40):
        with bridge._guard:
            reached = (
                bridge._safe_hold_command_reached_generation == bridge._safe_hold_request_generation
                and bridge._safe_hold_request_generation > 0
            )
        if reached:
            break
        await asyncio.sleep(0.005)
    assert reached
    with bridge._guard:
        required_ingress_token = bridge._safe_hold_feedback_ingress_token_required
    assert stale_ingress_token > 0
    assert required_ingress_token > stale_ingress_token

    callback_may_finish.set()
    await asyncio.get_running_loop().run_in_executor(None, stale_callback.join, 0.2)
    assert not stale_callback.is_alive()
    assert not bridge.status().safe_hold_settled

    # A genuinely post-write frame crosses the sequence, ingress-token and
    # monotonic timestamp fences and may acknowledge the same hold generation.
    for _ in range(20):
        bridge._on_low_state(_LowState(config))
        if bridge.status().safe_hold_settled:
            break
        await asyncio.sleep(0.001)
    assert bridge.status().safe_hold_settled
    assert (await acquire_task).ok
    fake.motion.mode_name = "normal"
    assert (await bridge.release(_permit(config), "causal ACK test", ownership_epoch=704)).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_safe_hold_does_not_settle_until_slew_reaches_hold_command(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    config = _low_level_config(
        # This test exercises slew/settling semantics, not the scheduler
        # watchdog. A slower test-only period leaves enough Windows/CI
        # scheduling margin that unrelated process load cannot trip jitter
        # before the target has moved.
        send_period_s=0.1,
        maximum_jitter_s=0.099,
        target_ttl_s=2.0,
        acquire_timeout_s=1.0,
        release_timeout_s=2.5,
        safe_hold_ack_timeout_s=2.0,
        tau_limit_nm=(50.0,) * 12,
        firmware_torque_limit_nm=(40.0,) * 12,
    )
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "safe-hold-slew.lock",
            epoch_factory=lambda: 705,
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    request.addfinalizer(lambda: _force_cleanup_fake_owner(bridge, 705))
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    try:
        assert config.mapping_hash is not None
        next_sequence = 1
        assert (
            await bridge.submit(
                _target(config, next_sequence, 0.5),
                ownership_epoch=705,
                mapping_hash=config.mapping_hash,
            )
        ).ok
        movement_deadline = time.monotonic() + 1.5
        last: Optional[Tuple[float, ...]] = None
        while time.monotonic() < movement_deadline:
            last = bridge._last_commanded_q
            if last is not None and last[0] >= 0.1:
                break
            status = bridge.status()
            if status.writer_enqueued_target_sequence == next_sequence:
                # A mailbox sequence is frozen at its first writer-limited q.
                # Progress toward a farther target therefore requires a fresh
                # strictly increasing sequence from the high-rate controller.
                next_sequence += 1
                submitted = await bridge.submit(
                    _target(config, next_sequence, 0.5),
                    ownership_epoch=705,
                    mapping_hash=config.mapping_hash,
                )
                assert submitted.ok
            await asyncio.sleep(0.01)
        assert last is not None and last[0] >= 0.1

        measured_hold = _LowState(config)
        assert config.zero_offsets_rad is not None
        assert config.directions is not None
        for index in range(12):
            measured_hold.motor_state[index].q = (
                config.zero_offsets_rad[index] + config.directions[index] * -0.1
            )
        fake.subscriber.state = measured_hold
        bridge._on_low_state(measured_hold)

        revoke_task = asyncio.create_task(
            bridge.revoke("large slew to measured hold", ownership_epoch=705)
        )
        safe_hold_deadline = time.monotonic() + 1.0
        status = bridge.status()
        while time.monotonic() < safe_hold_deadline and not status.safe_hold_active:
            await asyncio.sleep(0.01)
            status = bridge.status()
        assert status.safe_hold_active
        assert not status.safe_hold_settled
        with bridge._guard:
            assert (
                bridge._safe_hold_command_reached_generation != bridge._safe_hold_request_generation
            )

        assert (await revoke_task).ok
        assert bridge.status().safe_hold_settled
        fake.motion.mode_name = "normal"
        assert (
            await bridge.release(
                _permit(config, ttl=1.0),
                "slew settle test",
                ownership_epoch=705,
            )
        ).ok
        assert (await bridge.disconnect()).ok
    finally:
        # Production must use verified ground release. This fallback is only
        # for fake transports when an assertion has already failed.
        _force_cleanup_fake_owner(bridge, 705)


def test_slew_limit_uses_actual_short_interval(tmp_path: Path) -> None:
    config = _low_level_config()
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(lock_path=tmp_path / "slew.lock"),
    )
    limited = bridge._slew_limit((0.0,) * 12, (1.0,) * 12, 0.005)
    assert limited == pytest.approx((0.005,) * 12)


@pytest.mark.asyncio
async def test_bridge_serializes_owner_lifecycle_operations(tmp_path: Path) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    audit_started = threading.Event()
    release_audit = threading.Event()

    def blocking_audit() -> bool:
        audit_started.set()
        return release_audit.wait(0.2)

    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "lifecycle.lock",
            epoch_factory=lambda: 702,
            network_exclusivity_verifier=blocking_audit,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    acquire_task = asyncio.create_task(bridge.acquire(_permit(config)))
    assert await asyncio.get_running_loop().run_in_executor(None, audit_started.wait, 0.1)
    busy = await bridge.disconnect()
    assert busy.code == "GO2_LOW_LEVEL_LIFECYCLE_BUSY"
    release_audit.set()
    assert (await acquire_task).ok
    fake.motion.mode_name = "normal"
    assert (await bridge.release(_permit(config), "lifecycle test", ownership_epoch=702)).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_release_fails_closed_when_high_level_handoff_is_unconfirmed(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    fake.motion.select_activates = False
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "handoff.lock",
        epoch_factory=lambda: 88,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    bridge._on_low_state(_LowState(config))
    assert bridge.status().safe_hold_settled
    released = await bridge.release(_permit(config), "handoff test", ownership_epoch=88)
    assert released.code == "GO2_HIGH_LEVEL_HANDOFF_UNCONFIRMED"
    status = bridge.status()
    assert status.ownership_state is LowCmdOwnershipState.FAULT
    assert not status.healthy
    assert status.high_level_released
    assert not status.writer_alive
    assert status.owner_epoch == 88
    assert status.network_exclusivity_verified
    assert arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).code == "GO2_LOW_LEVEL_OWNER_ACTIVE"
    contender = Go2ControlArbiter(
        lock_path=tmp_path / "handoff.lock",
        epoch_factory=lambda: 888,
        network_exclusivity_verifier=lambda: True,
    )
    assert contender.acquire_low_level().code == "GO2_LOW_LEVEL_OWNER_LOCKED"
    # Once Close() has succeeded there is no LowCmd stream to preserve. A
    # telemetry failure must remain visible, but it must not make exact
    # high-level recovery impossible.
    bridge._on_low_state(object())
    assert bridge.status().fault_reason is not None
    assert not bridge.status().publisher_active
    fake.motion.select_activates = True
    assert (await bridge.release(_permit(config), "confirm delayed handoff", ownership_epoch=88)).ok
    assert bridge.status().ownership_state is LowCmdOwnershipState.OBSERVE_ONLY
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_publisher_close_failure_retains_epoch_and_forbids_select_mode(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "publisher-close.lock",
        epoch_factory=lambda: 881,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    fake.publisher.close_error = True

    rejected = await bridge.release(_permit(config), "injected close failure", ownership_epoch=881)
    assert rejected.code == "GO2_LOW_CMD_PUBLISHER_CLOSE_FAILED"
    assert rejected.data["owner_lock_retained"] is True
    assert bridge.status().ownership_state is LowCmdOwnershipState.FAULT
    assert bridge.status().owner_epoch == 881
    assert not bridge.status().writer_alive
    assert arbiter.status().local_single_instance_held
    assert fake.motion.select_calls == []
    assert bridge._publisher is fake.publisher

    fake.publisher.close_error = False
    recovered = await bridge.release(_permit(config), "retry verified close", ownership_epoch=881)
    assert recovered.ok
    assert fake.publisher.closed
    assert fake.motion.select_calls == ["normal"]
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_close_exception_retry_requires_binding_idempotency_evidence(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    fake.bindings = replace(
        fake.bindings,
        publisher_close_retry_idempotency_verified=False,
    )
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "publisher-close-retry-unverified.lock",
        epoch_factory=lambda: 884,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok

    close_calls = 0
    original_close = fake.publisher.Close

    def counted_close() -> None:
        nonlocal close_calls
        close_calls += 1
        original_close()

    fake.publisher.Close = counted_close  # type: ignore[method-assign]
    fake.publisher.close_error = True
    first = await bridge.release(
        _permit(config),
        "inject ambiguous Close exception",
        ownership_epoch=884,
    )
    fake.publisher.close_error = False
    retry = await bridge.release(
        _permit(config),
        "retry without idempotency evidence",
        ownership_epoch=884,
    )

    assert first.code == "GO2_LOW_CMD_PUBLISHER_CLOSE_FAILED"
    assert retry.code == "GO2_LOW_CMD_PUBLISHER_CLOSE_RETRY_UNVERIFIED"
    assert retry.data["process_restart_or_manual_recovery_required"] is True
    assert close_calls == 1
    assert fake.motion.select_calls == []
    assert bridge._owner_epoch == 884
    assert arbiter.status().low_level_epoch == 884

    _force_cleanup_fake_owner(bridge, 884)
    assert close_calls == 2
    assert not arbiter.status().local_single_instance_held


@pytest.mark.asyncio
async def test_blocked_publisher_close_keeps_event_loop_alive_and_epoch_owned(
    tmp_path: Path,
) -> None:
    config = _low_level_config(
        release_timeout_s=0.2,
        safe_hold_ack_timeout_s=0.1,
    )
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "publisher-close-timeout.lock",
        epoch_factory=lambda: 882,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok

    close_entered = threading.Event()
    close_may_finish = threading.Event()
    original_close = fake.publisher.Close

    def blocked_close() -> None:
        close_entered.set()
        if not close_may_finish.wait(1.0):
            raise RuntimeError("test did not release blocked DDS Close")
        original_close()

    fake.publisher.Close = blocked_close  # type: ignore[method-assign]
    release_task = asyncio.create_task(
        bridge.release(
            _permit(config, ttl=0.15),
            "blocked Close must not freeze the event loop",
            ownership_epoch=882,
        )
    )
    assert await asyncio.get_running_loop().run_in_executor(None, close_entered.wait, 0.3)
    heartbeat = asyncio.create_task(asyncio.sleep(0.02, result=True))

    assert await asyncio.wait_for(heartbeat, timeout=0.1)
    rejected = await asyncio.wait_for(release_task, timeout=0.4)

    assert rejected.code == "GO2_LOW_CMD_PUBLISHER_CLOSE_TIMEOUT"
    assert rejected.data["owner_lock_retained"] is True
    assert bridge._owner_epoch == 882
    assert bridge._publisher is fake.publisher
    assert arbiter.status().low_level_epoch == 882
    assert fake.motion.select_calls == []
    assert bridge._writer_thread is None

    close_may_finish.set()
    for _ in range(100):
        task = bridge._publisher_close_task
        if task is not None and task.done():
            break
        await asyncio.sleep(0.005)
    recovered = await bridge.release(
        _permit(config, ttl=0.15),
        "retry after the retained Close result",
        ownership_epoch=882,
    )
    assert recovered.ok
    assert fake.publisher.closed
    assert fake.motion.select_calls == ["normal"]
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_successful_close_is_not_repeated_when_commit_guard_times_out(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    config = _low_level_config(release_timeout_s=0.5)
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "publisher-close-result-latch.lock",
        epoch_factory=lambda: 883,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    request.addfinalizer(lambda: _force_cleanup_fake_owner(bridge, 883))
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok

    guard_held = threading.Event()
    release_guard = threading.Event()
    holder: Optional[threading.Thread] = None
    close_calls = 0
    original_close = fake.publisher.Close

    def hold_commit_guard() -> None:
        with bridge._guard:
            guard_held.set()
            if not release_guard.wait(1.0):
                raise RuntimeError("test did not release the commit guard")

    def close_then_block_commit() -> None:
        nonlocal close_calls, holder
        close_calls += 1
        original_close()
        holder = threading.Thread(target=hold_commit_guard, daemon=True)
        holder.start()
        if not guard_held.wait(0.2):
            raise RuntimeError("test did not acquire the commit guard")

    fake.publisher.Close = close_then_block_commit  # type: ignore[method-assign]
    first = await bridge.release(
        _permit(config),
        "inject contention after successful Close",
        ownership_epoch=883,
    )

    assert first.code == "GO2_OWNER_GUARD_TIMEOUT"
    assert close_calls == 1
    assert bridge._publisher is fake.publisher
    assert bridge._publisher_close_task is not None
    assert bridge._publisher_close_task.done()
    assert fake.motion.select_calls == []
    assert arbiter.status().low_level_epoch == 883

    release_guard.set()
    assert holder is not None
    await asyncio.get_running_loop().run_in_executor(None, holder.join, 0.3)
    assert not holder.is_alive()
    recovered = await bridge.release(
        _permit(config),
        "reuse the retained successful Close result",
        ownership_epoch=883,
    )

    assert recovered.ok
    assert close_calls == 1
    assert fake.motion.select_calls == ["normal"]
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_release_keeps_writer_when_post_write_safe_hold_is_not_settled(
    tmp_path: Path,
) -> None:
    config = _low_level_config(
        safe_hold_ack_timeout_s=0.2,
        release_timeout_s=0.4,
    )
    fake = _fake_sdk(config, auto_feedback=False)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "unsettled-release.lock",
        epoch_factory=lambda: 889,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    acquire_task = asyncio.create_task(bridge.acquire(_permit(config)))
    for _ in range(40):
        # A real LowState stream is continuous.  Repeated frames also cover
        # the intentional race where a callback captured its timestamp just
        # before the writer committed the causal safe-hold write fence.
        bridge._on_low_state(_LowState(config))
        if bridge.status().safe_hold_settled:
            break
        await asyncio.sleep(0.005)
    assert (await acquire_task).ok
    assert config.mapping_hash is not None
    assert (
        await bridge.submit(
            _target(config, 1, 0.1),
            ownership_epoch=889,
            mapping_hash=config.mapping_hash,
        )
    ).ok
    rejected = await bridge.release(
        _permit(config, ttl=0.3), "must await feedback", ownership_epoch=889
    )
    assert rejected.code == "GO2_SAFE_HOLD_NOT_SETTLED"
    rejected_status = bridge.status()
    assert rejected_status.writer_alive
    assert rejected_status.owner_epoch == 889
    assert arbiter.status().local_single_instance_held

    for _ in range(40):
        bridge._on_low_state(_LowState(config))
        if bridge.status().safe_hold_settled:
            break
        await asyncio.sleep(0.005)
    assert bridge.status().safe_hold_settled
    fake.motion.mode_name = "normal"
    assert (await bridge.release(_permit(config, ttl=0.3), "settled retry", ownership_epoch=889)).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_post_stop_lowstate_fault_restarts_hold_before_publisher_close(
    tmp_path: Path,
) -> None:
    config = _low_level_config(release_timeout_s=0.7)
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "post-stop-feedback-race.lock",
        epoch_factory=lambda: 890,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    bridge._on_low_state(_LowState(config))
    assert bridge.status().safe_hold_settled

    original_stop = bridge._writer_stop
    injected = False

    class _InjectingStopEvent:
        def is_set(self) -> bool:
            return original_stop.is_set()

        def wait(self, timeout: Any = None) -> bool:
            return original_stop.wait(timeout)

        def clear(self) -> None:
            original_stop.clear()

        def set(self) -> None:
            nonlocal injected
            original_stop.set()
            if not injected:
                injected = True
                invalid = _LowState(config)
                invalid.motor_state[0].lost = 1
                bridge._on_low_state(invalid)

    bridge._writer_stop = _InjectingStopEvent()  # type: ignore[assignment]

    rejected = await bridge.release(
        _permit(config, ttl=0.5), "inject stop-boundary fault", ownership_epoch=890
    )

    assert rejected.code == "GO2_RELEASE_POST_STOP_FENCE_FAILED"
    assert rejected.data["owner_lock_retained"] is True
    assert rejected.data["publisher_close_called"] is False
    assert rejected.data["high_level_select_called"] is False
    assert not fake.publisher.closed
    assert fake.motion.select_calls == []
    assert arbiter.status().local_single_instance_held
    for _ in range(80):
        bridge._on_low_state(_LowState(config))
        if bridge.status().safe_hold_settled:
            break
        await asyncio.sleep(0.005)
    assert bridge.status().writer_alive
    assert bridge.status().safe_hold_settled
    assert (
        await bridge.release(
            _permit(config, ttl=0.5), "settled post-race retry", ownership_epoch=890
        )
    ).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_lowstate_fault_at_close_boundary_cannot_be_cleared_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _low_level_config(release_timeout_s=0.7)
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "close-boundary-feedback-race.lock",
        epoch_factory=lambda: 891,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    bridge._on_low_state(_LowState(config))
    assert bridge.status().safe_hold_settled

    original_close = bridge._close_owned_publisher

    def inject_fault_then_close(publisher: object) -> OperationResult:
        invalid = _LowState(config)
        invalid.motor_state[0].lost = 1
        bridge._on_low_state(invalid)
        return original_close(publisher)

    monkeypatch.setattr(bridge, "_close_owned_publisher", inject_fault_then_close)
    rejected = await bridge.release(
        _permit(config, ttl=0.5), "inject close-boundary fault", ownership_epoch=891
    )

    assert rejected.code == "GO2_HANDOFF_FEEDBACK_CHANGED"
    assert rejected.data["owner_lock_retained"] is True
    assert rejected.data["publisher_closed"] is True
    assert rejected.data["high_level_reactivation_acknowledged"] is True
    assert fake.publisher.closed
    assert fake.motion.select_calls == ["normal"]
    status = bridge.status()
    assert status.ownership_state is LowCmdOwnershipState.FAULT
    assert status.owner_epoch == 891
    assert not status.high_level_released
    assert arbiter.status().local_single_instance_held

    monkeypatch.setattr(bridge, "_close_owned_publisher", original_close)
    bridge._on_low_state(_LowState(config))
    recovered = await bridge.release(
        _permit(config, ttl=0.5), "fresh feedback retry", ownership_epoch=891
    )
    assert recovered.ok
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_valid_but_moving_lowstate_at_close_boundary_is_sticky(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _low_level_config(release_timeout_s=0.7)
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "close-boundary-motion-race.lock",
        epoch_factory=lambda: 892,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    bridge._on_low_state(_LowState(config))
    assert bridge.status().safe_hold_settled

    original_close = bridge._close_owned_publisher

    def inject_motion_then_close(publisher: object) -> OperationResult:
        moving = _LowState(config)
        assert config.zero_offsets_rad is not None
        assert config.directions is not None
        moving.motor_state[0].q = config.zero_offsets_rad[0] + config.directions[0] * 0.03
        bridge._on_low_state(moving)
        return original_close(publisher)

    monkeypatch.setattr(bridge, "_close_owned_publisher", inject_motion_then_close)
    rejected = await bridge.release(
        _permit(config, ttl=0.5), "inject valid motion at Close", ownership_epoch=892
    )

    assert rejected.code == "GO2_HANDOFF_FEEDBACK_CHANGED"
    assert rejected.data["owner_lock_retained"] is True
    assert rejected.data["publisher_closed"] is True
    assert "safe-hold" in str(rejected.data["late_feedback_fault"])
    assert bridge.status().owner_epoch == 892
    assert not bridge.status().publisher_active
    assert arbiter.status().local_single_instance_held

    monkeypatch.setattr(bridge, "_close_owned_publisher", original_close)
    # This explicit ground-authorized retry is handback-only and must not
    # depend on returning LowState to the frozen LowCmd pose.
    recovered = await bridge.release(
        _permit(config, ttl=0.5), "recover closed endpoint", ownership_epoch=892
    )
    assert recovered.ok
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_pre_fence_inflight_lowstate_cannot_escape_after_publisher_close(
    tmp_path: Path,
) -> None:
    config = _low_level_config(release_timeout_s=0.9)
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "inflight-lowstate-close-race.lock",
        epoch_factory=lambda: 895,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    bridge._on_low_state(_LowState(config))
    assert bridge.status().safe_hold_settled

    callback_entered = threading.Event()
    release_callback = threading.Event()
    delayed = _BlockingLowState(config, callback_entered, release_callback)
    assert config.zero_offsets_rad is not None
    assert config.directions is not None
    delayed._motor_state[0].q = config.zero_offsets_rad[0] + config.directions[0] * 0.03
    callback_thread = threading.Thread(
        target=bridge._on_low_state,
        args=(delayed,),
        daemon=True,
    )
    callback_thread.start()
    assert callback_entered.wait(0.2)

    release_task = asyncio.create_task(
        bridge.release(
            _permit(config, ttl=0.8),
            "drain pre-fence callback",
            ownership_epoch=895,
        )
    )
    for _ in range(160):
        if fake.publisher.closed and not bridge.status().publisher_active:
            break
        await asyncio.sleep(0.005)
    assert fake.publisher.closed
    assert not bridge.status().publisher_active

    # The callback entered before the final fence, but only completes after
    # Close and after the bridge clears its publisher reference.
    release_callback.set()
    callback_thread.join(0.5)
    assert not callback_thread.is_alive()
    rejected = await release_task

    assert rejected.code == "GO2_HANDOFF_FEEDBACK_CHANGED"
    assert rejected.data["owner_lock_retained"] is True
    assert rejected.data["publisher_closed"] is True
    assert bridge.status().owner_epoch == 895
    assert arbiter.status().local_single_instance_held
    assert (
        await bridge.release(
            _permit(config, ttl=0.5),
            "recover drained callback",
            ownership_epoch=895,
        )
    ).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_callback_registered_at_close_return_is_in_same_handoff_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _low_level_config(release_timeout_s=0.9)
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "close-return-registration-race.lock",
        epoch_factory=lambda: 896,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    bridge._on_low_state(_LowState(config))
    assert bridge.status().safe_hold_settled

    callback_entered = threading.Event()
    release_callback = threading.Event()
    callback_threads: List[threading.Thread] = []
    original_close = bridge._close_owned_publisher

    def close_then_register_callback(publisher: object) -> OperationResult:
        result = original_close(publisher)
        delayed = _BlockingLowState(config, callback_entered, release_callback)
        assert config.zero_offsets_rad is not None
        assert config.directions is not None
        delayed._motor_state[0].q = config.zero_offsets_rad[0] + config.directions[0] * 0.03
        callback_thread = threading.Thread(
            target=bridge._on_low_state,
            args=(delayed,),
            daemon=True,
        )
        callback_threads.append(callback_thread)
        callback_thread.start()
        assert callback_entered.wait(0.2)
        return result

    monkeypatch.setattr(
        bridge,
        "_close_owned_publisher",
        close_then_register_callback,
    )
    release_task = asyncio.create_task(
        bridge.release(
            _permit(config, ttl=0.8),
            "callback registers as Close returns",
            ownership_epoch=896,
        )
    )
    for _ in range(160):
        if callback_entered.is_set() and not bridge.status().publisher_active:
            break
        await asyncio.sleep(0.005)
    assert callback_entered.is_set()
    assert not bridge.status().publisher_active

    release_callback.set()
    for callback_thread in callback_threads:
        callback_thread.join(0.5)
        assert not callback_thread.is_alive()
    rejected = await release_task

    assert rejected.code == "GO2_HANDOFF_FEEDBACK_CHANGED"
    assert rejected.data["owner_lock_retained"] is True
    assert bridge.status().owner_epoch == 896
    monkeypatch.setattr(bridge, "_close_owned_publisher", original_close)
    assert (
        await bridge.release(
            _permit(config, ttl=0.5),
            "recover Close-return callback",
            ownership_epoch=896,
        )
    ).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_final_live_ground_failure_restarts_same_safe_hold_writer(
    tmp_path: Path,
) -> None:
    config = _low_level_config(release_timeout_s=0.7)
    fake = _fake_sdk(config)
    release_checks = 0

    def changing_ground(transfer: str, permit: Go2OwnershipPermit) -> OperationResult:
        nonlocal release_checks
        assert isinstance(permit, Go2OwnershipPermit)
        if transfer == "release":
            release_checks += 1
            if release_checks == 3:
                return OperationResult.failure(
                    "TEST_GROUND_CHANGED", "injected final boundary change"
                )
        return OperationResult.success("test ground remains safe")

    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "final-ground-race.lock",
        epoch_factory=lambda: 893,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=changing_ground,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    bridge._on_low_state(_LowState(config))
    assert bridge.status().safe_hold_settled

    rejected = await bridge.release(
        _permit(config, ttl=0.5), "ground changes after join", ownership_epoch=893
    )

    assert rejected.code == "TEST_GROUND_CHANGED"
    assert rejected.data["publisher_close_called"] is False
    assert rejected.data["owner_lock_retained"] is True
    assert not fake.publisher.closed
    assert fake.motion.select_calls == []
    assert bridge.status().publisher_active
    for _ in range(80):
        bridge._on_low_state(_LowState(config))
        if bridge.status().safe_hold_settled:
            break
        await asyncio.sleep(0.005)
    assert bridge.status().writer_alive
    assert bridge.status().safe_hold_settled
    assert (
        await bridge.release(_permit(config, ttl=0.5), "stable final retry", ownership_epoch=893)
    ).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_callback_at_epoch_unlock_is_ordered_after_atomic_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _low_level_config(release_timeout_s=0.7)
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "atomic-epoch-commit.lock",
        epoch_factory=lambda: 894,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    bridge._on_low_state(_LowState(config))
    assert bridge.status().safe_hold_settled

    original_release = arbiter.release_low_level
    callback_started = threading.Event()
    callback_finished = threading.Event()
    callback_threads: List[threading.Thread] = []

    def release_while_callback_waits(epoch: int) -> OperationResult:
        def callback() -> None:
            callback_started.set()
            bridge._on_low_state(object())
            callback_finished.set()

        callback_thread = threading.Thread(target=callback, daemon=True)
        callback_threads.append(callback_thread)
        callback_thread.start()
        assert callback_started.wait(0.2)
        # The callback has started but must be blocked on the bridge guard
        # until arbiter unlock and local epoch clear are both complete.
        assert not callback_finished.wait(0.01)
        return original_release(epoch)

    monkeypatch.setattr(arbiter, "release_low_level", release_while_callback_waits)
    released = await bridge.release(
        _permit(config, ttl=0.5), "atomic callback fence", ownership_epoch=894
    )
    assert released.ok
    for callback_thread in callback_threads:
        callback_thread.join(0.5)
    assert callback_finished.is_set()
    status = bridge.status()
    assert status.ownership_state is LowCmdOwnershipState.OBSERVE_ONLY
    assert status.owner_epoch == 0
    assert status.fault_reason is not None
    assert not status.healthy
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_arbiter_release_exception_retains_visible_recovery_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _low_level_config(release_timeout_s=0.7)
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "arbiter-release-exception.lock",
        epoch_factory=lambda: 897,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    bridge._on_low_state(_LowState(config))
    assert bridge.status().safe_hold_settled

    original_release = arbiter.release_low_level

    def raise_release(epoch: int) -> OperationResult:
        del epoch
        raise OSError("injected arbiter release exception")

    monkeypatch.setattr(arbiter, "release_low_level", raise_release)
    failed = await bridge.release(
        _permit(config, ttl=0.5),
        "inject arbiter exception",
        ownership_epoch=897,
    )

    assert failed.code == "GO2_ARBITER_RELEASE_FAILED"
    assert failed.data["owner_epoch_retained"] is True
    assert bridge.status().owner_epoch == 897
    assert not bridge.status().publisher_active
    assert fake.publisher.closed
    assert arbiter.status().local_single_instance_held

    monkeypatch.setattr(arbiter, "release_low_level", original_release)
    assert (
        await bridge.release(
            _permit(config, ttl=0.5),
            "recover arbiter exception",
            ownership_epoch=897,
        )
    ).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_release_mode_ack_then_check_failure_retains_lock_without_lowcmd(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    fake.motion.fail_check_after_release = True
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "ambiguous-acquire.lock",
        epoch_factory=lambda: 188,
        network_exclusivity_verifier=lambda: True,
        network_exclusivity_verifier_name="test-dds-graph-audit",
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    acquired = await bridge.acquire(_permit(config))
    assert acquired.code == "GO2_MOTION_CHECK_REJECTED"
    assert acquired.data["release_rpc_attempted"] is True
    assert acquired.data["release_rpc_acknowledged"] is True
    assert acquired.data["owner_lock_retained"] is True
    status = bridge.status()
    assert status.ownership_state is LowCmdOwnershipState.FAULT
    assert status.owner_epoch == 188
    assert not status.high_level_released
    assert not status.safe_hold_active
    assert not status.writer_alive
    assert status.network_exclusivity_verified
    assert fake.publisher.writes == []
    assert acquired.data["fault_hold_started"] is False
    assert arbiter.status().local_single_instance_held
    with pytest.raises(ControlOwnershipError):
        with arbiter.sport_lease():
            pass

    fake.motion.fail_check_after_release = False
    fake.motion.mode_name = "normal"
    assert (
        await bridge.release(_permit(config), "recover ambiguous acquire", ownership_epoch=188)
    ).ok
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_release_mode_without_ack_retains_lock_but_does_not_start_lowcmd(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    fake.motion.release_code = 5
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "release-rejected.lock",
        epoch_factory=lambda: 189,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    acquired = await bridge.acquire(_permit(config))
    assert acquired.code == "GO2_MOTION_RELEASE_REJECTED"
    assert acquired.data["release_rpc_attempted"] is True
    assert acquired.data["release_rpc_acknowledged"] is False
    assert acquired.data["owner_lock_retained"] is True
    assert acquired.data["fault_hold_started"] is False
    status = bridge.status()
    assert status.ownership_state is LowCmdOwnershipState.FAULT
    assert status.owner_epoch == 189
    assert not status.writer_alive
    assert not status.safe_hold_active
    assert fake.publisher.writes == []
    assert arbiter.status().local_single_instance_held

    # CheckMode still reports the high-level service, so a ground release can
    # now finish the retained transaction and unlock the host.
    assert (
        await bridge.release(_permit(config), "recover rejected release RPC", ownership_epoch=189)
    ).ok
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_partial_publisher_init_and_close_failure_retains_exact_owner(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    fake.publisher.init_error = True
    fake.publisher.close_error = True
    lock_path = tmp_path / "partial-init-close.lock"
    arbiter = Go2ControlArbiter(
        lock_path=lock_path,
        epoch_factory=lambda: 190,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok

    acquired = await bridge.acquire(_permit(config))

    assert acquired.code == "GO2_LOW_CMD_PUBLISHER_CLOSE_FAILED"
    assert acquired.data["original_failure_code"] == "GO2_LOW_CMD_PUBLISHER_INIT_FAILED"
    assert acquired.data["owner_lock_retained"] is True
    assert acquired.data["release_rpc_attempted"] is False
    assert bridge.status().owner_epoch == 190
    assert bridge._publisher is fake.publisher
    assert fake.motion.release_calls == 0
    contender = Go2ControlArbiter(
        lock_path=lock_path,
        network_exclusivity_verifier=lambda: True,
    )
    assert contender.acquire_low_level().code == "GO2_LOW_LEVEL_OWNER_LOCKED"

    fake.publisher.close_error = False
    recovered = await bridge.release(
        _permit(config), "retry pre-release cleanup", ownership_epoch=190
    )
    assert recovered.ok
    assert fake.motion.select_calls == []
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_final_pre_release_revalidation_close_failure_retains_exact_owner(
    tmp_path: Path,
) -> None:
    config = _low_level_config(acquire_timeout_s=0.05)
    fake = _fake_sdk(config)
    fake.publisher.init_delay_s = 0.06
    fake.publisher.close_error = True
    lock_path = tmp_path / "final-revalidation-close.lock"
    arbiter = Go2ControlArbiter(
        lock_path=lock_path,
        epoch_factory=lambda: 191,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok

    acquired = await bridge.acquire(_permit(config, ttl=0.03))

    assert acquired.code == "GO2_LOW_CMD_PUBLISHER_CLOSE_TIMEOUT"
    assert acquired.data["original_failure_code"] == "GO2_ACQUIRE_PRE_RELEASE_REVALIDATION_FAILED"
    assert acquired.data["owner_lock_retained"] is True
    assert bridge.status().owner_epoch == 191
    assert bridge._publisher is fake.publisher
    assert acquired.data["publisher_close_called"] is False
    assert bridge._publisher_close_task is None
    assert fake.motion.release_calls == 0
    contender = Go2ControlArbiter(
        lock_path=lock_path,
        network_exclusivity_verifier=lambda: True,
    )
    assert contender.acquire_low_level().code == "GO2_LOW_LEVEL_OWNER_LOCKED"

    fake.publisher.close_error = False
    assert (
        await bridge.release(
            _permit(config), "retry final revalidation cleanup", ownership_epoch=191
        )
    ).ok
    assert fake.motion.select_calls == []
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_mode_mismatch_and_close_failure_retains_exact_owner(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    fake.motion.mode_name = "unexpected-mode"
    fake.publisher.close_error = True
    lock_path = tmp_path / "mode-mismatch-close.lock"
    arbiter = Go2ControlArbiter(
        lock_path=lock_path,
        epoch_factory=lambda: 192,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok

    acquired = await bridge.acquire(_permit(config))

    assert acquired.code == "GO2_LOW_CMD_PUBLISHER_CLOSE_FAILED"
    assert acquired.data["original_failure_code"] == "GO2_HIGH_LEVEL_RESTORE_MODE_MISMATCH"
    assert acquired.data["owner_lock_retained"] is True
    assert bridge.status().owner_epoch == 192
    assert bridge._publisher is fake.publisher
    assert fake.motion.release_calls == 0
    contender = Go2ControlArbiter(
        lock_path=lock_path,
        network_exclusivity_verifier=lambda: True,
    )
    assert contender.acquire_low_level().code == "GO2_LOW_LEVEL_OWNER_LOCKED"

    fake.publisher.close_error = False
    fake.motion.mode_name = "normal"
    assert (
        await bridge.release(_permit(config), "retry mismatched-mode cleanup", ownership_epoch=192)
    ).ok
    assert fake.motion.select_calls == []
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_empty_mode_before_release_is_restored_after_close_retry(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    fake.motion.mode_name = ""
    fake.publisher.close_error = True
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "empty-mode-close.lock",
        epoch_factory=lambda: 193,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    acquired = await bridge.acquire(_permit(config))
    assert acquired.code == "GO2_LOW_CMD_PUBLISHER_CLOSE_FAILED"
    assert acquired.data["original_failure_code"] == "GO2_HIGH_LEVEL_RESTORE_MODE_UNKNOWN"
    assert bridge.status().owner_epoch == 193

    fake.publisher.close_error = False
    recovered = await bridge.release(
        _permit(config), "restore empty high-level mode", ownership_epoch=193
    )
    assert recovered.ok
    assert fake.motion.select_calls == ["normal"]
    assert fake.motion.mode_name == "normal"
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_target_ttl_watchdog_freezes_latest_feedback_not_acquisition_pose(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "ttl.lock",
            epoch_factory=lambda: 89,
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    assert config.mapping_hash is not None
    target = _target(config, 1, 0.5)
    assert (await bridge.submit(target, ownership_epoch=89, mapping_hash=config.mapping_hash)).ok
    measured = _LowState(config)
    assert config.zero_offsets_rad is not None
    assert config.directions is not None
    for index in range(12):
        measured.motor_state[index].q = (
            config.zero_offsets_rad[index] + config.directions[index] * 0.03
        )
    fake.subscriber.state = measured
    bridge._on_low_state(measured)
    bridge._writer_cycle(target.valid_until_s + 0.001, expected_deadline=None)
    assert bridge.status().ownership_state is LowCmdOwnershipState.SAFE_HOLD
    assert bridge._safe_hold_q == pytest.approx((0.03,) * 12)
    for _ in range(20):
        bridge._on_low_state(measured)
        if bridge.status().safe_hold_settled:
            break
        await asyncio.sleep(0.005)
    assert bridge.status().safe_hold_settled
    fake.motion.mode_name = "normal"
    assert (await bridge.release(_permit(config), "ttl bench teardown", ownership_epoch=89)).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_writer_rechecks_target_ttl_after_command_construction(tmp_path: Path) -> None:
    """A target expiring during message construction must never reach DDS."""

    config = _low_level_config(low_state_max_age_s=1.0)
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "construct-crosses-ttl.lock",
            epoch_factory=lambda: 890,
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    test_clock = ManualClock(time.monotonic())
    original_clock = bridge._clock
    now = test_clock.monotonic()
    target = Go2JointPositionCommand(
        sequence=1,
        timestamp_s=now - 0.001,
        valid_until_s=now + 0.05,
        joint_positions_rad=(0.5,) * 12,
        desired_contact_forces_world_n=(0.0,) * 12,
    )
    original_make = bridge._make_low_command
    advanced = False

    def make_and_cross_deadline(*args: Any, **kwargs: Any) -> Any:
        nonlocal advanced
        result = original_make(*args, **kwargs)
        if not advanced:
            advanced = True
            test_clock.advance(0.051)
        return result

    try:
        with bridge._guard:
            bridge._clock = test_clock
            bridge._target = target
            bridge._status = replace(
                bridge._status,
                ownership_state=LowCmdOwnershipState.MPC_ACTIVE,
                healthy=True,
                watchdog_healthy=True,
                target_sequence=target.sequence,
                mailbox_staged_target_sequence=target.sequence,
                target_age_s=0.0,
                target_deadline=target.valid_until_s,
            )
            bridge._make_low_command = make_and_cross_deadline  # type: ignore[method-assign]
            writes_before = len(fake.publisher.writes)
            bridge._writer_cycle(test_clock.monotonic(), expected_deadline=None)
            bridge._make_low_command = original_make  # type: ignore[method-assign]
            status = bridge.status()
            assert len(fake.publisher.writes) == writes_before + 1
            assert status.ownership_state is LowCmdOwnershipState.SAFE_HOLD
            assert status.writer_enqueued_target_sequence is None
            assert status.writer_enqueued_q_rad == pytest.approx((0.0,) * 12)
            assert bridge._target is None
    finally:
        bridge._clock = original_clock
        bridge._make_low_command = original_make  # type: ignore[method-assign]
        _force_cleanup_fake_owner(bridge, 890)


@pytest.mark.asyncio
async def test_writer_crossing_target_ttl_during_dds_write_faults_without_target_ack(
    tmp_path: Path,
) -> None:
    """Host detects a late Write, but target-side watchdog is still required."""

    config = _low_level_config(low_state_max_age_s=1.0)
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "write-crosses-ttl.lock",
            epoch_factory=lambda: 891,
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    test_clock = ManualClock(time.monotonic())
    original_clock = bridge._clock
    now = test_clock.monotonic()
    target = Go2JointPositionCommand(
        sequence=1,
        timestamp_s=now - 0.001,
        valid_until_s=now + 0.05,
        joint_positions_rad=(0.5,) * 12,
        desired_contact_forces_world_n=(0.0,) * 12,
    )
    original_write = fake.publisher.Write

    def write_and_cross_deadline(message: _LowCommand, timeout: Any = None) -> bool:
        accepted = original_write(message, timeout)
        test_clock.advance(0.051)
        return accepted

    try:
        with bridge._guard:
            bridge._clock = test_clock
            bridge._target = target
            bridge._status = replace(
                bridge._status,
                ownership_state=LowCmdOwnershipState.MPC_ACTIVE,
                healthy=True,
                watchdog_healthy=True,
                target_sequence=target.sequence,
                mailbox_staged_target_sequence=target.sequence,
                target_age_s=0.0,
                target_deadline=target.valid_until_s,
            )
            fake.publisher.Write = write_and_cross_deadline  # type: ignore[method-assign]
            bridge._writer_cycle(test_clock.monotonic(), expected_deadline=None)
            fake.publisher.Write = original_write  # type: ignore[method-assign]
            status = bridge.status()
            assert status.ownership_state is LowCmdOwnershipState.FAULT
            assert status.writer_enqueued_target_sequence is None
            assert status.writer_enqueue_generation > 0
            assert "DDS Write was in progress" in str(status.fault_reason)
            assert bridge._target is None
    finally:
        bridge._clock = original_clock
        fake.publisher.Write = original_write  # type: ignore[method-assign]
        _force_cleanup_fake_owner(bridge, 891)


@pytest.mark.asyncio
async def test_lowstate_age_fault_keeps_writer_and_owner_lock_visible(tmp_path: Path) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "stale.lock",
        epoch_factory=lambda: 99,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok
    bridge._writer_cycle(time.monotonic() + 2.0, expected_deadline=None)
    status = bridge.status()
    assert status.ownership_state is LowCmdOwnershipState.FAULT
    assert status.safe_hold_active
    assert status.writer_alive
    assert arbiter.status().local_single_instance_held
    assert (await bridge.revoke("fault path", ownership_epoch=99)).ok
    assert bridge.status().ownership_state is LowCmdOwnershipState.FAULT
    bridge._on_low_state(_LowState(config))
    assert bridge.status().safe_hold_settled
    fake.motion.mode_name = "normal"
    assert (
        await bridge.release(_permit(config), "stale-state bench teardown", ownership_epoch=99)
    ).ok
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_abandon_release_mode_transfer(
    tmp_path: Path,
) -> None:
    config = _low_level_config(
        acquire_timeout_s=1.0,
        release_timeout_s=1.0,
        safe_hold_ack_timeout_s=0.8,
    )
    fake = _fake_sdk(config)
    release_entered = threading.Event()
    release_proceed = threading.Event()
    original_release = fake.motion.ReleaseMode

    def blocked_release() -> Tuple[int, None]:
        release_entered.set()
        if not release_proceed.wait(1.0):
            raise RuntimeError("test did not release MotionSwitcher.ReleaseMode")
        return original_release()

    fake.motion.ReleaseMode = blocked_release  # type: ignore[method-assign]
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "double-cancel-acquire.lock",
        epoch_factory=lambda: 1201,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok

    acquire_task = asyncio.create_task(bridge.acquire(_permit(config, ttl=0.9)))
    for _ in range(100):
        if release_entered.is_set():
            break
        await asyncio.sleep(0.005)
    assert release_entered.is_set()
    acquire_task.cancel()
    await asyncio.sleep(0)
    acquire_task.cancel()
    await asyncio.sleep(0)
    assert not acquire_task.done()
    release_proceed.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(acquire_task, 2.0)
    status = bridge.status()
    owner_survived_cancellation = (
        status.owner_epoch == 1201
        and status.writer_alive
        and status.safe_hold_active
        and status.safe_hold_settled
        and arbiter.status().local_single_instance_held
    )
    released = await bridge.release(
        _permit(config, ttl=0.9),
        "double-cancel acquisition teardown",
        ownership_epoch=1201,
    )
    disconnected = await bridge.disconnect()
    assert owner_survived_cancellation
    assert released.ok
    assert disconnected.ok


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_abandon_close_to_select_handoff(
    tmp_path: Path,
) -> None:
    config = _low_level_config(release_timeout_s=1.0, safe_hold_ack_timeout_s=0.8)
    fake = _fake_sdk(config)
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "double-cancel-release.lock",
        epoch_factory=lambda: 1202,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    assert (await bridge.acquire(_permit(config))).ok

    select_entered = threading.Event()
    select_proceed = threading.Event()
    original_select = fake.motion.SelectMode

    def blocked_select(name: str) -> Tuple[int, None]:
        select_entered.set()
        if not select_proceed.wait(1.0):
            raise RuntimeError("test did not release MotionSwitcher.SelectMode")
        return original_select(name)

    fake.motion.SelectMode = blocked_select  # type: ignore[method-assign]
    release_task = asyncio.create_task(
        bridge.release(
            _permit(config, ttl=0.9),
            "double-cancel handoff",
            ownership_epoch=1202,
        )
    )
    for _ in range(100):
        if select_entered.is_set():
            break
        await asyncio.sleep(0.005)
    assert select_entered.is_set()
    assert fake.publisher.closed
    release_task.cancel()
    await asyncio.sleep(0)
    release_task.cancel()
    await asyncio.sleep(0)
    assert not release_task.done()
    select_proceed.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(release_task, 2.0)
    status = bridge.status()
    handoff_completed = (
        status.owner_epoch == 0
        and status.ownership_state is LowCmdOwnershipState.OBSERVE_ONLY
        and not status.ownership_pending
        and not arbiter.status().local_single_instance_held
        and fake.motion.mode_name == "normal"
    )
    disconnected = await bridge.disconnect()
    assert handoff_completed
    assert disconnected.ok


@pytest.mark.asyncio
async def test_old_lowstate_subscription_callback_cannot_mutate_reconnect(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "lowstate-reconnect-generation.lock",
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    old_callback = fake.subscriber.callback
    assert callable(old_callback)

    callback_entered = threading.Event()
    callback_proceed = threading.Event()
    delayed_old_state = _BlockingLowState(
        config,
        callback_entered,
        callback_proceed,
    )
    assert config.zero_offsets_rad is not None
    assert config.directions is not None
    for index in range(12):
        delayed_old_state._motor_state[index].q = (
            config.zero_offsets_rad[index] + config.directions[index] * 0.6
        )
    callback_thread = threading.Thread(
        target=old_callback,
        args=(delayed_old_state,),
        daemon=True,
    )
    callback_thread.start()
    for _ in range(100):
        if callback_entered.is_set():
            break
        await asyncio.sleep(0.005)
    assert callback_entered.is_set()

    assert (await bridge.disconnect()).ok
    assert (await bridge.connect()).ok
    callback_proceed.set()
    callback_thread.join(1.0)
    assert not callback_thread.is_alive()
    status = bridge.status()
    assert len(status.motors) == 12
    assert status.motors[0].q_rad == pytest.approx(0.0)
    assert status.healthy
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_cancelled_lowstate_connect_closes_reader_and_invalidates_callback(
    tmp_path: Path,
) -> None:
    config = _low_level_config()
    fake = _fake_sdk(config)
    init_called = threading.Event()

    def init_without_feedback(callback: Any, queue_depth: int) -> None:
        assert queue_depth == 10
        fake.subscriber.callback = callback
        init_called.set()

    fake.subscriber.Init = init_without_feedback  # type: ignore[method-assign]
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=Go2ControlArbiter(
            lock_path=tmp_path / "cancelled-lowstate-connect.lock",
            network_exclusivity_verifier=lambda: True,
        ),
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    connect_task = asyncio.create_task(bridge.connect())
    for _ in range(100):
        if init_called.is_set():
            break
        await asyncio.sleep(0.005)
    assert init_called.is_set()
    old_callback = fake.subscriber.callback
    assert callable(old_callback)
    connect_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(connect_task, 1.0)

    status_before_old_callback = bridge.status()
    old_callback(_LowState(config))
    status_after_old_callback = bridge.status()
    assert fake.subscriber.closed
    assert not status_before_old_callback.connected
    assert status_before_old_callback.ownership_state is LowCmdOwnershipState.DISCONNECTED
    assert not status_after_old_callback.connected
    assert status_after_old_callback.low_state_timestamp == pytest.approx(
        status_before_old_callback.low_state_timestamp
    )
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_lowstate_fault_during_check_mode_blocks_release_mode(
    tmp_path: Path,
) -> None:
    config = _low_level_config(acquire_timeout_s=1.0)
    fake = _fake_sdk(config)
    check_entered = threading.Event()
    check_proceed = threading.Event()
    original_check = fake.motion.CheckMode
    check_calls = 0

    def blocked_first_check() -> Tuple[int, Any]:
        nonlocal check_calls
        check_calls += 1
        if check_calls == 1:
            check_entered.set()
            if not check_proceed.wait(1.0):
                raise RuntimeError("test did not release MotionSwitcher.CheckMode")
        return original_check()

    fake.motion.CheckMode = blocked_first_check  # type: ignore[method-assign]
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "fault-before-release-mode.lock",
        epoch_factory=lambda: 1203,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    acquire_task = asyncio.create_task(bridge.acquire(_permit(config, ttl=0.9)))
    for _ in range(100):
        if check_entered.is_set():
            break
        await asyncio.sleep(0.005)
    assert check_entered.is_set()
    unsafe = _LowState(config)
    unsafe.motor_state[0].temperature = 100.0
    bridge._on_low_state(unsafe)
    check_proceed.set()
    result = await asyncio.wait_for(acquire_task, 2.0)

    status = bridge.status()
    assert not result.ok
    assert result.code == "GO2_ACQUIRE_PRE_RELEASE_LOCAL_STATE_FAILED"
    assert fake.motion.release_calls == 0
    assert fake.publisher.closed
    assert status.owner_epoch == 0
    assert not status.publisher_active
    assert not arbiter.status().local_single_instance_held
    assert (await bridge.disconnect()).ok


@pytest.mark.asyncio
async def test_capture_current_refreshes_pose_at_release_mode_boundary(
    tmp_path: Path,
) -> None:
    config = _low_level_config(acquire_timeout_s=1.0, release_timeout_s=1.0)
    fake = _fake_sdk(config)
    check_entered = threading.Event()
    check_proceed = threading.Event()
    original_check = fake.motion.CheckMode
    check_calls = 0

    def blocked_first_check() -> Tuple[int, Any]:
        nonlocal check_calls
        check_calls += 1
        if check_calls == 1:
            check_entered.set()
            if not check_proceed.wait(1.0):
                raise RuntimeError("test did not release MotionSwitcher.CheckMode")
        return original_check()

    fake.motion.CheckMode = blocked_first_check  # type: ignore[method-assign]
    arbiter = Go2ControlArbiter(
        lock_path=tmp_path / "refresh-release-mode-pose.lock",
        epoch_factory=lambda: 1204,
        network_exclusivity_verifier=lambda: True,
    )
    bridge = UnitreeGo2LowLevelSdkBridge(
        _go2_config(config),
        arbiter=arbiter,
        allow_hardware_write=True,
        ground_transfer_verifier=_ground_transfer_ok,
        sdk_bindings=fake.bindings,
    )
    assert (await bridge.connect()).ok
    acquire_task = asyncio.create_task(bridge.acquire(_permit(config, ttl=0.9)))
    for _ in range(100):
        if check_entered.is_set():
            break
        await asyncio.sleep(0.005)
    assert check_entered.is_set()

    moved = _LowState(config)
    assert config.zero_offsets_rad is not None
    assert config.directions is not None
    for index in range(12):
        moved.motor_state[index].q = config.zero_offsets_rad[index] + config.directions[index] * 0.2
    fake.subscriber.state = moved
    bridge._on_low_state(moved)
    check_proceed.set()
    acquired = await asyncio.wait_for(acquire_task, 2.0)
    assert acquired.ok

    first_command = fake.publisher.writes[0]
    assert config.motor_ids is not None
    for index, motor_id in enumerate(config.motor_ids):
        expected_sdk_q = config.zero_offsets_rad[index] + config.directions[index] * 0.2
        assert first_command.motor_cmd[motor_id].q == pytest.approx(expected_sdk_q)
    with bridge._guard:
        assert bridge._safe_hold_q == pytest.approx((0.2,) * 12)
        assert bridge._last_commanded_q == pytest.approx((0.2,) * 12)

    released = await bridge.release(
        _permit(config, ttl=0.9),
        "final-pose refresh teardown",
        ownership_epoch=1204,
    )
    assert released.ok
    assert (await bridge.disconnect()).ok
