"""Local ownership arbitration for Go2 SportClient and ``rt/lowcmd``.

The file lock serializes this host's AeroGo2 Sport RPCs and LowCmd owner. DDS
does not provide an actuator-ownership token, so deployment must separately
exclude phone apps and publishers on every other computer on the robot LAN.
"""

from __future__ import annotations

import math
import os
import secrets
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from aerogo2.common.results import OperationResult


class ControlOwnershipError(RuntimeError):
    """Raised when code attempts a Sport operation during LowCmd ownership."""


class InterProcessOwnerLock:
    """Small cross-platform advisory lock retained for the owner lifetime."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._fd: Optional[int] = None
        self._poisoned = False
        self._guard = threading.Lock()

    @property
    def held(self) -> bool:
        with self._guard:
            return self._fd is not None and not self._poisoned

    @property
    def poisoned(self) -> bool:
        """Whether an OS unlock/close failure made ownership unknowable."""

        with self._guard:
            return self._poisoned

    def acquire(self) -> bool:
        with self._guard:
            if self._fd is not None or self._poisoned:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
            lock_acquired = False
            try:
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                self._lock_fd(fd)
                lock_acquired = True
                # Diagnostic content is explicitly non-authoritative.  Never
                # unlink a lock file: replacing its inode creates a lock race.
                metadata = f" pid={os.getpid()}\n".encode("ascii")
                os.lseek(fd, 1, os.SEEK_SET)
                os.write(fd, metadata)
                os.ftruncate(fd, 1 + len(metadata))
                os.fsync(fd)
                self._fd = fd
                return True
            except Exception:
                try:
                    os.close(fd)
                except Exception:
                    # If locking succeeded, close is the operation expected to
                    # release that OS lock. Losing the descriptor after an
                    # ambiguous close would allow this process to pretend the
                    # host boundary is free. Retain it and poison permanently.
                    self._fd = fd
                    self._poisoned = True
                    raise
                if lock_acquired:
                    # A successful close releases the advisory lock. There is
                    # no active owner, so a later explicit attempt may retry.
                    self._fd = None
                return False

    def release(self) -> None:
        with self._guard:
            fd = self._fd
            if fd is None:
                return
            # Do not close the descriptor in an unlock-exception path. Keeping
            # the handle is the only conservative chance of retaining the OS
            # lock; closing it unconditionally would certainly release any
            # lock that still existed while falsely encouraging a retry owner.
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                self._unlock_fd(fd)
                os.close(fd)
            except Exception:
                # The OS may have applied part of the operation. Neither
                # ``held`` nor a future acquire may claim exclusivity after an
                # ambiguous unlock/close boundary.
                self._poisoned = True
                raise
            else:
                self._fd = None

    @staticmethod
    def _lock_fd(fd: int) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl_api: Any = fcntl
        fcntl_api.flock(fd, fcntl_api.LOCK_EX | fcntl_api.LOCK_NB)

    @staticmethod
    def _unlock_fd(fd: int) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl_api: Any = fcntl
        fcntl_api.flock(fd, fcntl_api.LOCK_UN)


@dataclass(frozen=True)
class Go2ArbiterStatus:
    low_level_epoch: int = 0
    active_sport_operations: int = 0
    local_single_instance_held: bool = False
    local_single_instance_poisoned: bool = False
    network_exclusivity_verified: bool = False
    # The injected verifier is sampled only while acquiring an epoch.  This
    # must not be presented as continuous DDS graph/remote-owner monitoring.
    continuous_network_monitoring_active: bool = False
    network_verifier_name: str = "not_configured"
    network_verification_timestamp_s: float = 0.0
    network_verification_detail: str = "no runtime network-exclusivity verifier configured"


class Go2ControlArbiter:
    """Serialize local Sport calls and grant exactly one LowCmd epoch."""

    _channel_guard = threading.Lock()
    _channel_key: Optional[Tuple[int, str]] = None

    def __init__(
        self,
        *,
        lock_path: Optional[Path] = None,
        domain_id: int = 0,
        network_interface: str = "eth0",
        epoch_factory: Optional[Callable[[], int]] = None,
        network_exclusivity_verifier: Optional[Callable[[], bool]] = None,
        network_exclusivity_verifier_name: Optional[str] = None,
    ) -> None:
        if isinstance(domain_id, bool) or not isinstance(domain_id, int) or domain_id < 0:
            raise ValueError("domain_id must be a nonnegative integer")
        if not isinstance(network_interface, str) or not network_interface.strip():
            raise ValueError("network_interface must be a nonempty string")
        # Linux production services must share the same lock inode with SSH
        # shells even when systemd enables PrivateTmp.  ``/run/lock`` is the
        # host-wide volatile lock namespace. Use one conservative host lock,
        # independent of NIC spelling/domain configuration, so two processes
        # cannot evade mutual exclusion by describing the same robot LAN
        # differently. Multi-robot hosts remain intentionally serialized until
        # a hardware-backed robot identity is commissioned.
        lock_root = (
            Path("/run/lock") if sys.platform.startswith("linux") else Path(tempfile.gettempdir())
        )
        default_path = lock_root / "aerogo2-go2-control.lock"
        self._owner_lock = InterProcessOwnerLock(lock_path or default_path)
        self._epoch_factory = epoch_factory or (lambda: secrets.randbits(62) + 1)
        if network_exclusivity_verifier is not None and not callable(network_exclusivity_verifier):
            raise TypeError("network_exclusivity_verifier must be callable")
        if network_exclusivity_verifier_name is not None and (
            not isinstance(network_exclusivity_verifier_name, str)
            or not network_exclusivity_verifier_name.strip()
        ):
            raise ValueError("network_exclusivity_verifier_name must be nonempty")
        inferred_name = "not_configured"
        if network_exclusivity_verifier is not None:
            inferred_name = str(
                getattr(
                    network_exclusivity_verifier,
                    "__qualname__",
                    type(network_exclusivity_verifier).__name__,
                )
            )
        self._network_exclusivity_verifier = network_exclusivity_verifier
        self._network_verifier_name = network_exclusivity_verifier_name or inferred_name
        self._guard = threading.RLock()
        self._low_level_epoch = 0
        self._sport_operations = 0
        self._network_exclusivity_verified = False
        self._network_verification_timestamp_s = 0.0
        self._network_verification_detail = "no runtime network-exclusivity verifier configured"

    @classmethod
    def initialize_channel_factory(
        cls,
        initializer: Callable[..., Any],
        domain_id: int,
        network_interface: str,
    ) -> OperationResult:
        """Initialize Unitree's process-global channel factory exactly once."""

        key = (domain_id, network_interface)
        with cls._channel_guard:
            if cls._channel_key is not None:
                if cls._channel_key == key:
                    return OperationResult.success(
                        "Unitree ChannelFactory already initialized with the same transport"
                    )
                return OperationResult.failure(
                    "GO2_CHANNEL_FACTORY_CONFLICT",
                    "Unitree ChannelFactory was already initialized with different settings",
                    {"existing": cls._channel_key, "requested": key},
                )
            try:
                initializer(domain_id, network_interface)
            except Exception as exc:
                return OperationResult.failure("GO2_CHANNEL_FACTORY_INIT_FAILED", str(exc))
            cls._channel_key = key
            return OperationResult.success("Unitree ChannelFactory initialized")

    @classmethod
    def _reset_channel_factory_for_tests(cls) -> None:
        """Reset only Python bookkeeping; never call this in a live process."""

        with cls._channel_guard:
            cls._channel_key = None

    def status(self) -> Go2ArbiterStatus:
        with self._guard:
            return Go2ArbiterStatus(
                low_level_epoch=self._low_level_epoch,
                active_sport_operations=self._sport_operations,
                local_single_instance_held=self._owner_lock.held,
                local_single_instance_poisoned=self._owner_lock.poisoned,
                network_exclusivity_verified=self._network_exclusivity_verified,
                continuous_network_monitoring_active=False,
                network_verifier_name=self._network_verifier_name,
                network_verification_timestamp_s=self._network_verification_timestamp_s,
                network_verification_detail=self._network_verification_detail,
            )

    def verify_network_exclusivity(
        self, verification_timeout_s: Optional[float] = None
    ) -> OperationResult:
        """Run the injected network audit; absence or ambiguity rejects ownership.

        A valid verifier must inspect runtime DDS/network state.  Configuration
        flags, operator confirmation, and this process's file lock are not
        acceptable substitutes and are intentionally unavailable here.
        """

        with self._guard:
            if self._owner_lock.poisoned:
                self._invalidate_network_verification(
                    "network proof invalidated because the host owner lock is poisoned"
                )
                return OperationResult.failure(
                    "GO2_CONTROL_ARBITER_POISONED",
                    "The host owner lock is ambiguous; restart this process before any control operation",
                    {"process_restart_required": True},
                )
        verifier = self._network_exclusivity_verifier
        checked_at = time.monotonic()
        if verifier is None:
            detail = "no runtime network-exclusivity verifier configured"
            self._record_network_verification(False, checked_at, detail)
            return OperationResult.failure(
                "GO2_NETWORK_EXCLUSIVITY_UNVERIFIED",
                "LowCmd acquisition requires an injected runtime network-exclusivity verifier",
                self._network_evidence_data(),
            )
        if verification_timeout_s is not None and (
            isinstance(verification_timeout_s, bool)
            or not isinstance(verification_timeout_s, (int, float))
            or not math.isfinite(float(verification_timeout_s))
            or float(verification_timeout_s) <= 0.0
        ):
            detail = "network verification timeout must be finite and positive"
            self._record_network_verification(False, checked_at, detail)
            return OperationResult.failure(
                "GO2_NETWORK_EXCLUSIVITY_UNVERIFIED",
                detail,
                self._network_evidence_data(),
            )

        result_box: Dict[str, Any] = {}
        completed = threading.Event()

        def invoke_verifier() -> None:
            try:
                result_box["result"] = verifier()
            except BaseException as exc:
                result_box["exception"] = exc
            finally:
                completed.set()

        # A verifier is an observation-only hook.  Running it in a daemon
        # helper gives acquisition a hard upper bound even if a platform DDS
        # graph query stalls.  A timed-out result is discarded and never
        # grants ownership if that helper eventually returns.
        verifier_thread = threading.Thread(
            target=invoke_verifier,
            name="aerogo2-go2-network-exclusivity-audit",
            daemon=True,
        )
        verifier_thread.start()
        completed_in_time = completed.wait(verification_timeout_s)
        if not completed_in_time:
            detail = "runtime network verifier exceeded its acquisition deadline"
            self._record_network_verification(False, checked_at, detail)
            return OperationResult.failure(
                "GO2_NETWORK_EXCLUSIVITY_TIMEOUT",
                detail,
                self._network_evidence_data(),
            )
        verifier_exception = result_box.get("exception")
        if isinstance(verifier_exception, BaseException):
            exc = verifier_exception
            detail = f"network verifier raised {type(exc).__name__}: {exc}"
            self._record_network_verification(False, checked_at, detail)
            return OperationResult.failure(
                "GO2_NETWORK_EXCLUSIVITY_UNVERIFIED",
                detail,
                self._network_evidence_data(),
            )
        result = result_box.get("result")
        if type(result) is not bool or not result:
            detail = (
                "network verifier did not return the exact boolean True"
                if type(result) is not bool
                else "network verifier reported another LowCmd publisher or ambiguous state"
            )
            self._record_network_verification(False, checked_at, detail)
            return OperationResult.failure(
                "GO2_NETWORK_EXCLUSIVITY_UNVERIFIED",
                detail,
                self._network_evidence_data(),
            )
        detail = "runtime network verifier reported exclusive LowCmd publication authority"
        self._record_network_verification(True, checked_at, detail)
        return OperationResult.success(
            "Runtime network exclusivity verified",
            self._network_evidence_data(),
            code="GO2_NETWORK_EXCLUSIVITY_VERIFIED",
        )

    def _record_network_verification(
        self, verified: bool, checked_at_s: float, detail: str
    ) -> None:
        with self._guard:
            self._network_exclusivity_verified = verified
            self._network_verification_timestamp_s = checked_at_s
            self._network_verification_detail = detail

    def _invalidate_network_verification(self, detail: str) -> None:
        self._record_network_verification(False, time.monotonic(), detail)

    def _network_evidence_data(self) -> Dict[str, Any]:
        with self._guard:
            return {
                "network_exclusivity_verified": self._network_exclusivity_verified,
                "continuous_network_monitoring_active": False,
                "network_verifier_name": self._network_verifier_name,
                "network_verification_timestamp_s": self._network_verification_timestamp_s,
                "network_verification_detail": self._network_verification_detail,
            }

    def _abort_ungranted_low_level_lock(
        self,
        original_code: str,
        original_message: str,
    ) -> OperationResult:
        """Undo a host-lock acquisition before any epoch was granted."""

        try:
            self._owner_lock.release()
        except Exception as exc:
            self._invalidate_network_verification(
                "network proof invalidated by ambiguous failed-grant lock cleanup"
            )
            return OperationResult.failure(
                "GO2_CONTROL_ARBITER_POISONED",
                (
                    f"{original_message}; host-lock cleanup became ambiguous and "
                    "this process must be restarted "
                    f"({type(exc).__name__}: {exc})"
                ),
                {
                    "original_failure_code": original_code,
                    "local_single_instance_held": self._owner_lock.held,
                    "local_single_instance_poisoned": self._owner_lock.poisoned,
                    "ownership_exclusivity_lost": not self._owner_lock.held,
                    "process_restart_required": True,
                },
            )
        self._invalidate_network_verification(
            "network proof invalidated because the ownership grant was aborted"
        )
        return OperationResult.failure(original_code, original_message)

    def assert_sport_allowed(self) -> OperationResult:
        """Non-racy callers should use ``sport_lease``; this aids diagnostics."""

        with self._guard:
            if self._owner_lock.poisoned:
                return OperationResult.failure(
                    "GO2_CONTROL_ARBITER_POISONED",
                    "SportClient is forbidden after an ambiguous owner-lock release; restart this process",
                    {"process_restart_required": True},
                )
            if self._low_level_epoch != 0:
                return OperationResult.failure(
                    "GO2_LOW_LEVEL_OWNS_CONTROL",
                    "SportClient is forbidden while the LowCmd owner holds authority",
                    {"ownership_epoch": self._low_level_epoch},
                )
            if self._sport_operations:
                return OperationResult.success(
                    "SportClient is allowed under this process's active host lease"
                )
            if not self._owner_lock.acquire():
                return OperationResult.failure(
                    "GO2_HOST_CONTROL_LOCKED",
                    "Another process holds the host-wide Go2 control lock",
                )
            try:
                self._owner_lock.release()
            except Exception as exc:
                self._invalidate_network_verification(
                    "network proof invalidated by an ambiguous Sport probe unlock"
                )
                return OperationResult.failure(
                    "GO2_CONTROL_ARBITER_POISONED",
                    "Sport availability probe poisoned the host control lock; restart this process",
                    {
                        "process_restart_required": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            return OperationResult.success("SportClient is host-wide allowed")

    @contextmanager
    def sport_lease(self) -> Iterator[None]:
        """Hold a local lease around one complete blocking SportClient RPC."""

        with self._guard:
            if self._owner_lock.poisoned:
                raise ControlOwnershipError(
                    "SportClient is forbidden after an ambiguous owner-lock release; restart this process"
                )
            if self._low_level_epoch != 0:
                raise ControlOwnershipError(
                    "SportClient is forbidden while the LowCmd owner holds authority"
                )
            if self._sport_operations == 0 and not self._owner_lock.acquire():
                raise ControlOwnershipError(
                    "SportClient is forbidden because another process holds the host-wide Go2 control lock"
                )
            self._sport_operations += 1
            self._invalidate_network_verification(
                "network proof invalidated by a local SportClient operation"
            )
        try:
            yield
        finally:
            with self._guard:
                self._sport_operations = max(0, self._sport_operations - 1)
                if self._sport_operations == 0:
                    try:
                        self._owner_lock.release()
                    except Exception as exc:
                        self._invalidate_network_verification(
                            "network proof invalidated by an ambiguous Sport lease unlock"
                        )
                        raise ControlOwnershipError(
                            "Sport host-lock release became ambiguous; restart this process"
                        ) from exc

    def acquire_low_level(self, verification_timeout_s: Optional[float] = None) -> OperationResult:
        """Freeze compliant host writers, verify the network, and grant an epoch."""

        with self._guard:
            if self._owner_lock.poisoned:
                return OperationResult.failure(
                    "GO2_CONTROL_ARBITER_POISONED",
                    "LowCmd acquisition is forbidden after an ambiguous owner-lock release; restart this process",
                    {"process_restart_required": True},
                )
            if self._low_level_epoch != 0:
                return OperationResult.failure(
                    "GO2_LOW_LEVEL_ALREADY_OWNED",
                    "This arbiter already granted LowCmd ownership",
                    {"ownership_epoch": self._low_level_epoch},
                )
            if self._sport_operations:
                return OperationResult.failure(
                    "GO2_SPORT_OPERATION_ACTIVE",
                    "Cannot acquire LowCmd during an active SportClient operation",
                )
            # Hold the cross-process lock *before* inspecting the DDS graph.
            # Otherwise a compliant local Sport/LowCmd process could start in
            # the verifier-to-lock gap and make the proof refer to a different
            # network state.  No publisher exists yet at this point.
            if not self._owner_lock.acquire():
                if self._owner_lock.held:
                    return OperationResult.failure(
                        "GO2_LOW_LEVEL_ACQUIRE_BUSY",
                        "This arbiter is already verifying a LowCmd ownership grant",
                    )
                self._invalidate_network_verification(
                    "network proof was not attempted because another local owner holds the file lock"
                )
                return OperationResult.failure(
                    "GO2_LOW_LEVEL_OWNER_LOCKED",
                    "Another local process owns the Go2 control lock",
                )
        network_result = self.verify_network_exclusivity(verification_timeout_s)
        if not network_result.ok:
            with self._guard:
                cleanup = self._abort_ungranted_low_level_lock(
                    network_result.code,
                    network_result.message,
                )
            if cleanup.code == "GO2_CONTROL_ARBITER_POISONED":
                return cleanup
            return OperationResult.failure(
                network_result.code,
                network_result.message,
                dict(network_result.data),
            )
        with self._guard:
            # Recheck after the external verifier returns so no local Sport RPC
            # or inconsistent lock state can race with the ownership grant.
            if self._owner_lock.poisoned:
                self._invalidate_network_verification(
                    "network proof invalidated because the host owner lock became poisoned"
                )
                return OperationResult.failure(
                    "GO2_CONTROL_ARBITER_POISONED",
                    "LowCmd acquisition is forbidden after an ambiguous owner-lock release; restart this process",
                    {"process_restart_required": True},
                )
            if self._low_level_epoch != 0:
                self._invalidate_network_verification(
                    "network proof invalidated because ownership changed during verification"
                )
                return OperationResult.failure(
                    "GO2_LOW_LEVEL_ALREADY_OWNED",
                    "This arbiter already granted LowCmd ownership",
                    {"ownership_epoch": self._low_level_epoch},
                )
            if self._sport_operations:
                self._invalidate_network_verification(
                    "network proof invalidated by a racing SportClient operation"
                )
                return OperationResult.failure(
                    "GO2_SPORT_OPERATION_ACTIVE",
                    "Cannot acquire LowCmd during an active SportClient operation",
                )
            if not self._network_exclusivity_verified:
                return self._abort_ungranted_low_level_lock(
                    "GO2_NETWORK_EXCLUSIVITY_UNVERIFIED",
                    "Network-exclusivity evidence was invalidated before the ownership grant",
                )
            if not self._owner_lock.held:
                self._invalidate_network_verification(
                    "network proof invalidated because the pre-verification host lock was lost"
                )
                return OperationResult.failure(
                    "GO2_CONTROL_ARBITER_POISONED",
                    "The host lock was lost during network verification; restart this process",
                    {"process_restart_required": True},
                )
            try:
                raw_epoch = self._epoch_factory()
            except Exception as exc:
                return self._abort_ungranted_low_level_lock(
                    "GO2_OWNERSHIP_EPOCH_FAILED",
                    f"Ownership epoch generation raised {type(exc).__name__}: {exc}",
                )
            if type(raw_epoch) is not int or raw_epoch <= 0:
                return self._abort_ungranted_low_level_lock(
                    "GO2_INVALID_OWNERSHIP_EPOCH",
                    "Ownership epoch factory must return an exact positive integer",
                )
            epoch = raw_epoch
            self._low_level_epoch = epoch
            return OperationResult.success(
                "Local LowCmd ownership granted",
                {
                    "ownership_epoch": epoch,
                    "local_single_instance_held": True,
                    **self._network_evidence_data(),
                },
            )

    def release_low_level(self, ownership_epoch: int) -> OperationResult:
        """Release only the exact epoch that was granted."""

        with self._guard:
            if type(ownership_epoch) is not int or ownership_epoch <= 0:
                return OperationResult.failure(
                    "GO2_OWNERSHIP_EPOCH_MISMATCH",
                    "Ownership release requires an exact positive integer epoch",
                    {"ownership_epoch": self._low_level_epoch},
                )
            if self._low_level_epoch == 0:
                self._invalidate_network_verification(
                    "network proof invalidated because no LowCmd ownership is held"
                )
                return OperationResult.failure(
                    "GO2_OWNERSHIP_EPOCH_MISMATCH",
                    "No currently held LowCmd ownership matches the supplied epoch",
                    {"ownership_epoch": 0},
                )
            if ownership_epoch != self._low_level_epoch:
                return OperationResult.failure(
                    "GO2_OWNERSHIP_EPOCH_MISMATCH",
                    "Stale code cannot release the current LowCmd owner",
                    {"ownership_epoch": self._low_level_epoch},
                )
            try:
                self._owner_lock.release()
            except Exception as exc:
                self._invalidate_network_verification(
                    "network proof invalidated by an ambiguous host-lock release"
                )
                return OperationResult.failure(
                    "GO2_OWNER_LOCK_RELEASE_FAILED",
                    (
                        "Host owner lock release became ambiguous; quarantine and "
                        "restart this process before any new LowCmd owner is allowed "
                        f"({type(exc).__name__}: {exc})"
                    ),
                    {
                        "ownership_epoch": self._low_level_epoch,
                        "local_single_instance_held": self._owner_lock.held,
                        "local_single_instance_poisoned": self._owner_lock.poisoned,
                        "owner_lock_retained": self._owner_lock.held,
                        "lock_release_ambiguous": True,
                        "ownership_exclusivity_lost": not self._owner_lock.held,
                        "process_restart_required": True,
                    },
                )
            self._low_level_epoch = 0
            self._invalidate_network_verification(
                "network proof invalidated when LowCmd ownership was released"
            )
            return OperationResult.success("Local LowCmd ownership released")


__all__ = [
    "ControlOwnershipError",
    "Go2ArbiterStatus",
    "Go2ControlArbiter",
    "InterProcessOwnerLock",
]
