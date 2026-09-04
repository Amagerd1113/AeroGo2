#!/usr/bin/env bash
set -Eeuo pipefail

INSTALL_ROOT=/opt/aerogo2
CONFIG_ROOT=/etc/aerogo2
LOG_ROOT=/var/log/aerogo2
ENABLE_MONITOR=0
INSTALL_PACKAGES=1
TARGET_USER=
ID=
SUDO_USER_VALUE="${SUDO_USER:-}"

usage() {
    echo "Usage: $0 [--target-user USER] [--enable-monitor] [--skip-system-packages]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-user)
            TARGET_USER="$2"
            shift 2
            ;;
        --enable-monitor)
            ENABLE_MONITOR=1
            shift
            ;;
        --skip-system-packages)
            INSTALL_PACKAGES=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

ARCH="$(uname -m)"
case "$ARCH" in
    aarch64|arm64) ;;
    *)
        echo "Refusing installation: expected aarch64/arm64, found $ARCH" >&2
        exit 2
        ;;
esac

if [[ -r /etc/os-release ]]; then
    . /etc/os-release
fi
if [[ "$ID" != "ubuntu" ]]; then
    echo "Refusing installation: this installer supports Ubuntu only (found $ID)" >&2
    exit 2
fi

if [[ -z "$TARGET_USER" ]]; then
    if [[ -n "$SUDO_USER_VALUE" && "$SUDO_USER_VALUE" != "root" ]]; then
        TARGET_USER="$SUDO_USER_VALUE"
    else
        TARGET_USER="$(id -un)"
    fi
fi
if ! id "$TARGET_USER" >/dev/null 2>&1; then
    echo "Target user does not exist: $TARGET_USER" >&2
    exit 2
fi
TARGET_GROUP="$(id -gn "$TARGET_USER")"

if [[ "$EUID" -eq 0 ]]; then
    SUDO=
else
    command -v sudo >/dev/null 2>&1 || {
        echo "sudo is required for system installation" >&2
        exit 2
    }
    SUDO=sudo
fi

run_as_target() {
    if [[ "$(id -un)" == "$TARGET_USER" ]]; then
        "$@"
    elif [[ "$EUID" -eq 0 ]]; then
        runuser -u "$TARGET_USER" -- "$@"
    else
        $SUDO -u "$TARGET_USER" -H "$@"
    fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CYCLONEDDS_SRC="$INSTALL_ROOT/vendor/cyclonedds"
UNITREE_SDK_SRC="$INSTALL_ROOT/vendor/unitree_sdk2_python"
# Keep every generated file outside the hash-pinned source checkouts.  This
# lets a repeated installation distinguish an immutable upstream worktree from
# build products without deleting or silently ignoring unknown files.
BUILD_ROOT="$INSTALL_ROOT/build"
DEPENDENCY_ROOT="$INSTALL_ROOT/dependencies"
CYCLONEDDS_BUILD="$BUILD_ROOT/cyclonedds"
CYCLONEDDS_PREFIX="$DEPENDENCY_ROOT/cyclonedds"
VENV="$INSTALL_ROOT/venv"
# Reproducible candidate revisions.  These are source identities, not a
# hardware-qualification claim; the WCET/API/firmware matrix still has to be
# accepted on the target before LowCmd can be enabled.
CYCLONEDDS_COMMIT=9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd
UNITREE_SDK_COMMIT=65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5
UNITREE_SDK_ARCHIVE="$BUILD_ROOT/unitree_sdk2_python-$UNITREE_SDK_COMMIT.tar"

checkout_pinned_repo() {
    local repository_url="$1"
    local destination="$2"
    local expected_commit="$3"
    local actual_url
    local actual_commit
    local worktree_status
    local submodule_status
    if [[ -d "$destination/.git" ]]; then
        worktree_status="$(
            run_as_target git -C "$destination" status \
                --porcelain=v1 --untracked-files=all
        )"
        if [[ -n "$worktree_status" ]]; then
            echo "Refusing polluted vendor checkout: $destination" >&2
            echo "Tracked, staged and untracked source-tree changes must be removed manually:" >&2
            printf '%s\n' "$worktree_status" >&2
            exit 2
        fi
        actual_url="$(run_as_target git -C "$destination" remote get-url origin)"
        if [[ "$actual_url" != "$repository_url" ]]; then
            echo "Vendor origin mismatch at $destination: $actual_url" >&2
            exit 2
        fi
    elif [[ -e "$destination" ]]; then
        echo "Vendor destination exists but is not a Git checkout: $destination" >&2
        exit 2
    else
        run_as_target git clone --filter=blob:none --no-checkout \
            "$repository_url" "$destination"
    fi
    run_as_target git -C "$destination" fetch --depth 1 origin "$expected_commit"
    run_as_target git -C "$destination" checkout --detach "$expected_commit"
    actual_commit="$(run_as_target git -C "$destination" rev-parse HEAD)"
    if [[ "$actual_commit" != "$expected_commit" ]]; then
        echo "Pinned checkout mismatch at $destination" >&2
        exit 2
    fi
    worktree_status="$(
        run_as_target git -C "$destination" status \
            --porcelain=v1 --untracked-files=all
    )"
    if [[ -n "$worktree_status" ]]; then
        echo "Pinned checkout became polluted at $destination" >&2
        printf '%s\n' "$worktree_status" >&2
        exit 2
    fi
    submodule_status="$(run_as_target git -C "$destination" submodule status --recursive)"
    if [[ -n "$submodule_status" ]]; then
        echo "Refusing vendor repository with unpinned submodule content: $destination" >&2
        printf '%s\n' "$submodule_status" >&2
        exit 2
    fi
}

if [[ "$INSTALL_PACKAGES" -eq 1 ]]; then
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        build-essential cmake git pkg-config
fi

$SUDO install -d -m 0755 -o "$TARGET_USER" -g "$TARGET_GROUP" \
    "$INSTALL_ROOT" "$INSTALL_ROOT/vendor" "$INSTALL_ROOT/bin" \
    "$BUILD_ROOT" "$DEPENDENCY_ROOT" "$CYCLONEDDS_BUILD" "$CYCLONEDDS_PREFIX"
$SUDO install -d -m 0750 -o root -g "$TARGET_GROUP" "$CONFIG_ROOT"
$SUDO install -d -m 0750 -o "$TARGET_USER" -g "$TARGET_GROUP" "$LOG_ROOT"

checkout_pinned_repo \
    https://github.com/eclipse-cyclonedds/cyclonedds.git \
    "$CYCLONEDDS_SRC" "$CYCLONEDDS_COMMIT"
run_as_target cmake -S "$CYCLONEDDS_SRC" -B "$CYCLONEDDS_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_PREFIX" \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DDSPERF=OFF
run_as_target cmake --build "$CYCLONEDDS_BUILD" --parallel
run_as_target cmake --install "$CYCLONEDDS_BUILD"

checkout_pinned_repo \
    https://github.com/unitreerobotics/unitree_sdk2_python.git \
    "$UNITREE_SDK_SRC" "$UNITREE_SDK_COMMIT"
# Install from a git-generated snapshot outside the source checkout.  Running
# pip directly on UNITREE_SDK_SRC may create egg-info/build files there and
# would make the next source-identity check fail (or tempt an unsafe allowlist).
run_as_target git -C "$UNITREE_SDK_SRC" archive --format=tar \
    --output="$UNITREE_SDK_ARCHIVE" "$UNITREE_SDK_COMMIT"

if [[ ! -x "$VENV/bin/python" ]]; then
    run_as_target python3 -m venv "$VENV"
fi
run_as_target "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
run_as_target env CYCLONEDDS_HOME="$CYCLONEDDS_PREFIX" \
    "$VENV/bin/python" -m pip install "$UNITREE_SDK_ARCHIVE"
run_as_target env CYCLONEDDS_HOME="$CYCLONEDDS_PREFIX" \
    "$VENV/bin/python" -m pip install \
    -c "$PROJECT_ROOT/deploy/constraints-aarch64.txt" \
    "${PROJECT_ROOT}[mpc]"
# Import verification catches an absent/incompatible target-architecture
# SciPy wheel before the operator mistakes a partial install for MPC readiness.
run_as_target "$VENV/bin/python" -c \
    'import numpy, scipy; print("AeroGo2 numerical stack:", numpy.__version__, scipy.__version__)'
$SUDO install -m 0755 -o root -g root "$PROJECT_ROOT/scripts/pixhawk_x8_cli_diag.py" \
    "$INSTALL_ROOT/bin/pixhawk_x8_cli_diag.py"

CONFIG_MIGRATION_REQUIRED=0
for source in "$PROJECT_ROOT"/configs/*.yaml; do
    [[ -e "$source" ]] || continue
    name="$(basename "$source")"
    destination="$CONFIG_ROOT/$name"
    if [[ ! -e "$destination" ]]; then
        $SUDO install -m 0640 -o root -g "$TARGET_GROUP" "$source" "$destination"
    elif ! $SUDO cmp -s -- "$source" "$destination"; then
        candidate="$destination.dist"
        $SUDO install -m 0640 -o root -g "$TARGET_GROUP" "$source" "$candidate"
        echo "WARNING: preserving locally edited/older configuration: $destination" >&2
        echo "WARNING: review and merge the new candidate before use: $candidate" >&2
        CONFIG_MIGRATION_REQUIRED=1
    fi
done

# These are immutable model/provenance assets referenced by hashes in the
# configuration, not operator-edited configuration.  Always install the exact
# copies shipped with this project so an old asset cannot masquerade as the
# newly installed release.
for source in "$PROJECT_ROOT"/configs/*.urdf \
              "$PROJECT_ROOT"/configs/UNITREE_ROS_LICENSE.txt; do
    [[ -e "$source" ]] || continue
    name="$(basename "$source")"
    $SUDO install -m 0640 -o root -g "$TARGET_GROUP" "$source" "$CONFIG_ROOT/$name"
done

ENV_FILE="$CONFIG_ROOT/aerogo2.env"
$SUDO tee "$ENV_FILE" >/dev/null <<EOF
CYCLONEDDS_HOME=$CYCLONEDDS_PREFIX
PYTHONUNBUFFERED=1
AEROGO2_X8_DIAG=$INSTALL_ROOT/bin/pixhawk_x8_cli_diag.py
AEROGO2_CYCLONEDDS_COMMIT=$CYCLONEDDS_COMMIT
AEROGO2_UNITREE_SDK_COMMIT=$UNITREE_SDK_COMMIT
EOF
$SUDO chown root:"$TARGET_GROUP" "$ENV_FILE"
$SUDO chmod 0640 "$ENV_FILE"

SERVICE_TMP="$(mktemp)"
trap 'rm -f "$SERVICE_TMP"' EXIT
sed \
    -e "s|@AEROGO2_USER@|$TARGET_USER|g" \
    -e "s|@AEROGO2_GROUP@|$TARGET_GROUP|g" \
    -e "s|@AEROGO2_ROOT@|$INSTALL_ROOT|g" \
    -e "s|@AEROGO2_CONFIG@|$CONFIG_ROOT/hardware.yaml|g" \
    -e "s|@AEROGO2_ENV@|$ENV_FILE|g" \
    -e "s|@AEROGO2_LOG@|$LOG_ROOT|g" \
    "$PROJECT_ROOT/deploy/aerogo2-monitor.service.in" >"$SERVICE_TMP"
$SUDO install -m 0644 -o root -g root \
    "$SERVICE_TMP" /etc/systemd/system/aerogo2-monitor.service
$SUDO usermod -a -G dialout "$TARGET_USER"
$SUDO systemctl daemon-reload

echo
echo "Installation complete. Hardware writes remain locked."
if [[ "$CONFIG_MIGRATION_REQUIRED" -eq 1 ]]; then
    echo "WARNING: one or more existing YAML files were preserved." >&2
    echo "WARNING: merge each corresponding *.dist candidate and validate the configuration before starting AeroGo2." >&2
fi
echo "Edit $CONFIG_ROOT/hardware.yaml and replace both /dev/serial/by-id placeholders."
echo "Log out and back in so the dialout group change takes effect."
echo "Then start the HW-RO shell and run: connect all, preflight, status --full"
echo "  $VENV/bin/aerogo2 shell --hardware-readonly --config $CONFIG_ROOT/hardware.yaml --confirm-hardware I_UNDERSTAND_HARDWARE_RISK"

if [[ "$ENABLE_MONITOR" -eq 1 ]]; then
    echo "Starting the read-only monitor only; no actuator writes are enabled."
    $SUDO systemctl enable --now aerogo2-monitor.service
else
    echo "The read-only monitor was installed but not enabled."
    echo "After preflight: sudo systemctl enable --now aerogo2-monitor.service"
fi
