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
CYCLONEDDS_PREFIX="$CYCLONEDDS_SRC/install"
UNITREE_SDK_SRC="$INSTALL_ROOT/vendor/unitree_sdk2_python"
VENV="$INSTALL_ROOT/venv"

if [[ "$INSTALL_PACKAGES" -eq 1 ]]; then
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        build-essential cmake git pkg-config
fi

$SUDO install -d -m 0755 -o "$TARGET_USER" -g "$TARGET_GROUP" \
    "$INSTALL_ROOT" "$INSTALL_ROOT/vendor" "$INSTALL_ROOT/bin"
$SUDO install -d -m 0750 -o root -g "$TARGET_GROUP" "$CONFIG_ROOT"
$SUDO install -d -m 0750 -o "$TARGET_USER" -g "$TARGET_GROUP" "$LOG_ROOT"

if [[ -d "$CYCLONEDDS_SRC/.git" ]]; then
    run_as_target git -C "$CYCLONEDDS_SRC" pull --ff-only
else
    run_as_target git clone --branch releases/0.10.x --depth 1 \
        https://github.com/eclipse-cyclonedds/cyclonedds.git "$CYCLONEDDS_SRC"
fi
run_as_target cmake -S "$CYCLONEDDS_SRC" -B "$CYCLONEDDS_SRC/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_PREFIX" \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DDSPERF=OFF
run_as_target cmake --build "$CYCLONEDDS_SRC/build" --parallel
run_as_target cmake --install "$CYCLONEDDS_SRC/build"

if [[ -d "$UNITREE_SDK_SRC/.git" ]]; then
    run_as_target git -C "$UNITREE_SDK_SRC" pull --ff-only
else
    run_as_target git clone --depth 1 \
        https://github.com/unitreerobotics/unitree_sdk2_python.git "$UNITREE_SDK_SRC"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
    run_as_target python3 -m venv "$VENV"
fi
run_as_target "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
run_as_target env CYCLONEDDS_HOME="$CYCLONEDDS_PREFIX" \
    "$VENV/bin/python" -m pip install "$UNITREE_SDK_SRC"
run_as_target env CYCLONEDDS_HOME="$CYCLONEDDS_PREFIX" \
    "$VENV/bin/python" -m pip install \
    -c "$PROJECT_ROOT/deploy/constraints-aarch64.txt" \
    "$PROJECT_ROOT"
$SUDO install -m 0755 -o root -g root "$PROJECT_ROOT/scripts/pixhawk_x8_cli_diag.py" \
    "$INSTALL_ROOT/bin/pixhawk_x8_cli_diag.py"

for source in "$PROJECT_ROOT"/configs/*.yaml; do
    name="$(basename "$source")"
    destination="$CONFIG_ROOT/$name"
    if [[ ! -e "$destination" ]]; then
        $SUDO install -m 0640 -o root -g "$TARGET_GROUP" "$source" "$destination"
    fi
done

ENV_FILE="$CONFIG_ROOT/aerogo2.env"
$SUDO tee "$ENV_FILE" >/dev/null <<EOF
CYCLONEDDS_HOME=$CYCLONEDDS_PREFIX
PYTHONUNBUFFERED=1
AEROGO2_X8_DIAG=$INSTALL_ROOT/bin/pixhawk_x8_cli_diag.py
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
