"""Status, RC, Pixhawk, Go2, and ESC monitoring commands."""

from typing import Tuple

from aerogo2.cli.command_models import CommandSpec
from aerogo2.cli.commands._helpers import command, readonly


def command_specs() -> Tuple[CommandSpec, ...]:
    definitions = (
        ("state", "Show the current system state", "query_state"),
        ("state transitions", "Show transition records", "query_transitions"),
        ("state guards", "Show current guard checks", "query_guards"),
        ("watch status", "Continuously show status", "watch_status"),
        ("watch rc", "Continuously show parsed RC channels", "watch_rc"),
        ("watch f446", "Continuously show F446 state", "watch_f446"),
        ("watch esc", "Continuously show ESC telemetry", "watch_esc"),
        ("watch faults", "Continuously show active faults", "watch_faults"),
        ("rc", "Show parsed RC requests", "query_rc"),
        ("rc raw", "Show raw RC channel values", "query_rc_raw"),
        ("rc mapping", "Show configured RC mapping", "query_rc_mapping"),
        ("rc check", "Audit RC values and debounce state", "query_rc_check"),
        ("pixhawk status", "Show Pixhawk telemetry", "query_pixhawk"),
        ("pixhawk messages", "Show received simulated MAVLink messages", "query_pixhawk"),
        ("pixhawk statustext", "Show Pixhawk STATUSTEXT", "query_pixhawk"),
        ("pixhawk params", "Show safety-relevant configured parameters", "query_pixhawk"),
        ("go2 status", "Show Go2 telemetry", "query_go2"),
        ("go2 motion", "Show Go2 motion/stability state", "query_go2"),
        ("go2 controller", "Show Go2 controller ownership", "query_go2"),
        ("esc", "Show all ESC telemetry", "query_esc"),
        ("esc 1", "Show ESC slot 1", "query_esc"),
        ("esc 2", "Show ESC slot 2", "query_esc"),
        ("esc 3", "Show ESC slot 3", "query_esc"),
        ("esc 4", "Show ESC slot 4", "query_esc"),
        ("esc mapping", "Show physical ESC slot mapping", "query_esc_mapping"),
        ("esc health", "Evaluate ESC telemetry health", "query_esc"),
    )
    return (
        command(
            "status",
            "Show the integrated snapshot",
            "monitoring",
            "query_status",
            usage="status [--full|--json|--watch SECONDS]",
            options=("--full", "--json", "--watch"),
        ),
    ) + tuple(
        readonly(
            path,
            description,
            "monitoring",
            action,
            usage=f"{path} [SECONDS]" if path.startswith("watch ") else "",
        )
        for path, description, action in definitions
    )
