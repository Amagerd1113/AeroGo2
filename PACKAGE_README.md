# AeroGo2

Safety-first supervised control console for the AeroGo2 dual-mode robot.

The package supports dry-run simulation, hardware read-only diagnostics, guarded
F446 morphology control, Unitree Go2 high-level stop/stand integration, Pixhawk
MAVLink telemetry, and an isolated X8 bench-test path.

Hardware writes are disabled by default and require an explicit process unlock,
command confirmation, and live safety interlocks. The console never arms or
disarms Pixhawk and must not be used as the sole flight-safety mechanism.
