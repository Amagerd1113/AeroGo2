"""Phase gate for future hardware read-only integration."""


def main() -> int:
    print(
        "NOT_AVAILABLE_IN_PHASE_1: no serial, MAVLink, or Unitree hardware "
        "connection was attempted. F446 read-only support begins in Phase 2."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
