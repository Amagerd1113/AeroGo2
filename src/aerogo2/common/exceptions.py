"""Domain-specific exceptions for programming/configuration failures."""


class AeroGo2Error(Exception):
    """Base class for AeroGo2 errors."""


class ConfigurationError(AeroGo2Error):
    """Configuration cannot be safely loaded or validated."""


class TransitionRejected(AeroGo2Error):
    """A state transition was rejected by its guard."""


class BridgeError(AeroGo2Error):
    """A device bridge failed."""


class BridgeTimeout(BridgeError):
    """A bridge operation timed out."""


class UnsupportedPhaseOperation(AeroGo2Error):
    """An operation is deliberately unavailable in the current development phase."""


class CommandParseError(AeroGo2Error):
    """A console command could not be parsed."""
