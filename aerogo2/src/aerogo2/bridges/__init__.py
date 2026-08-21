"""Hardware-independent bridge contracts and Phase 1 simulators."""

from aerogo2.bridges.f446_interface import F446CommandRecord, F446Interface
from aerogo2.bridges.f446_parser import F446TextParser
from aerogo2.bridges.f446_text_bridge import F446TextBridge, TextF446Bridge
from aerogo2.bridges.fake_f446 import FakeF446
from aerogo2.bridges.fake_go2 import FakeGo2
from aerogo2.bridges.fake_pixhawk import FakePixhawk
from aerogo2.bridges.go2_bridge import Go2Bridge, Go2BridgeStub
from aerogo2.bridges.go2_interface import Go2Interface
from aerogo2.bridges.pixhawk_bridge import PixhawkBridgeStub, ReadOnlyPixhawkBridge
from aerogo2.bridges.pixhawk_interface import PixhawkInterface, VelocitySetpoint
from aerogo2.bridges.rc_monitor import RCMonitor

__all__ = [
    "F446CommandRecord",
    "F446Interface",
    "F446TextBridge",
    "F446TextParser",
    "FakeF446",
    "FakeGo2",
    "FakePixhawk",
    "Go2Bridge",
    "Go2BridgeStub",
    "Go2Interface",
    "PixhawkBridgeStub",
    "PixhawkInterface",
    "RCMonitor",
    "ReadOnlyPixhawkBridge",
    "TextF446Bridge",
    "VelocitySetpoint",
]
