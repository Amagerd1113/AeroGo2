# AeroGo2

AeroGo2 当前同时提供 DRY-RUN、HW-RO 和显式解锁的实机模式，已接入
Unitree SDK2、Pixhawk MAVLink 遥测与 F446 限流变形控制。硬件写权限默认关闭，
必须由单次进程参数、命令确认词和实时安全守卫共同打开。

Unitree aarch64 Ubuntu 的安装、接线检查和实机调试见
[aarch64 实机部署手册](docs/HARDWARE_AARCH64_ZH.md)。

F446 �˹� `mr/mf`����� `s`���˹���λȷ�ϡ�ʵʱ HW-039 ���Զ���ת��ֵ�ĵ�ǰ�����������̼�
[F446 �˹���λ���ָ��](docs/F446_MANUAL_POSITIONING_ZH.md)��README �󲿱����� Phase ·��˵�������Ǹ���ʵ��·����

主系统不会解锁 Pixhawk，也不会自动启动 X8 桨电机。

<!-- Encoding-damaged duplicate hidden.

AeroGo2 ?????? DRY-RUN?HW-RO ??????????????
Unitree SDK2?Pixhawk MAVLink ??? F446 ?????????????????
???????????????????????????

Unitree aarch64 Ubuntu ??????????????
[aarch64 ??????](docs/HARDWARE_AARCH64_ZH.md)?

??????? Pixhawk???????? X8 ????
-->

<!-- Legacy Phase 1 introduction retained for historical context.
AeroGo2 是 Unitree Go2 四足平台与 Pixhawk 6X / ArduCopter 飞行平台组合而成的
地空双模机器人。常驻控制台当前仍是 **Phase 1：纯软件模拟、状态机、安全互锁、
交互式终端和自动化测试**，不会连接、解锁或驱动真实硬件。仓库另提供与常驻状态机
隔离的 `x8-bench` 入口；它直接复用根目录中已验证的 `pixhawk_x8_cli_diag.py`，仅用于
拆除桨叶、限流供电并固定机体后的 Pixhawk/Hobbywing X8 台架诊断。
-->

## 安全警告

> **This software is initially provided for simulation and bench testing only.**
>
> **Remove all propellers before hardware integration testing.**
>
> **The console does not arm or disarm Pixhawk.**
>
> **The console must never be used as the sole flight safety mechanism.**
>
> **The RadioMaster-to-Pixhawk control link must remain independent.**
>
> **Manual F446 motor commands have no automatic limit protection.**
>
> **Current-based limit detection does not prove that the mechanical lock is engaged.**
>
> **A separate mechanical lock or position sensor is strongly recommended before flight.**

<!-- Legacy Phase 1 hardware gate note.
Phase 1 常驻控制台的硬件写使能仍被配置校验器强制为 `false`。真实 Pixhawk、F446
和 Go2 Bridge 都显式拒绝连接/写入；只有 `FakePixhawk`、`FakeF446`、`FakeGo2`
可以进入 SystemManager。`x8-bench` 不进入 SystemManager，不改变飞行状态机或
`hardware_write_enabled`，而是在独立进程中启动同一个已验证诊断文件。
-->

<!-- Encoding-damaged duplicate hidden.
????????????? false???
--hardware --enable-hardware-write --confirm-hardware I_UNDERSTAND_HARDWARE_RISK
???????????? F446/Go2 ???HW-RO ????????
-->

磁盘配置中的硬件写使能保持 false。只有
--hardware --enable-hardware-write --confirm-hardware I_UNDERSTAND_HARDWARE_RISK
组合能为当前进程临时解锁 F446/Go2 控制；HW-RO 不能绕过该门控。



## 硬件架构与控制权

```text
地面运动
Go2 原装遥控器 ───────────────> Go2 原生高层运动控制

独立飞行控制链（Ubuntu 不得截断）
RadioMaster TX16S
  -> RP4TD ExpressLRS
  -> CRSF UART
  -> Pixhawk 6X / ArduCopter
  -> DroneCAN
  -> 4 x Hobbywing X8 G2 ESC

构型执行链
Go2 Ubuntu / AeroGo2 SystemManager
  -> 高层文本请求（后续阶段）
  -> STM32 NUCLEO-F446RE
  -> HW-039 / BTS7960
  -> WGM4632-370 减速电机
  -> 机械变形机构
```

控制权严格分工：

- Go2 原装遥控器只负责 `WALK` 构型中的站立、步行、转向和停止。切换前必须由
  人工确认原装遥控器已停止使用；第一版不声称 Ubuntu 能屏蔽它。
- RadioMaster 负责 Roll、Pitch、Throttle、Yaw、飞行模式、RTL、Land、
  `FLIGHT_ENABLE`、构型请求、自动降落请求和人工接管。
- Pixhawk 负责飞行姿态/位置内环、Quad X 混控、DroneCAN ESC、RC 输入及
  ArduPilot failsafe。AeroGo2 常驻控制台不提供 arm、disarm、takeoff、motor-test
  或 raw throttle；只有明确启动的独立 `x8-bench` 诊断进程具有 `cli_diag` 原有命令。
- F446 负责变形电机的本地实时控制、电流限位、机械限位、PWM/使能和本地
  FAULT。Ubuntu 永不直接生成 PWM。
- Go2 Ubuntu 上的 AeroGo2 只做状态编排、互锁、状态读取、RC 解析、模拟
  自动降落、日志和终端。

## 软件架构

```text
InteractiveShell
  -> CommandParser
  -> CommandDispatcher
  -> ConfirmationService
  -> CommandService / SystemManager
  -> StateMachine + TransitionGuards + SafetyMonitor
  -> Bridge interface
  -> Fake device (Phase 1 only)
```

主要目录：

- `common/`：枚举、不可变快照、结果类型、单调时钟和 YAML 配置。
- `manager/`：唯一状态写入口、转换守卫、权限、命令服务和系统编排。
- `bridges/`：设备抽象、Phase 1 Fake、F446 增量文本解析器，以及明确拒绝
  真实访问的后续阶段边界。
- `safety/`：纯评估式 SafetyMonitor、互锁、看门狗和故障记录。
- `landing/`：安全下降控制骨架、估计器、轨迹和限幅器；输出只进入
  FakePixhawk。
- `cli/`：常驻异步 Shell、分层命令注册、补全、历史和确认。
- `logging/`：JSONL 事件与遥测日志。
- `simulation/`：名义任务、故障注入和可复位仿真世界。
- `tests/`：解析器、桥、状态机、CLI 和端到端任务测试。

Shell 不持有 Bridge 引用。即使是 `motor to-flight`，也只能通过
`SystemManager`，再由状态机守卫决定是否可以触达 Fake Bridge。

## 状态机

每个实例都从 `BOOT_SAFE` 启动，不从日志或磁盘恢复上一次运动命令。所有状态
改变只能通过：

```python
await state_machine.transition_to(
    new_state,
    reason=reason,
    snapshot=snapshot,
)
```

转换会依次执行：守卫检查、生成 `TransitionRecord`、JSONL 记录、唯一状态更新、
发布事件、进入动作，以及进入动作失败时转入 `FAULT`。

完整的 15 个状态、所有合法下一状态、每条激活条件和真机/DRY-RUN 边界见
[`docs/STATE_TRANSITIONS_ZH.md`](docs/STATE_TRANSITIONS_ZH.md)。`FLIGHT_READY -> FLIGHT_MANUAL`
采用一次性两把钥匙：AeroGo2 Shell `flight authorize` 成功后，30 秒内再由 RadioMaster CH5 LOW->HIGH 请求正常 Arm。

规格未定义的恢复边（例如从飞行中的任意 FAULT 直接变形）不会被推测性开放。

## 持续安全不变量

1. 未确认 FLIGHT 构型时，不允许进入飞行准备。
2. Pixhawk armed 时，F446 不得运动。
3. 任何非零 ESC RPM 都拒绝新的变形动作；这比报警阈值更保守。
4. 未确认 WALK 构型时，不允许步行许可。
5. 变形中 Go2 必须静止，Pixhawk 不得进入本软件控制的解锁流程。
6. F446 状态过期时禁止新的变形动作。
7. Pixhawk 状态过期时禁止新的变形动作。
8. RC failsafe 立即把 CH5/CH9/CH10 高层请求恢复为安全值。
9. 自动降落失效会停止外部 setpoint，但绝不自动停旋翼或 disarm。
10. RadioMaster 人工接管优先于自动降落。
11. 启动固定进入 `BOOT_SAFE`，不恢复运动。
12. 构型未知时同时禁止飞行许可和步行许可。
13. F446 手动命令在普通模式不可用；Phase 1 中维护模式本身也不可用。
14. 终端不能绕过 `SystemManager` 直调硬件运动接口。
15. Ubuntu 不实现飞行中电机急停。

`SafetyMonitor.evaluate(snapshot)` 是纯函数：它只返回违规项，由
`SystemManager` 决定停止 setpoint、停止变形或进入 `FAULT`，Monitor 本身不
调用任何 Bridge。

## F446 文本协议

现有 F446 固件在 Phase 1 不作修改。USART2/NUCLEO USB VCP 参数为
115200、8 data bits、no parity、1 stop bit、no flow control。

固件状态：

```text
IDLE
MANUAL_FWD
MANUAL_REV
LIMIT_FWD
LIMIT_REV
LIMIT_REACHED_FWD
LIMIT_REACHED_REV
FAULT
```

固件命令：

```text
help                status              is
mf DUTY             mr DUTY             raw SIGNED_DUTY
mlimit LIMIT        limf DUTY           limr DUTY
sense max|r|l       thr ADC             thrmv MILLIVOLT
blank MS            overms MS           timeout MS
auto on|off         stop                disable
clear
```

正式构型切换将来只允许 `limf`/`limr`、`stop`、`status`、`is`，方向由
`configs/f446.yaml` 映射，绝不硬编码 “forward=FLIGHT”。`mf`、`mr`、
`raw` 无自动限位，必须等 Phase 3 维护模式和双重确认。`clear` 绝不自动调用。

Phase 1 的增量 parser 支持 CRLF/LF、半行分包、多行粘包、命令回显、自动状态
交错、未知行和有界缓冲；所有运行中的 F446 动作都发生在内存 Fake。

## 安装

推荐 Python 3.8 或更高版本。Linux / Go2 Ubuntu：

```bash
cd aerogo2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Windows PowerShell：

```powershell
Set-Location aerogo2
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

依赖中保留 `pymavlink`、`pyserial`、`pyserial-asyncio`。常驻 Phase 1 模块不会在
import 或启动时连接硬件；只有显式运行 `x8-bench` 才会导入并启动诊断文件。
`--can-probe`、`--can-config-probe`、`--can-node-info` 和 `--set-can-throttle` 还需要
按 `cli_diag` 提示单独安装 `dronecan`，普通 MAVLink 电机测试不依赖它。

## Dry-run

启动交互终端：

```bash
aerogo2 shell --dry-run --config configs/system.yaml
```

运行非交互名义演示：

```bash
python scripts/run_dry.py --scenario nominal
```

预期初始界面：

```text
AeroGo2 Integrated Control Console
==================================
Runtime       : DRY-RUN
System state  : BOOT_SAFE
Pixhawk       : DISCONNECTED
F446          : DISCONNECTED
Go2           : DISCONNECTED
Logging       : ON

Type "help" to list commands.

aerogo2[DRY-RUN|BOOT_SAFE]>
```

`--hardware-readonly` 会连接 Unitree SDK2、F446 和 Pixhawk，只读取状态并执行
`preflight`；权限层和桥接器层都会拒绝执行器写入。真实控制还必须同时使用
`--hardware --enable-hardware-write --confirm-hardware I_UNDERSTAND_HARDWARE_RISK`。

## X8 台架诊断

先拆除全部桨叶、使用限流电源、固定机体，并关闭 Mission Planner、QGC、MAVProxy
或其他占用同一串口的程序。AeroGo2 会强制使用 `configs/serial.yaml` 中的端口、
115200 波特率和 30 秒心跳等待，不允许通过透传参数临时覆盖这三项。

只检查 AeroGo2、配置和 `cli_diag` 是否对齐，不打开串口：

```bash
aerogo2 x8-bench --check --config configs/system.yaml
```

通过 AeroGo2 运行 `cli_diag` 的无硬件自测或端口枚举：

```bash
aerogo2 x8-bench -- --self-test
aerogo2 x8-bench -- --list-ports
```

只读检查真实 Pixhawk 参数和 X8 就绪状态：

```bash
aerogo2 x8-bench -- --commands "streams on; audit std; x8diag 3"
```

进入原始交互诊断终端：

```bash
aerogo2 x8-bench
```

推荐先用受限入口完成单臂低油门、短时间测试：

```bash
aerogo2 x8-spin \
  --config configs/hardware.yaml \
  --target lf \
  --percent 10 \
  --duration 2 \
  --props-removed \
  --airframe-secured \
  --confirm-x8 X8_PROPS_REMOVED_AND_AIRFRAME_SECURED
```

`x8-spin` 只允许 5–20% 和 0.5–5 秒，并要求 Pixhawk 明确为 DISARMED、标准参数
匹配、ESC 遥测新鲜。它依次完成 `safety off`、`triggercheck`、`triggerwin`、
`trigger off` 和 `safety on`，异常退出时也会再次停转并恢复安全开关。首次测试不要
使用 `all`；四个单臂方向和映射验收通过后再测试全部 X8。

## 终端命令

命令支持引号参数、别名、补全、历史、拼写建议、Rich 表格、`--watch` 和精确
确认。确认短语使用独立无历史的 prompt，永不写入历史文件。

```text
通用
  help [COMMAND] | version | clear | history | exit | quit

设备
  devices
  connect all|pixhawk|f446|go2
  disconnect all|pixhawk|f446|go2
  health [--watch SECONDS]

地面行走
  walk status|permit|stop|stand

状态与监控
  status [--full|--json|--watch SECONDS]
  state | state transitions | state guards
  watch status|rc|f446|esc|faults
  rc | rc raw | rc mapping | rc check
  pixhawk status|messages|statustext|params
  go2 status|motion|controller
  esc [1|2|3|4] | esc mapping | esc health

构型与检查
  transform status|flight|walk|stop
  audit [pixhawk|f446|rc|configuration]
  preflight [flight|transform-flight|transform-walk|autoland]
  check invariant|communication|sensors
  flight status|enable-check|ready

F446
  motor status|current|parameters
  motor auto-status on|off
  motor to-flight|to-walk|stop|disable|clear-fault
  motor maintenance enter|exit
  motor mf|mr|raw|limf|limr DUTY
  motor threshold|threshold-mv|blank|overms|timeout VALUE
  motor sense max|r|l

自动降落
  autoland status|prepare|start|abort
  abort
  controller status|timing|inputs|output|reset

故障、配置与日志
  faults [active|history|explain CODE]
  clear-fault | stop
  config show|get KEY|validate|reload|diff
  log status|start|stop|mark TEXT|tail|export PATH

仿真
  sim status|reset|run|pause|step|clear
  sim scenario nominal|transform-failure|rc-loss|pixhawk-timeout|f446-overcurrent|landing
  sim inject FAULT
```

Phase 1 常驻 Shell 注册了 130 条 canonical command 和 2 条 alias。完整命令树始终可见，
但属于 Phase 2–8 的动作统一返回 `PHASE_NOT_AVAILABLE`，包括真实设备连接、所有真实
F446 写命令、arm/disarm、motor-test 和真实 setpoint。顶层 `x8-bench` 是隔离进程入口，
不注册进常驻 Shell，也不能绕过 SystemManager 的 Phase 1 边界。

危险动作确认：

| 命令 | 等级 | 精确文本 |
|---|---|---|
| `transform flight` | `EXACT_PHRASE` | `TRANSFORM_TO_FLIGHT` |
| `transform walk` | `EXACT_PHRASE` | `TRANSFORM_TO_WALK` |
| `motor maintenance enter` | `EXACT_PHRASE` | `ENTER_F446_MAINTENANCE` |
| `motor mf/mr/raw` | `TWO_STAGE` | `RUN_MANUAL_MOTOR` |
| armed 时退出 | `EXACT_PHRASE` | `EXIT_WHILE_ARMED` |

## `stop` 和 Ctrl+C

`stop` 只执行：

- 停止 F446 变形；
- 请求 Go2 停止；
- 停止自动降落外部 setpoint。

Pixhawk armed 时输出：

```text
Pixhawk is armed.
Rotor shutdown is not performed by this console.
Automatic setpoints have been stopped.
Use RadioMaster to control or land the vehicle.
```

它不发送 disarm，不停止旋翼。普通输入时 Ctrl+C 清当前行；watch 中退出 watch；
自动降落中询问是否 abort；变形中停止 F446 并进入 `FAULT`。Ctrl+D 正常退出
Shell；armed 时必须精确确认。

## 配置

入口是 `configs/system.yaml`，其 `includes` 依次合并：

- `serial.yaml`
- `rc_channels.yaml`
- `safety_limits.yaml`
- `f446.yaml`
- `landing.yaml`

后加载值覆盖先加载值，include 循环、缺少章节、RC 通道冲突、阈值重叠、
F446 方向/期望状态冲突、ESC 物理映射冲突、非正超时，以及 Phase 1
`hardware_write_enabled=true` 都会在启动前失败。

X8 映射按已验证 NodeID/ThrottleID/output 固定，任何换位都会使配置加载失败：

| 输出槽位 | AeroGo2 标签 | 物理位置 | ArduCopter motor-test | SERVO function |
|---:|---|---|---:|---:|
| 1 | `RR` | 右后 | M2 | 36 |
| 2 | `LF` | 左前 | M4 | 35 |
| 3 | `LR` | 左后 | M3 | 34 |
| 4 | `RF` | 右前 | M1 | 33 |

如果 Linux 设备名不是 `/dev/ttyACM0`，只修改 `configs/serial.yaml` 的
`pixhawk.connection`，优先使用真实的 `/dev/serial/by-id/...` 稳定路径；不要通过
`x8-bench` 透传 `--port`、`--baud` 或 `--connect-timeout`。

若机械方向相反，只修改：

```yaml
f446:
  flight_direction: "reverse"
  walk_direction: "forward"
  expected_flight_state: "LIMIT_REACHED_REV"
  expected_walk_state: "LIMIT_REACHED_FWD"
```

不要修改代码。RC CH9/CH10 不得绑定 Pixhawk 辅助功能：

```yaml
pixhawk:
  rc9_option: 0
  rc10_option: 0
```

`config get safety.f446_timeout_s` 可读取点路径；Phase 1 不提供在线
`config set`。

## 日志

默认目录为 `./logs`（相对项目根目录解析）。事件采用 JSONL，每条记录至少
保留以下 schema 字段；缺失值写 `null`，不会改变字段名：

```text
wall_timestamp             monotonic_timestamp
event_type                 system_state
previous_state             command_id
command_name               command_result
pixhawk_status             f446_status
go2_status                 operator_request
safety_violations          transition_reason
landing_command
```

自动事件包括启动、连接、两向变形、限位确认、FLIGHT_READY、观察到 armed、
自动降落准备/启动/接管、触地确认、观察到 disarmed、故障进入/清除和退出。
日志只用于审计，永不用于启动状态恢复。

## 测试与质量检查

```bash
ruff format --check .
ruff check .
mypy src/aerogo2
pytest -q
```

当前交付基线（2026-07-27）：Ruff format/check 通过，strict mypy 通过，
完整测试套件 `595 passed`；六个规定 dry-run 场景均通过。

测试覆盖：

- F446 完整 status/current/limit/fault 解析、CRLF/LF、分包、粘包、回显、
  auto status 和未知行。
- 方向映射、最终 status/duty 二次确认、timeout 后 stop、FAULT 失败，以及
  绝不自动 `clear`/绝不在正式路径调用 `mf`/`mr`/`raw`。
- `BOOT_SAFE -> WALK -> FLIGHT_READY -> FLIGHT_MANUAL -> AUTO_LANDING_READY
  -> AUTO_LANDING -> TOUCHDOWN_VERIFY -> LANDING_COMPLIANT -> FLIGHT_READY
  -> FLIGHT_TO_WALK_PRECHECK -> TRANSFORM_TO_WALK -> WALK`；未启用已校准落地适应时跳过 `LANDING_COMPLIANT`。
- armed、RPM、Go2 移动、CH5、F446 fault/timeout、未知构型、非法转换和
  控制器超时。
- 命令解析、help、补全、拼写建议、确认、维护模式拒绝、Ctrl+C 及后台任务
  清理。
- 过流、RC 丢失、Pixhawk 超时、人工接管和变形超时仿真。

## 故障处理

1. 使用 `status --full`、`faults active` 和 `health` 确认来源。
2. 如果正在飞行，使用 RadioMaster 控制或降落；不要期待 Console 停旋翼。
3. F446 变形失败时，Console 发送 `stop` 到 FakeF446 并进入 `FAULT`；真实
   F446 写路径尚未开放。
4. `clear-fault` 只清除已消失且由操作员确认的 Manager 故障，不发送 F446
   `clear`。
5. 构型为 `UNKNOWN` 时同时禁止步行和飞行准备；通过独立检查确认机械位置。
6. 重启始终回到 `BOOT_SAFE`，必须重新读取并验证所有状态。

## 开发阶段

- **Phase 1（当前）**：配置、模型、状态机、安全、Fake 设备、RC 模拟、
  CLI、JSONL 日志、场景和 pytest；另有不进入 SystemManager 的 X8 台架诊断入口。
- **Phase 2**：经代码审查后加入真实 F446 只读 `status`、`is`、
  `auto on/off`；仍不发送运动命令。
- **Phase 3**：台架拆桨条件下加入 F446 维护模式、双重确认和手动命令。
- **Phase 4**：真实 F446 构型切换；前提是 Pixhawk/ESC/Go2 真实只读状态和
  完整 preflight。
- **Phase 5**：真实 Pixhawk MAVLink 状态读取；仍不 arm/disarm。
- **Phase 6**：Unitree SDK 高层状态与 stop；不接低层关节电流。
- **Phase 7**：只在 FakePixhawk 中测试自动降落 setpoint。
- **Phase 8**：独立代码审查与硬件测试后，才考虑真实高层 setpoint。

## 已知安全限制

- 电流限位只能表示堵转/负载变化，不能证明机械锁已啮合。
- 没有独立位置传感器时，F446 `LIMIT_REACHED_*` 仍是间接证据。
- Ubuntu 不能被假定为能完全屏蔽 Go2 原装遥控器。
- ESC 遥测缺失不等于 RPM=0；真实集成时必须 fail closed。
- Phase 1 的 RC 人工接管以 CH10 回到 MANUAL 为确定信号。CH1–CH4 摇杆
  deadband 需要在真实 RC 基线/油门语义明确后再接入，不能硬编码假设。
- 自动降落坐标采用 NED（正 `vz` 向下）仅用于 Fake；真实坐标/模式必须在
  Phase 8 单独审查。
- `EMERGENCY_STOP` 不是飞行电机急停；它仍然不会 disarm 或切断旋翼。
- 本软件不能替代飞控 failsafe、独立急停设计、机械锁、位置传感器、地面
  测试程序或合格飞手。

