# AeroGo2 状态转换与激活条件

本文描述 `SystemState`，不是 ArduPilot 的 `STABILIZE`、`LOITER`、`RTL` 等飞行模式。AeroGo2 只读取 Pixhawk 飞行模式；飞行模式切换仍由 RadioMaster/ArduPilot 负责。

## 真机共同前提

除只读命令外，真机控制必须以 `--hardware --enable-hardware-write --confirm-hardware I_UNDERSTAND_HARDWARE_RISK` 启动。所有形态动作还共同要求：

- Pixhawk、F446、Go2、RC 数据均已连接且未过期；RC 与 Pixhawk 均无 failsafe。
- Pixhawk 未 Arm；RC CH5 有效并保持 LOW（不高于 1200 us）。
- 四个已配置 ESC 必须唯一、在线、健康、RPM 为有限值。形态动作开始时必须为 0 RPM；只有 `WALK -> WALK_TO_FLIGHT_PRECHECK` 的第一阶段允许低于 50 RPM，进入下一阶段前仍必须回到精确 0 RPM。
- Go2 的合速度和三个轴速度均小于 0.05 m/s，`stable=true`、`moving=false`、`controller_active=false`，并连续保持 1.0 秒。
- 飞行构型要求关节锁已确认。Go2 原始 `SportModeState.mode=6`，或本机实测的 `mode=0,error_code=1002`，都会产生 `joints_locked=true` 与 `joint_lock_source=telemetry`。其他不回报锁定信号的固件仍可由操作者目视确认 Lock On 后执行 `go2 confirm-lock`；人工来源独立记录，不会改写原始状态码。
- F446 已连接、无 fault、状态不是 UNKNOWN、duty=0；电流值有效，且 `used_current_adc <= threshold_adc - 200`，并连续保持 1.0 秒。
- 没有活动故障；维护模式不能与自动形态转换同时存在。
- 任何转换失败都会先停 F446 和 AeroGo2 自己发送的外部 setpoint，再进入 `FAULT`；AeroGo2 不会自动停 X8、不会自动 Disarm。

## 完整状态矩阵

“允许的下一状态”是守卫图中的完整合法边；`FAULT`/`EMERGENCY_STOP` 是异常边，不代表所有边都有公开 Shell 命令。

| 当前状态 | 允许的下一状态 | 激活/离开要求 |
|---|---|---|
| `BOOT_SAFE` | `MANUAL_POSITIONING`, `HOMING_TO_WALK`, `WALK`, `GO2_JOINT_LOCK_WAIT`, `FLIGHT_READY`, `FAULT` | 每次进程启动固定进入此状态。`connect all` 后，若 F446 权威状态证明 WALK，可采用为 `WALK`；若证明 FLIGHT 且 Go2 已是 mode=6，可进入 `FLIGHT_READY`，否则可写真机进入人工锁关节等待态；UNKNOWN 只能人工定位或 home。 |
| `MANUAL_POSITIONING` | `BOOT_SAFE`, `WALK`, `GO2_JOINT_LOCK_WAIT`, `FLIGHT_READY`, `FAULT`, `EMERGENCY_STOP` | 从 `BOOT_SAFE/WALK/FLIGHT_READY/TOUCHDOWN_VERIFY` 执行 `motor maintenance enter`，通过双重确认、共同前提、Go2 停止与静止/电流保持。落地后仍要求 Pixhawk Disarm、四 ESC 精确 0 RPM 和 CH5 LOW。若命令从 `LANDING_COMPLIANT` 发起，必须先结束柔顺姿态并重新锁关节；真机需完成锁定确认后再次执行进入命令。`mf/mr/limf/limr` 每次发送前重新检查。可以不先运动，但必须先 `ms` 停止，再用 `motor endpoint walk/flight` 标记当前端点，然后用匹配的 `motor confirm walk/flight` 确认；相反限位、非零 duty 或故障仍拒绝。FLIGHT 端点确认后先进入人工锁关节等待态。 |
| `HOMING_TO_WALK` | `WALK`, `FAULT`, `EMERGENCY_STOP` | 仅 `BOOT_SAFE` 且构型 UNKNOWN、F446=IDLE/duty=0 时，执行 `transform home-walk` 并完成双重确认；满足共同前提和保持时间后向 WALK 方向运行自动限位。只有验证 WALK 限位和 duty=0 才进入 `WALK`。 |
| `WALK` | `MANUAL_POSITIONING`, `BOOT_SAFE`, `WALK_TO_FLIGHT_PRECHECK`, `FAULT`, `EMERGENCY_STOP` | F446 必须证明 WALK 限位、duty=0，Pixhawk Disarm、ESC=0、设备新鲜、无故障。Go2 运动命令只在该状态且上述 WALK 互锁保持时可用。 |
| `WALK_TO_FLIGHT_PRECHECK` | `TRANSFORM_TO_FLIGHT`, `WALK`, `FAULT`, `EMERGENCY_STOP` | `transform flight` 只能从 `WALK` 发起；需输入 `TRANSFORM_TO_FLIGHT`，RC CH9 为去抖后的 FLIGHT_REQUEST、CH5 LOW、当前构型仍为 WALK、Go2 静止、F446 电流安全。静止或电流保持不足会回退 `WALK`，不启动机构。 |
| `TRANSFORM_TO_FLIGHT` | `GO2_JOINT_LOCK_WAIT`, `FAULT`, `EMERGENCY_STOP` | 第二次检查全部共同前提，特别是四 ESC 精确 0 RPM；F446 到达 FLIGHT 限位且 duty=0 后再次请求 `StopMove`，随后进入人工锁关节等待态。不会再把 `StandUp()` 当作 mode=6。 |
| `GO2_JOINT_LOCK_WAIT` | `FLIGHT_READY`, `FAULT`, `EMERGENCY_STOP` | F446 已在已验证 FLIGHT 端点且 duty=0。手机端选择 Lock On 后，原始 mode=6 或配置的 `error_code=1002` 都会自动确认。切换产生的运动先经过 2.0 秒初始宽限和 0.5 秒连续越界确认；锁定信号出现但尚未静止时保持等待，稳定后才调用 `SwitchJoystick(false)` 并进入 `FLIGHT_READY`。真正持续进入 `LOCOMOTION`/超速、未知状态码、设备/RC/ESC/F446 异常或 60 秒超时仍进入 `FAULT`。其他固件才使用 `go2 confirm-lock` 人工后备。 |
| `FLIGHT_READY` | `MANUAL_POSITIONING`, `BOOT_SAFE`, `FLIGHT_MANUAL`, `FLIGHT_TO_WALK_PRECHECK`, `FAULT`, `EMERGENCY_STOP` | 构型必须为 FLIGHT、F446 为期望 FLIGHT 限位且 duty=0、关节锁来源为原始 mode 6 或本次等待阶段的守卫式人工确认、Pixhawk Disarm。进入 `FLIGHT_MANUAL` 还必须先执行一次 `flight authorize`，Pixhawk Lua 返回 ACK 后在 30 秒内把 RadioMaster CH5 从 LOW 切到 HIGH，并通过 ArduPilot 全部正常 PreArm 检查。 |
| `FLIGHT_MANUAL` | `AUTO_LANDING_READY`, `TOUCHDOWN_VERIFY`, `FLIGHT_TO_WALK_PRECHECK`, `FAULT`, `EMERGENCY_STOP` | 从 `FLIGHT_READY` 进入时必须同时观察到未过期的一次性 Shell 授权、Go2 `JOINT_LOCK` 和 Pixhawk armed。授权进入后立即消费。进入后必须由新鲜 Pixhawk 遥测连续证明 `armed=true` 且 `landed=false` 达 `safety.airborne_confirm_s`（默认 1.0 秒），才锁存本架次的 `AIRBORNE_CONFIRMED` 并启用触地检测；地面等待不会直接进入 `TOUCHDOWN_VERIFY`。飞行中丢失关节锁触发 `GO2_JOINT_LOCK_LOST`，但不会自动 Disarm。`AUTO_LANDING_READY` 当前仅 DRY-RUN。 |
| `AUTO_LANDING_READY` | `AUTO_LANDING`, `FLIGHT_MANUAL`, `FAULT`, `EMERGENCY_STOP` | 当前只允许 DRY-RUN。要求 Pixhawk armed、CH10 为 AUTO_READY/AUTO_EXECUTE、落地估计有效。任何人工接管/中止返回 `FLIGHT_MANUAL`，且停止外部 setpoint。 |
| `AUTO_LANDING` | `TOUCHDOWN_VERIFY`, `FLIGHT_MANUAL`, `FAULT`, `EMERGENCY_STOP` | 当前只允许 DRY-RUN。要求 CH10=AUTO_EXECUTE、Pixhawk armed 且无 failsafe、RC 新鲜无 failsafe/人工接管、落地估计有效且新鲜、检测到地面、所有数值有限、无活动故障。任一条件丢失立即中止到 `FLIGHT_MANUAL`。 |
| `TOUCHDOWN_VERIFY` | `MANUAL_POSITIONING`, `FLIGHT_MANUAL`, `LANDING_COMPLIANT`, `FLIGHT_TO_WALK_PRECHECK`, `FAULT`, `EMERGENCY_STOP` | 只有本架次已经锁存 `AIRBORNE_CONFIRMED` 才会自动触发。随后要求 Pixhawk `landed=true`，垂直速度绝对值不大于 0.1 m/s，roll/pitch 绝对值不大于 0.2 rad，ESC RPM 不大于 50，参考高度变化不大于 0.02 m，并连续保持 2.0 秒。可用 `touchdown status` 查看离地锁存、各计时器和判据。RadioMaster Disarm、四 ESC 精确 0 RPM、CH5 LOW、Go2 静止及电流保持通过后，可以选择 `transform walk` 自动回 WALK，也可以用 `motor maintenance enter` 进入落地后人工定位后门。启用落地适应时，自动变形仍等待适应流程；人工后门是显式双重确认的恢复路径。 |
| `LANDING_COMPLIANT` | `MANUAL_POSITIONING`, `GO2_JOINT_LOCK_WAIT`, `FLIGHT_READY`, `FAULT`, `EMERGENCY_STOP` | 只在校准功能已启用且上述接触条件连续保持后自动进入；保持 `SwitchJoystick(false)`，调用高层 `BalanceStand()` 并要求权威回报稳定 `BALANCE_STAND`。达到稳定时间后执行 `transform walk` 或 `motor maintenance enter` 都会先进入人工锁关节等待态；手机端完成锁定并回到 `FLIGHT_READY` 后，再次执行相应命令。绝不在 `BALANCE_STAND` 柔顺状态下启动 F446。 |
| `FLIGHT_TO_WALK_PRECHECK` | `TRANSFORM_TO_WALK`, `FLIGHT_READY`, `FAULT`, `EMERGENCY_STOP` | 从 `FLIGHT_READY/FLIGHT_MANUAL/TOUCHDOWN_VERIFY` 请求；若此前进入 `LANDING_COMPLIANT`，必须已事务性重锁并回到 `FLIGHT_READY`。必须 landed、Disarm、四 ESC 精确 0 RPM、Go2 静止、FLIGHT 构型仍受验证、自动落地/setpoint 已停止、共同前提通过。保持不足回退 `FLIGHT_READY`。 |
| `TRANSFORM_TO_WALK` | `WALK`, `FAULT`, `EMERGENCY_STOP` | 必须从上一预检查进入且机构仍证明 FLIGHT；再次检查共同前提后按 WALK 方向运行。只有验证 WALK 限位和 duty=0 才进入 `WALK`。 |
| `FAULT` | `BOOT_SAFE`, `EMERGENCY_STOP` | 通信、互锁、F446、状态入口或未授权 armed 等故障触发。入口停止 F446 和 AeroGo2 setpoint，但不停止旋翼、不 Disarm。`clear-fault` 仅在根因消失、F446 不再 fault、Pixhawk Disarm、ESC=0 后回 `BOOT_SAFE`。 |
| `EMERGENCY_STOP` | `FAULT`, `BOOT_SAFE` | 保留的监督停止状态；入口调用 supervised stop，只停止 AeroGo2 拥有的 F446/Go2/setpoint，不接管 Pixhawk 旋翼。当前没有直接进入它的公开 Shell 命令。恢复到 `BOOT_SAFE` 仍要求故障清除、Disarm、ESC=0。 |

## Go2 飞行关节锁

- Unitree `SportClient` 没有可证明进入 mode=6 的公开 `JointLock()` 调用；`StandUp()` 只代表站立，不能当作锁关节成功。AeroGo2 不发布 `LowCmd`，也不直接设置十二个关节的扭矩、角度或 PD 参数。
- 进入等待态前先 `StopMove`，但保留手机端切换能力。操作者在手机端选择 Lock On；收到原始 mode=6 或配置的 `joint_lock_state_codes: [1002]` 时自动完成。
- `go2.accepted_state_codes` 默认 `[0, 100, 1002]`；`go2.joint_lock_state_codes` 默认 `[1002]`，只把实测的 1002 识别成锁定，不会把普通 100 当成锁定，也不会伪造 mode=6。
- `transform status` 同时显示原始 `go2_joints_locked`、有效 `joint_lock_confirmed`、`joint_lock_source` 和剩余等待秒数；`go2 status` 也保留两套字段，便于审计。
- `FLIGHT_READY` 到向 WALK 变形期间持续要求关节锁确认。原始 mode 6 丢失会触发 `GO2_JOINT_LOCK_LOST`；人工来源若出现行走模式、运动、控制器活动或未接受状态码，会触发 `GO2_OPERATOR_LOCK_UNSAFE`。两者都不会自动停 X8 或自动 Disarm。
- 人工路径的硬件限制：若手机端在保持静止且遥测仍为 mode 0 的情况下解除 Lock On，软件无法从当前 SportModeState 区分，因此无法可靠检测。操作者必须把手机端解锁视为禁止操作，并始终保留 RadioMaster/Pixhawk 接管和断动力能力。
- `request_stop` 在已锁定时保持锁不变；回到 `WALK` 后由 `walk stand` 重新启用 Unitree 原装遥控并进入 `BalanceStand`。`Damp()` 是阻尼模式，不满足飞行互锁。

- 落地适应默认关闭；必须先读取本机 `foot_force[0..3]` 的悬空/站立原始值，为四脚分别设置正阈值后才允许启用。
- 启用后，`TOUCHDOWN_VERIFY` 仍保持 `JOINT_LOCK`；只有 Disarm、四 ESC 精确 0 RPM、至少三脚接触持续 0.5 秒时才调用 `BalanceStand()`。原装遥控在此阶段仍保持禁用。
- `BalanceStand` 稳定保持默认 1.5 秒后，`transform walk` 先进入 `GO2_JOINT_LOCK_WAIT`；手机 Lock On 后由 mode 6 自动确认，或再次执行守卫式 `go2 confirm-lock`，回到 `FLIGHT_READY` 后再执行 `transform walk`。未重新确认绝不启动 F446。`Damp()` 不用于该流程。
- 使用 `landing compliance` 可只读查看四脚原始值、阈值、每脚接触判定、确认时间与稳定时间。

## 两把钥匙 Arm 流程

Pixhawk 必需配置：脚本位于 `/APM/scripts/aerogo2_arm_gate.lua`，`SCR_ENABLE=1`、`RC5_OPTION=153`、`ARMING_RUDDER=0`、`ARMING_CHECK=1`、`ARMING_SKIPCHK=0`（若该参数存在），修改后重启。Lua 会在每次授权时再次验证除 `SCR_ENABLE` 外的这些参数，不匹配即拒绝。

1. Pixhawk 脚本启动即让 AuxAuth 失败，未经授权无法通过正常 Arm 检查。
2. `flight authorize` 在 `FLIGHT_READY` 重新检查四 ESC、Go2、F446、RC、failsafe、构型和活动故障；只有 CH5 LOW 时才发送自定义 MAVLink 授权。
3. Lua 必须返回匹配序号的 `COMMAND_ACK`；无脚本、脚本报错、ACK 超时或拒绝都会使 Shell 命令失败。
4. 授权有效 30 秒；Ubuntu 每 0.4 秒发心跳，Lua 1.5 秒收不到即 fail-closed。
5. 授权后必须把 RadioMaster CH5 从 LOW 切到 HIGH。Lua 只调用普通 `arming:arm()`，不调用 force-arm，因此电池、安全开关、传感器、RC 和所有 ArduPilot PreArm 检查仍然有效。
6. 授权一次性消费；离开 `FLIGHT_READY`、超时、断连、RC 无效或失败都会撤销。Shell 不提供直接 Arm/Disarm API。
7. Lua 阻断 MAVLink Arm（包括 force-arm 绕过），但保留经过正常 API 的 MAVLink Disarm；RadioMaster 的 RC5 ArmDisarm 仍可用于 Disarm。

## 当前真机可用性边界

- WALK/FLIGHT 形态转换、人工 F446 定位、飞行前授权和 armed 状态观察属于真机路径。
- `AUTO_LANDING_READY` 与 `AUTO_LANDING` 仍明确锁在 DRY-RUN；在真实飞行控制和落地传感链完成验证前不会向真机发送自动落地 setpoint。
- ArduPilot 飞行模式不由 AeroGo2 改写。看到 `flight_mode=STABILIZE` 只表示 Pixhawk 当前模式，不是 AeroGo2 `SystemState`。
