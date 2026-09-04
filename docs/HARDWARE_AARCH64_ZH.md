# AeroGo2 aarch64 Ubuntu 实机部署与调试

> F446 �˹� `mr/mf`����� `s`���˹���λȷ�Ϻ� HW-039 �Զ���ת�ĵ�ǰ���̣��� [F446_MANUAL_POSITIONING_ZH.md](F446_MANUAL_POSITIONING_ZH.md) Ϊ׼��

本文用于 Unitree 机载 Ubuntu aarch64。实机链路包括：

- Unitree SDK2：订阅 Go2 SportModeState，只调用高层 StopMove、BalanceStand 和 SwitchJoystick；手机端人工选择飞行锁定。`mode=6` 或本机实测的 `mode=0,error_code=1002` 都会自动确认；其他固件仍保留带完整互锁和精确口令的人工确认。
- F446：异步串口读取文本协议，只用受过流和超时保护的 limf、limr；到位后再次核对限位状态和 duty=0。
- Pixhawk：读取心跳、解锁、落地、RC、姿态和 4 路 ESC 遥测。
- SystemManager：在 WALK 与 FLIGHT_READY 之间执行有安全守卫的形态切换。

主系统不会解锁 Pixhawk，不会发送 motor-test，也不会自动启动 X8 桨电机。变形电机和桨电机必须分开验证。

## 1. 首次联调硬条件

1. 拆掉全部桨叶，架空并可靠固定机体。
2. F446 使用限流电源，急停和总电源可由另一人直接操作。
3. 正反机械限位可承受惯性；建议增加独立位置传感器或机械锁。
4. Unitree 原装遥控器停止发运动指令；RadioMaster 到 Pixhawk 链路保持独立。
5. Pixhawk 未解锁，4 路 ESC 遥测全部在线且 RPM 精确为 0。
6. 单独确认 F446 固件方向与 configs/f446.yaml 一致。

软件通过不等于适航。载荷、重心、电磁兼容、失联、过流、卡滞和单点故障仍需逐项实测。

## 2. 安装

~~~bash
cd aerogo2
bash deploy/install_aarch64.sh --target-user unitree
~~~

脚本验证 aarch64/Ubuntu，在 /opt/aerogo2 构建 CycloneDDS，安装官方 unitree_sdk2_python、AeroGo2 虚拟环境、配置和 HW-RO systemd 服务。默认不启动服务，也不打开硬件写权限。

用户不是 unitree 时替换参数。安装后退出 SSH 并重新登录，使 dialout 组生效。升级不会覆盖 /etc/aerogo2 中已有 YAML，请人工比较新旧配置。

## 3. 配置稳定设备路径

~~~bash
ls -l /dev/serial/by-id/
ip -br link
sudoedit /etc/aerogo2/hardware.yaml
~~~

替换 Pixhawk 与 F446 的 /dev/serial/by-id 路径，并把 go2.network_interface 改成直连 Go2 DDS 网络的网卡。磁盘上的 hardware_write_enabled 必须保持 false；真实写权限只由一次性进程参数在内存中打开。

## 4. HW-RO 联机预检

~~~bash
set -a
source /etc/aerogo2/aerogo2.env
set +a
/opt/aerogo2/venv/bin/aerogo2 shell \
  --hardware-readonly \
  --config /etc/aerogo2/hardware.yaml \
  --confirm-hardware I_UNDERSTAND_HARDWARE_RISK
~~~

控制台执行：

~~~text
connect all
devices
status --full
preflight
preflight transform-flight
~~~

HW-RO 在权限层和桥接器层都拒绝执行器写入。确认 F446 是已知限位且 duty=0；Pixhawk 心跳新鲜、未解锁、RC 无 failsafe；4 路 ESC 映射正确、在线、RPM=0；Go2 无 fault 且三轴速度接近 0；CH5 为 LOW，CH9 与准备切换方向一致。

只读开机监控：

~~~bash
sudo systemctl enable --now aerogo2-monitor.service
journalctl -u aerogo2-monitor.service -f
~~~

该服务永远以 HW-RO 启动。手动实机控制前先停止它，避免串口和 DDS 被两个进程占用。

## 5. 让 F446 变形电机转起来

保持拆桨、Pixhawk 未解锁、机体固定：

~~~bash
sudo systemctl stop aerogo2-monitor.service
set -a
source /etc/aerogo2/aerogo2.env
set +a
/opt/aerogo2/venv/bin/aerogo2 shell \
  --hardware \  
  --enable-hardware-write \
  --config /etc/aerogo2/hardware.yaml \
  --confirm-hardware I_UNDERSTAND_HARDWARE_RISK
~~~

WALK 到 FLIGHT：CH5 LOW、CH9 稳定为 FLIGHT_REQUEST、原装遥控器停止使用。

~~~text
connect all
preflight transform-flight
transform flight
TRANSFORM_TO_FLIGHT
~~~

只有当前形态是 WALK、Go2 静止保持满足、RC 新鲜、Pixhawk 未解锁、ESC 在线且 RPM=0、F446 无故障且电流裕量满足时，才发送 limf 或 limr。目标限位与 duty=0 验证通过后进入 `GO2_JOINT_LOCK_WAIT`。此时先在 Unitree 手机端选择 Lock On；Shell 检测到 mode=6 或 `error_code=1002` 后，会等待姿态扰动滤波结束并重新静止，再自动进入 `FLIGHT_READY`。若固件两种信号都不回报，才使用 `go2 confirm-lock` 和精确口令 `CONFIRM_GO2_JOINT_LOCK`。可用 `transform status` 检查锁定来源和滤波计时。

FLIGHT 到 WALK：先确认落地且未解锁，CH9 稳定为 WALK_REQUEST。

~~~text
preflight transform-walk
transform walk
TRANSFORM_TO_WALK
~~~

卡滞或故障时执行 transform stop、stop、faults active 和 status --full。不要自动清除 F446 fault；先断动力并排除机械原因。

## 6. Go2 与 X8

仅在 WALK 且 F446 确认 WALK 限位时允许 walk stop 与 walk stand。它们使用 Unitree SportClient 高层接口，不直接下发关节电机命令。

Unitree 高层接口没有可证明进入 mode=6 的公开 JointLock 方法，`StandUp` 也不等于锁关节。0.3.13 根据这台 Go2 的前后对照实测，把配置项 `joint_lock_state_codes: [1002]` 作为 Lock On 遥测证据；普通站立的 100 不会被识别为锁定。识别后软件调用 `SwitchJoystick(false)`，并持续监测连接、状态码和速度；若 1002 消失，关节锁确认也会丢失并触发既有保护。

主状态机没有直接 arm/disarm API；`flight authorize` 只在完整互锁通过后开启 30 秒一次性许可，随后必须由 RadioMaster CH5 LOW->HIGH 触发 Pixhawk Lua 的普通受检 Arm。Lua 阻断 MAVLink Arm/force-arm 绕过，但保留正常 Disarm。完整条件见 `STATE_TRANSITIONS_ZH.md`。

X8 motor-test 与飞行 Arm 隔离；仅可在全部拆桨、限流供电并固定机体后使用台架入口。

先执行只读的映射与工具一致性检查：

~~~bash
/opt/aerogo2/venv/bin/aerogo2 x8-bench \
  --config /etc/aerogo2/hardware.yaml \
  --check
~~~

`x8-bench` 仅允许显式的非驱动诊断白名单，不能进入原始交互终端，也不能透传
`safety off`、arm、飞行、参数写入或电机命令。需要只读核对时使用例如
`x8-bench -- --commands "audit std; x8diag 3"`；唯一保留的电机台架路径是下面具有
拆桨、固定机架、精确确认和幅值/时长上限的 `x8-spin`。

确认 Pixhawk 为 DISARMED、4 路 ESC 遥测新鲜且 RPM=0 后，先以单臂 10%、2 秒测试：

~~~bash
/opt/aerogo2/venv/bin/aerogo2 x8-spin \
  --config /etc/aerogo2/hardware.yaml \
  --target lf \
  --percent 10 \
  --duration 2 \
  --props-removed \
  --airframe-secured \
  --confirm-x8 X8_PROPS_REMOVED_AND_AIRFRAME_SECURED
~~~

target 只可选 rr、lf、lr、rf；油门仅允许 5–20%，时间仅允许 0.5–5 秒。工程尚无可验证的四路映射验收记录，因此 `all` 在代码中始终被拒绝。

命令会依次审计标准参数、检查 ESC 遥测与 DISARMED 联锁、限时发送 MAVLink MOTOR_TEST、停转并恢复 safety on；任一步失败都终止，退出清理也会再次停转和恢复安全开关。

## 7. 验收

先让 HW-RO 连续运行 30 分钟，再做机构脱载低占空比测试；两个方向各循环至少 20 次并记录电流和时间。随后注入拔线、RC failsafe、Pixhawk armed、ESC 非零、Go2 运动和 F446 过流，确认全部 fail closed。最后才做无桨整机联调；桨叶和飞行测试需要独立方案与现场安全负责人。
