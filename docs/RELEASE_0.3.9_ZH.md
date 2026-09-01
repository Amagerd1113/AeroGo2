# AeroGo2 0.3.9：手机人工锁关节状态机

## 修改结果

0.3.9 不再把 Unitree `StandUp()` 当作 mode=6 `JOINT_LOCK`。F446 到达并验证 FLIGHT 端点、`duty=0` 后，状态机进入：

```text
TRANSFORM_TO_FLIGHT -> GO2_JOINT_LOCK_WAIT -> FLIGHT_READY
```

在 `GO2_JOINT_LOCK_WAIT` 中，由操作者在 Unitree 手机端选择“锁关节/Joint Lock”。AeroGo2 收到权威 `SportModeState.mode=6` 后才调用 `SwitchJoystick(false)`，再次确认 mode=6，然后自动进入 `FLIGHT_READY`。

模式 1 切到模式 6 时，允许小于 `safety.stationary_velocity_mps`（默认 0.05 m/s）的短暂姿态调整，即使此时上报 `stable=false`、`moving=true` 或 `controller_active=true`，也不会再仅因此误触发 `FAULT`。以下情况仍会立即 fail-closed：进入 `LOCOMOTION`、任一速度分量达到阈值、Pixhawk/RC failsafe、Pixhawk Arm、CH5 非 LOW、ESC 遥测不完整或 RPM 非零、F446 异常、Go2 数据过期。

## 安装

把 `aerogo2-0.3.9-py3-none-any.whl` 复制到 Go2 Ubuntu 的 `~/aerogo2/` 后执行：

```bash
sudo /opt/aerogo2/venv/bin/pip install --no-deps --force-reinstall ~/aerogo2/aerogo2-0.3.9-py3-none-any.whl
```

确认实际加载的版本：

```bash
/opt/aerogo2/venv/bin/python -c "import aerogo2; print(aerogo2.__version__, aerogo2.__file__)"
```

本次改动只更新 Go2 Ubuntu 上的 AeroGo2 Python 包，不需要把新文件烧录到 Pixhawk。原先用于“两把钥匙 Arm”的 `/APM/scripts/aerogo2_arm_gate.lua` 仍必须保留；0.3.9 没有修改该 Lua 文件。

## WALK 到 FLIGHT 的真机步骤

以可写真机模式启动 Shell：

```bash
set -a; source /etc/aerogo2/aerogo2.env; set +a; MAVLINK20=1 /opt/aerogo2/venv/bin/aerogo2 shell --hardware --enable-hardware-write --config /etc/aerogo2/hardware.yaml --confirm-hardware I_UNDERSTAND_HARDWARE_RISK
```

每条命令分别输入：

```text
connect all
preflight transform-flight
transform flight
TRANSFORM_TO_FLIGHT
```

F446 到位后，Shell 应显示 `GO2_JOINT_LOCK_OPERATOR_REQUIRED`，提示符状态应为 `GO2_JOINT_LOCK_WAIT`。这时：

1. 不要让 Go2 行走，不要 Arm Pixhawk。
2. 在 Unitree 手机端选择“锁关节/Joint Lock”（mode=6）。
3. 可随时输入单行命令 `transform status`，检查 `go2_mode`、`go2_joints_locked` 和 `joint_lock_operator_remaining_s`。
4. 检测到 mode=6 后，后台监控会自动切到 `FLIGHT_READY`，不需要再输入确认命令。

只有提示符已显示 `FLIGHT_READY` 后，才执行：

```text
preflight flight
flight authorize
```

随后必须在授权有效期内由 RadioMaster 把 CH5 从 LOW 切到 HIGH，由 Pixhawk 完成正常 PreArm/Arm；AeroGo2 Shell 不直接 Arm。

## 手动确认 FLIGHT 端点

若使用人工 F446 定位流程，停止并确认端点后也会进入同一个等待态：

```text
ms
motor endpoint flight
MARK_CURRENT_ENDPOINT_FLIGHT
motor confirm flight
CONFIRM_MANUAL_FLIGHT
```

最后一条确认成功后，不会直接跳到 `FLIGHT_READY`；应在手机端选择 mode=6，等待状态机自动完成。

## 等待时间配置

默认人工等待时间为 60 秒。`/etc/aerogo2/hardware.yaml` 没有该字段时仍自动使用 60 秒。需要调整时可在 `go2:` 段增加：

```yaml
go2:
  joint_lock_operator_timeout_s: 60.0
```

修改配置后必须重启 Shell。超时会进入 `FAULT`，但此时 F446 已经停止且 X8 必须保持 0 RPM。

## 落地适应后的重锁

若启用了 `LANDING_COMPLIANT`，第一次执行 `transform walk` 只会退出适应姿态并进入 `GO2_JOINT_LOCK_WAIT`。手机端选择 mode=6，等系统回到 `FLIGHT_READY` 后，再次执行 `transform walk` 才会启动向 WALK 端点的 F446 动作。

## 验证

发布前验证结果：完整 pytest、Ruff 与 mypy 均通过。真机仍须在拆桨、Pixhawk Disarm、机体固定、急停可用的条件下逐步验证状态反馈；自动测试不能代替实际 mode=1→6 的现场确认。
