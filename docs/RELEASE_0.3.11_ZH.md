# AeroGo2 0.3.11：Go2 Lock On 人工确认兼容方案

## 问题与修复

这台 Go2 在 Unitree 手机端进入 Lock On 后，`SportModeState` 仍回报 `mode=0`、`error_code=1002`，因此原有只接受 `mode=6` 的路径无法自动确认，而且 1002 会让旧版把静止姿态误判为不稳定。

0.3.11 是当前统一轮子，包含 0.3.10 的“确认离地后才启用触地判定”修复，并新增下面两项彼此独立的 Go2 修改：

- `go2.accepted_state_codes` 默认改为 `[0, 100, 1002]`，使 1002 可以参与正常姿态和静止判断。
- 新增 `go2 confirm-lock`。它只记录操作者对手机 Lock On 的明确确认，不会把原始 `mode=0` 改成 mode 6，也不会把原始 `joints_locked=false` 改成 true。

原始 mode 6 自动路径仍优先使用。人工路径成功后会显示：

```text
joint_lock_telemetry  false
joint_lock_confirmed  true
joint_lock_source     operator
```

## 人工确认前的硬互锁

`go2 confirm-lock` 仅允许在 `GO2_JOINT_LOCK_WAIT` 使用，并再次要求：

- Pixhawk、F446、Go2、RC 遥测均在线且新鲜。
- Pixhawk 为 Disarm，RC 无 failsafe，CH5 保持 LOW。
- 四个配置的 X8 ESC 全部唯一、在线、健康、RPM 有限且精确为 0。
- F446 已处于验证的 FLIGHT 构型、duty=0、无故障。
- Go2 速度有限并低于静止阈值、`stable=true`、`moving=false`、`controller_active=false`。
- Go2 `error_code` 位于 `accepted_state_codes`。
- 没有活动故障。

通过后 AeroGo2 调用 `SwitchJoystick(false)`，以 `joint_lock_source=operator` 进入 `FLIGHT_READY`。之后若检测到行走模式、运动、控制器重新活动或未知状态码，会触发 `GO2_OPERATOR_LOCK_UNSAFE` 并进入故障闭锁。

## 安装

把 `aerogo2-0.3.11-py3-none-any.whl` 复制到 Go2 Ubuntu 的 `~/aerogo2/`，执行单行命令：

```bash
sudo /opt/aerogo2/venv/bin/pip install --no-deps --force-reinstall ~/aerogo2/aerogo2-0.3.11-py3-none-any.whl
```

检查加载版本：

```bash
/opt/aerogo2/venv/bin/python -c "import aerogo2; print(aerogo2.__version__, aerogo2.__file__)"
```

应显示 `0.3.11`。不需要把新文件上传到 Pixhawk；原有 Arm Gate Lua 保持不变。

旧版 `/etc/aerogo2/hardware.yaml` 即使没有 `accepted_state_codes` 也会自动使用默认值。若希望显式记录，可在 `go2:` 下加入：

```yaml
accepted_state_codes: [0, 100, 1002]
```

## 真机单行操作顺序

按原流程启动可写真机 Shell 并执行 `connect all`。若从 WALK 自动变形，执行原有 `transform flight` 流程即可。若要由操作者确认当前机械位置为 FLIGHT 端点，完整单行输入顺序是：

```text
motor maintenance enter
y
ENTER_F446_MANUAL
ms
motor endpoint flight
MARK_CURRENT_ENDPOINT_FLIGHT
motor confirm flight
CONFIRM_MANUAL_FLIGHT
```

系统进入 `GO2_JOINT_LOCK_WAIT` 后：

1. 在 Unitree 手机端选择 Lock On。
2. 输入 `go2 status`。如果原始 `joints_locked=true`，系统会自动进入 `FLIGHT_READY`，无需人工命令。
3. 如果仍是 `IDLE_STAND`、`fault_code=1002`、`joints_locked=false`，输入 `go2 confirm-lock`。
4. 在无历史确认提示中输入 `CONFIRM_GO2_JOINT_LOCK`。
5. 输入 `transform status`，必须看到 `system_state=FLIGHT_READY`、`joint_lock_confirmed=true`、`joint_lock_source=operator`、`f446_duty=0`。
6. 保持 CH5 LOW，输入 `flight authorize`，再输入 `AUTHORIZE_FLIGHT`。
7. 授权成功后才把 RadioMaster CH5 从 LOW 切到 HIGH，由 Pixhawk 执行正常 Arm 和全部 PreArm 检查。

## 重要限制

人工确认是针对“手机已经锁定，但固件不给出可识别 mode 6”的兼容方案。若之后在手机端解除 Lock On，而 Go2 保持静止且遥测继续显示 mode 0，AeroGo2 无法从现有 `SportModeState` 可靠区分“仍锁定”和“已解锁”。因此：

- 人工确认后禁止在手机端解除 Lock On。
- 操作者必须始终保留 RadioMaster/Pixhawk 接管和物理断动力能力。
- 首次验证必须拆桨或确保旋翼完全断电，并固定机体；先验证状态机，再进行受控的推进系统测试。
- 软件单元测试不能替代这台真机的锁关节、关节保持力、急停和供电链验收。