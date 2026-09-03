# AeroGo2 0.3.13：Go2 EDU Lock On 1002 兼容与过渡滤波

## 锁定识别

这台 Go2 EDU 的实测前后对照为：普通站立上报 `mode=0,error_code=100`，
手机选择 Lock On 后上报 `mode=0,error_code=1002`。0.3.13 默认配置：

```yaml
go2:
  joint_lock_state_codes: [1002]
  joint_lock_transition_grace_s: 2.0
  joint_lock_unsafe_confirm_s: 0.5
  accepted_state_codes: [0, 100, 1002]
```

因此 mode=6 或 error_code=1002 都会成为关节锁遥测。普通 100 仍是未锁定。

## 滤波

进入 `GO2_JOINT_LOCK_WAIT` 后，手机切换 Lock On 造成的瞬时姿态运动享有
2.0 秒过渡宽限；宽限结束后，运动/模式异常必须连续保持 0.5 秒才触发
`GO2_UNSAFE_DURING_JOINT_LOCK`。Pixhawk、RC、ESC、F446 和设备超时类
故障不参与该滤波，仍立即拦截。

检测到 1002 但 Go2 仍在晃动时，系统保持 `GO2_JOINT_LOCK_WAIT`；只有
三轴速度、合速度、stable/moving/controller 状态全部恢复安全后，才调用
`SwitchJoystick(false)` 并进入 `FLIGHT_READY`。

## 安装与确认

```bash
sudo /opt/aerogo2/venv/bin/pip install --no-deps --force-reinstall ~/aerogo2/aerogo2-0.3.13-py3-none-any.whl
```

```bash
/opt/aerogo2/venv/bin/python -c "import aerogo2; print(aerogo2.__version__, aerogo2.__file__)"
```

进入 Shell 后，手机切换前后分别执行 `go2 status`。切换后应看到：

```text
fault_code             1002
joint_lock_telemetry   True
joint_lock_confirmed   True
joint_lock_source      telemetry
```

`transform status` 还会显示初始宽限剩余时间和持续异常已观察时间。
