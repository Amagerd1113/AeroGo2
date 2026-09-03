# AeroGo2 0.3.12：落地后人工回 WALK 后门

## 新增路径

0.3.12 在保留自动 `transform walk` 的同时，增加受保护的落地后人工恢复路径：

```text
TOUCHDOWN_VERIFY -> MANUAL_POSITIONING -> WALK
```

它用于自动回 WALK 机构动作不合适、需要操作者点动观察或手动确认端点的情况。它不是无条件绕过安全检查：Pixhawk 必须 Disarm，四个配置的 X8 必须在线、健康且 RPM 精确为 0，CH5 必须 LOW，RC/Pixhawk/F446/Go2 遥测必须新鲜，Go2 必须静止，F446 必须 duty=0、电流安全且无故障。

如果系统已经进入 `LANDING_COMPLIANT`，第一次进入人工维护前会先结束 `BalanceStand` 柔顺姿态并转入 `GO2_JOINT_LOCK_WAIT`。完成手机 Lock On 的 mode 6 自动确认，或执行 `go2 confirm-lock` 人工确认后，回到 `FLIGHT_READY`，再执行一次维护进入命令。F446 不会在柔顺腿部姿态下启动。

## 安装

把 `aerogo2-0.3.12-py3-none-any.whl` 复制到 Go2 Ubuntu 的 `~/aerogo2/` 后执行：

```bash
sudo /opt/aerogo2/venv/bin/pip install --no-deps --force-reinstall ~/aerogo2/aerogo2-0.3.12-py3-none-any.whl
```

检查版本：

```bash
/opt/aerogo2/venv/bin/python -c "import aerogo2; print(aerogo2.__version__, aerogo2.__file__)"
```

应显示 `0.3.12`。本次修改只在 Go2 Ubuntu 的 AeroGo2 Python 包中，不需要向 Pixhawk 或 F446 上传文件。

## TOUCHDOWN_VERIFY 中的单行操作顺序

先检查：

```text
touchdown status
```

```text
pixhawk status
```

```text
esc health
```

进入人工定位：

```text
motor maintenance enter
```

```text
y
```

```text
ENTER_F446_MANUAL
```

按实际机构方向和已标定占空比点动；以下仅表示命令格式：

```text
motor mr 300
```

随时停止 F446：

```text
ms
```

确认当前已经是 WALK 端点：

```text
motor endpoint walk
```

```text
MARK_CURRENT_ENDPOINT_WALK
```

```text
motor confirm walk
```

```text
CONFIRM_MANUAL_WALK
```

最后执行：

```text
state
```

必须看到 `WALK` 才能恢复 Go2 行走。不要把 `motor endpoint walk` 当作位置传感器；它是操作者对当前停止位置的明确声明，机械端点仍需目视确认。
