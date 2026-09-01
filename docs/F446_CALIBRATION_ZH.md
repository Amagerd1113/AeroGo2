# F446 变形电机与 HW-039 堵转阈值标定

本指南适用于 AeroGo2 0.3.9。所有终端命令均为单行。`ms` 只停止 F446 变形电机；`s`/`stop` 仍是全系统受控停止。

## 1. 安全准备

必须物理拆下四个旋翼、确保 Pixhawk 为 DISARMED，并把机体可靠支撑。若维护预检要求四个 ESC 遥测在线，只能在拆桨后给 X8/ESC 上电，并确认四路 RPM 都为 0；否则保持 X8 动力断电。变形连杆、插销和电机附近不得有人手或工具。准备好随时输入 `ms`，不要一开始就使用 500 占空比。

检查版本：

```bash
/opt/aerogo2/venv/bin/python -c "import aerogo2; print(aerogo2.__version__)"
```
首次进入 shell 前，先把主机与 F446 本地的总超时统一为 15 秒并把正式 WALK 占空比设为 300。此时 threshold 保持 0，表示暂不覆盖板上尚未标定的值：

```bash
sudo /opt/aerogo2/venv/bin/aerogo2-configure-f446 --config /etc/aerogo2/hardware.yaml --walk-duty 300 --flight-duty 120 --transform-timeout-s 15 --firmware-timeout-ms 15000 --threshold-adc 0 --blanking-ms 500 --overcurrent-ms 180
```

如果该命令未成功结束，不要进入电机测试；先根据它的 `Configuration not changed` 错误修复配置文件。


启动写入模式 shell：

```bash
set -a; source /etc/aerogo2/aerogo2.env; set +a; MAVLINK20=1 /opt/aerogo2/venv/bin/aerogo2 shell --hardware --config /etc/aerogo2/hardware.yaml --confirm-hardware I_UNDERSTAND_HARDWARE_RISK
```

连接设备并进入维护模式：

```text
connect all
```

```text
motor maintenance enter
```

按提示完成两阶段确认。若暂时不接 X8，只有在当前预检明确允许时才继续，不能绕过安全检查。

## 2. 先验证方向与最低可动占空比

查看当前参数：

```text
motor parameters
```

从小占空比开始正向点动：

```text
motor mf 80
```

立即只停变形电机：

```text
ms
```

从小占空比开始反向点动：

```text
motor mr 80
```

立即只停变形电机：

```text
ms
```

每次只增加 20 到 30，找到能稳定启动但冲击最小的值。确认 `mr` 确实朝 WALK/收缩方向、`mf` 朝 FLIGHT/展开方向。任何方向相反都应停止，不要通过软件名称强行继续。

## 3. 记录 HW-039 电流并选择 threshold

在静止、正常移动和机械端点分别多次执行：

```text
motor current
```
运动命令返回后，可连续输入 `motor current` 取得运动中的近实时样本；每次短测结束立即输入 `ms`。到机械端点只允许短时取样，不能为了读数持续堵转。


记录 `R_IS`、`L_IS` 和 `used` 的 ADC 原始值。建议至少取得：静止噪声最大值、正常运动峰值、端点堵转的最低重复值。

阈值必须高于“正常运动峰值 + 噪声余量”，并低于“端点堵转最低重复值”。本项目默认主机安全范围是 201 到 1400 ADC；不能复制别人的数值。若正常峰值与堵转值没有可靠间隔，应先检查 HW-039 接线、采样方向、驱动器和机械阻力，不能靠提高上限掩盖问题。

设置并读回候选阈值（示例 900 仅表示命令格式）：

```text
motor threshold 900
```

查看读回值：

```text
motor parameters
```

## 4. 调整 blank、overms 和 timeout

`blank` 是启动后忽略浪涌电流的时间，默认 500 ms；`overms` 是电流连续超过 threshold 才判定到位的时间，默认 180 ms；`timeout` 是 F446 本地最长运动时间，默认 15000 ms。

设置并读回本地超时：

```text
motor timeout 15000
```

设置并读回启动屏蔽时间：

```text
motor blank 500
```

设置并读回持续过流时间：

```text
motor overms 180
```

约束为 `blank + overms < timeout`。blank 太短会把启动浪涌误判为端点；blank 太长会延迟真正堵转保护。overms 太短会受尖峰触发，太长会增加端点受力时间。每次只改一个参数并重新测试。

## 5. 验证自动限位

进入手动维护后，先由操作员明确标记当前位置是哪个端点。标记本身不要求本次会话发生过运动，也不会离开 `MANUAL_POSITIONING`：

```text
motor endpoint walk
MARK_CURRENT_ENDPOINT_WALK
```

或者：

```text
motor endpoint flight
MARK_CURRENT_ENDPOINT_FLIGHT
```

标记完成后，再分别执行 `motor confirm walk` 或 `motor confirm flight` 进入 `WALK` 或 `FLIGHT_READY`。如果没有先标记匹配的端点，确认命令会被拒绝。F446 必须已经停止，且状态为 `IDLE` 或匹配的 `LIMIT_REACHED_FWD/REV`；若 F446 明确报告相反端点、仍在运动或处于故障，标记和确认仍会被拒绝。

把机构放在离端点有安全距离的位置，先验证反向自动限位：

```text
motor limr 300
```

正常结果必须是 `LIMIT_REACHED_REV` 且 `duty=0`。如果机构不动、方向错误、电流异常或接近端点仍不停，立即输入：

```text
ms
```

再以已验证的较低占空比测试正向：

```text
motor limf 120
```

正常结果必须是 `LIMIT_REACHED_FWD` 且 `duty=0`。同一组参数至少重复三次，确认没有正常运动误停，也没有端点不停。

## 6. 持久化参数

退出 shell 后，把已验证数值写入 `/etc/aerogo2/hardware.yaml`。下面的 900 必须替换为你的实测阈值：

```bash
sudo /opt/aerogo2/venv/bin/aerogo2-configure-f446 --config /etc/aerogo2/hardware.yaml --walk-duty 300 --flight-duty 120 --transform-timeout-s 15 --firmware-timeout-ms 15000 --threshold-adc 900 --blanking-ms 500 --overcurrent-ms 180
```

工具会先验证完整配置、生成 `/etc/aerogo2/hardware.yaml.bak`，然后原子替换。非零 `threshold-adc` 会在每次写入模式连接 F446 时自动重设；设为 0 则保留板上的阈值。

重新启动 shell 后确认：

```text
motor parameters
```

最后执行正式 WALK 回零：

```text
transform home-walk
```

只有收到 `F446_CONFIGURATION_VERIFIED`、状态为 `LIMIT_REACHED_REV` 且 `duty=0`，才可人工确认后进入下一状态。15 秒内未到位会停止并拒绝状态切换，不能通过 `clear-fault` 把未知机械位置当作已到位。
