# AeroGo2 0.3.10：离地锁存与触地误触发修复

## 修改结果

0.3.10 在每个新飞行周期增加一次性离地锁存。进入 `FLIGHT_MANUAL` 后，触地检测默认保持关闭；只有 Pixhawk 新鲜遥测连续满足 `armed=true`、`landed=false` 达到 `safety.airborne_confirm_s`（默认 1.0 秒），才记录 `AIRBORNE_CONFIRMED` 并启用原有触地判定。

因此，飞机还在地面、已经 Arm、ESC 尚未起转或停转时，即使 `landed=true`、姿态平稳、RPM 很低，也不会因为等待 2 秒误进 `TOUCHDOWN_VERIFY`。锁存会跨 `FLIGHT_MANUAL` 与 `AUTO_LANDING` 保留，在下一架次、回到安全地面态、FAULT 或急停时复位。

## 新只读命令

在 Shell 任意状态输入：

```text
touchdown status
```

重点字段：

- `airborne_confirmed=false`：本架次尚未确认离地。
- `touchdown_detection_enabled=false`：触地计时器被硬性禁用。
- `airborne_candidate_elapsed_s`：连续离地样本已经保持的秒数。
- `touchdown_candidate_elapsed_s`：离地锁存后，当前触地条件连续保持的秒数。
- `conditions`：landed、速度、姿态、ESC RPM 等每项判据。

## 配置

新配置默认值：

```yaml
safety:
  airborne_confirm_s: 1.0
```

旧版 `/etc/aerogo2/hardware.yaml` 没有此字段时自动使用 1.0 秒，不会因缺少字段而启动失败。若明确配置，必须是大于 0 的有限数。修改后需要重启 Shell。

## 安装

把 `aerogo2-0.3.10-py3-none-any.whl` 复制到 Go2 Ubuntu 的 `~/aerogo2/` 后执行单行命令：

```bash
sudo /opt/aerogo2/venv/bin/pip install --no-deps --force-reinstall ~/aerogo2/aerogo2-0.3.10-py3-none-any.whl
```

确认加载版本：

```bash
/opt/aerogo2/venv/bin/python -c "import aerogo2; print(aerogo2.__version__, aerogo2.__file__)"
```

本次只更新 Go2 Ubuntu 上的 AeroGo2 Python 包，不需要向 Pixhawk 复制新文件；原有 `/APM/scripts/aerogo2_arm_gate.lua` 继续保留。

## 验证流程

1. 地面进入 `FLIGHT_MANUAL` 后先输入 `touchdown status`，应看到 `airborne_confirmed=false`、`touchdown_detection_enabled=false`。在地面停留超过 3 秒，状态仍必须是 `FLIGHT_MANUAL`。
2. 按既定安全流程真实离地并保持超过 1 秒，再输入 `touchdown status`，应看到两个字段变为 `true`。
3. 降落后，只有 landed、垂直速度、倾角、ESC RPM 和高度稳定条件连续满足默认 2 秒，才自动进入 `TOUCHDOWN_VERIFY`。
4. `TOUCHDOWN_VERIFY` 仍不会自动 Disarm；必须继续使用 RadioMaster/Pixhawk 的既定 Disarm 流程。

首次真机验证必须在清空人员区域、可靠固定/系留、操作者随时能用 RadioMaster 接管和断动力的条件下逐项完成。软件测试不能证明 Pixhawk 的 landed 判定或实际旋翼系统必然正确。
