# AeroGo2 0.3.2 真机部署与落地适应操作

本文所有终端命令均为单行。0.3.2 包含从 0.3.0 起的全部运行时修改，包括 F446 慢速逐字节串口、高层 Go2 飞行关节锁、两把钥匙 Arm、ESC 遥测一致性修复，以及本次受控落地适应。

## 安全边界

- 真机自动降落 setpoint 仍未开放；降落、Arm 和 Disarm 继续由 RadioMaster/Pixhawk 完成。
- 首次触地仍保持 `JOINT_LOCK`。只有 Pixhawk landed、RadioMaster 已 Disarm、四个配置的 X8 全部在线健康且 RPM 精确为 0、F446 停止无故障、脚底压力满足校准条件时才进入 `BalanceStand`。
- `LANDING_COMPLIANT` 期间 Unitree 原装遥控保持禁用。`transform walk` 会先恢复并确认 mode=6 `JOINT_LOCK`，然后才允许 F446 动作。
- 任一 X8 缺失或离线都禁止进入落地适应和形态转换。三脚接触是 Go2 四个脚底传感器中的最少接触数，不代表允许只安装三个 X8。
- 未校准时功能默认关闭；四个阈值中任一为 0 都不能启用。

## 安装完整wheel

把 `aerogo2-0.3.2-py3-none-any.whl` 放到 Unitree 的 `~/aerogo2/release/` 后执行：

```bash
sudo /opt/aerogo2/venv/bin/pip install --no-deps --force-reinstall ~/aerogo2/release/aerogo2-0.3.2-py3-none-any.whl
```

确认加载的是 0.3.2：

```bash
/opt/aerogo2/venv/bin/python -c "import aerogo2; print(aerogo2.__version__, aerogo2.__file__)"
```

确认单行配置工具已经安装：

```bash
/opt/aerogo2/venv/bin/aerogo2-configure-landing --help
```

## 仅用于干净0.3.0源码树的统一补丁

如果目录是未经修改的0.3.0源码树，可先检查、再应用统一补丁：

```bash
cd ~/aerogo2 && git apply -p2 --check release/aerogo2-0.3.0-to-0.3.2.patch
```

```bash
cd ~/aerogo2 && git apply -p2 --whitespace=nowarn release/aerogo2-0.3.0-to-0.3.2.patch
```

然后重装源码：

```bash
cd ~/aerogo2 && sudo /opt/aerogo2/venv/bin/pip install --no-deps --no-build-isolation --force-reinstall .
```

如果目录已经包含0.3.1改动或之前的手工修复，不要重复应用此补丁，直接安装上面的完整0.3.2 wheel。

## 第一次只读采集四脚压力

先不要启用落地适应。保持 Pixhawk Disarm、X8 停转、F446 停止，然后启动只读Shell：

```bash
set -a; source /etc/aerogo2/aerogo2.env; set +a; MAVLINK20=1 /opt/aerogo2/venv/bin/aerogo2 shell --hardware-readonly --config /etc/aerogo2/hardware.yaml --confirm-hardware I_UNDERSTAND_HARDWARE_RISK
```

进入Shell后依次输入：

```text
connect all
landing compliance
go2 status
```

在安全吊起、四脚完全无载时记录多组 `foot_force[0..3]`，再让Go2正常站立并分别轻压四个脚，记录各通道有载值。只有每个通道在受力时都稳定增大，才可继续。每脚阈值应位于该脚“无载最大值”和“可靠有载最小值”之间，并留出噪声余量；不要复制另一台Go2的阈值。

## 一条命令写入并启用

将下面的 `F0 F1 F2 F3` 替换为四个实际正整数阈值：

```bash
sudo /opt/aerogo2/venv/bin/aerogo2-configure-landing --config /etc/aerogo2/hardware.yaml --enable --thresholds F0 F1 F2 F3 --minimum-contact-feet 3 --contact-confirm-s 0.5 --settle-s 1.5
```

工具会先构造候选文件并调用完整配置加载器验证，通过后备份为 `/etc/aerogo2/hardware.yaml.bak`，再原子替换原文件。验证最终值：

```bash
/opt/aerogo2/venv/bin/python -c "from aerogo2.common.config import load_config; c=load_config(\"/etc/aerogo2/hardware.yaml\"); print(c.go2)"
```

如需立即关闭功能：

```bash
sudo /opt/aerogo2/venv/bin/aerogo2-configure-landing --config /etc/aerogo2/hardware.yaml --disable
```

## 无桨地面验收

第一次必须拆除全部桨叶、固定机体、保持Pixhawk Disarm，并确认四个X8遥测全部在线。启动可写真机Shell：

```bash
set -a; source /etc/aerogo2/aerogo2.env; set +a; MAVLINK20=1 /opt/aerogo2/venv/bin/aerogo2 shell --hardware --enable-hardware-write --config /etc/aerogo2/hardware.yaml --confirm-hardware I_UNDERSTAND_HARDWARE_RISK
```

在Shell中检查：

```text
connect all
devices
esc health
landing compliance
```

完整落地后的状态顺序应为：

```text
FLIGHT_MANUAL -> TOUCHDOWN_VERIFY -> LANDING_COMPLIANT -> FLIGHT_READY -> FLIGHT_TO_WALK_PRECHECK -> TRANSFORM_TO_WALK -> WALK
```

实际操作顺序：

1. 使用RadioMaster人工降落；AeroGo2确认触地前一直保持 `JOINT_LOCK`。
2. 使用RadioMaster Disarm；等待 `pixhawk status` 显示 `armed=false`，并用 `esc health` 确认四个X8在线、健康且RPM全部为0。
3. 输入 `landing compliance`；脚接触持续0.5秒后状态应自动进入 `LANDING_COMPLIANT`，Go2模式应为 `BALANCE_STAND`。
4. 等 `settle_elapsed_s` 达到1.5秒，再输入 `preflight transform-walk`。
5. 输入 `transform walk`，再输入精确确认词 `TRANSFORM_TO_WALK`。
6. 变形成功进入 `WALK` 后关节仍保持锁定；输入 `walk stand` 才恢复 `BalanceStand` 并重新启用Unitree原装遥控。

任何一步出现 `FAULT` 时不要反复执行变形。依次只读检查 `faults active`、`landing compliance`、`go2 status`、`pixhawk status` 和 `esc health`。系统不会替你Disarm或停止旋翼。

## 0.3.2验收结果

- `ruff check`：通过。
- strict `mypy`：通过。
- 完整自动测试：`595 passed`。
- 统一补丁：`release/aerogo2-0.3.0-to-0.3.2.patch`。
- 完整安装包：`release/aerogo2-0.3.2-py3-none-any.whl`。
