# AeroGo2 0.3.3 ESC 遥测槽偏移配置

本文所有终端命令均为单行。0.3.3 包含 0.3.2 的全部真机功能，并将 Hobbywing X8-G2 的 MAVLink 遥测槽偏移改为显式配置，默认值为 `0`。

## 安装完整 wheel

把 `aerogo2-0.3.3-py3-none-any.whl` 放到 Unitree 的 `~/aerogo2/release/` 后执行：

```bash
sudo /opt/aerogo2/venv/bin/pip install --no-deps --force-reinstall ~/aerogo2/release/aerogo2-0.3.3-py3-none-any.whl
```

确认版本：

```bash
/opt/aerogo2/venv/bin/python -c "import aerogo2; print(aerogo2.__version__)"
```

## 选择遥测槽偏移

新版 X8-G2 固件按实际编号上报槽 `1,2,3,4`，使用默认值 `0`。现有 `/etc/aerogo2/hardware.yaml` 即使没有新键也会自动使用 `0`。

需要显式写入新版偏移时执行：

```bash
sudo /opt/aerogo2/venv/bin/python -c "import pathlib,yaml; p=pathlib.Path('/etc/aerogo2/hardware.yaml'); d=yaml.safe_load(p.read_text()) or {}; d.setdefault('esc',{})['mavlink_display_shift']=0; p.write_text(yaml.safe_dump(d,sort_keys=False))"
```

四只均为旧版固件、原始监测槽为 `2,3,4,5` 时，将值改成 `1`：

```bash
sudo /opt/aerogo2/venv/bin/python -c "import pathlib,yaml; p=pathlib.Path('/etc/aerogo2/hardware.yaml'); d=yaml.safe_load(p.read_text()) or {}; d.setdefault('esc',{})['mavlink_display_shift']=1; p.write_text(yaml.safe_dump(d,sort_keys=False))"
```

确认最终加载值：

```bash
/opt/aerogo2/venv/bin/python -c "from aerogo2.common.config import load_config; c=load_config('/etc/aerogo2/hardware.yaml'); print('mavlink_display_shift=',c.esc.mavlink_display_shift)"
```

偏移只允许 `0` 或 `1`。修改后必须完全退出并重新启动 AeroGo2 Shell，已运行的 Pixhawk bridge 不会热切换该硬件映射。

## 只读检查

桨叶拆除，不 ARM，不执行电机测试。启动 Shell 后依次执行：

```text
connect pixhawk
```

```text
esc mapping
```

```text
pixhawk status
```

```text
esc health
```

新版四只固件统一且偏移为 `0` 时，预期 `esc_raw_present_slots` 为 `[1,2,3,4]`。旧版四只统一且偏移为 `1` 时，预期为 `[2,3,4,5]`。

## 重要限制

偏移是整条 MAVLink ESC 遥测流的全局设置，不是单电机设置。旧 1 号和新 2 号同时上报监测槽 2 时，Pixhawk 的 MAVLink 数组中已经丢失唯一身份，AeroGo2 无法通过软件恢复。混用两套编号规则不得进入 ARM 或飞行状态，必须统一四只 ESC 固件。

若原始遥测出现不属于当前偏移的槽位，0.3.3 会把全部 ESC 标记为不健康，避免错误映射后继续执行真机状态转换。
