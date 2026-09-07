# AeroGo2 0.3.15：起飞前自动 Disarm 后可重新授权

0.3.15 修复 Pixhawk 在已进入 `FLIGHT_MANUAL`、但尚未起油门离地时自动
Disarm，导致 AeroGo2 上层仍停留在 `FLIGHT_MANUAL` 的问题。

现在只有同时满足以下条件时才自动回到 `FLIGHT_READY`：

- 当前为 `FLIGHT_MANUAL`；
- 观察到新鲜 Pixhawk 遥测的 `armed: true -> false` 边沿；
- Pixhawk 报告 `landed=true`；
- 本架次尚未锁存 `AIRBORNE_CONFIRMED`。

回退会清除本架次离地/触地计时，且原一次性授权已被消费。操作者必须把
RadioMaster CH5 拉回 LOW，再重新执行 `flight authorize`。一旦已经确认
离地，Disarm 不会触发这条地面回退。

Pixhawk 的起飞前自动 Disarm 延时由 ArduPilot 参数控制。本版本要求：

```text
DISARM_DELAY = 20
```

新版 `aerogo2_arm_gate.lua` 会在每次授权时校验该参数；不是 20 时会拒绝
授权。更新轮子后，还必须把新版 Lua 复制到 Pixhawk SD 卡的
`/APM/scripts/aerogo2_arm_gate.lua`，设置参数并重启 Pixhawk。

安装：

```bash
sudo /opt/aerogo2/venv/bin/pip install --no-deps --force-reinstall ~/aerogo2/aerogo2-0.3.15-py3-none-any.whl
```
