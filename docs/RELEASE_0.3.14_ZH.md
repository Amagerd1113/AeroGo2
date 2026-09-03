# AeroGo2 0.3.14：人工 FLIGHT 端点可正常授权

0.3.14 修复人工端点路径进入 `FLIGHT_READY` 后，`flight authorize` 错误
报告 `F446_FLIGHT_STATE_MISMATCH` 的问题。

人工执行 `ms`、`motor endpoint flight`、`motor confirm flight` 后，F446
保持 `IDLE + duty=0`，构型来源为内部守卫记录的 `operator`。现在授权、
后续变形和 WALK 检查统一接受以下两类构型证明：

- F446 回报配置的硬件限位状态；
- `configuration_source=operator` 且 F446 为 `IDLE + duty=0 + 无故障`。

F446 断连、出现故障、恢复非零 duty 或进入不匹配限位时，人工构型证明
仍会失效。

此外，当 1002 已让系统从 `GO2_JOINT_LOCK_WAIT` 自动进入
`FLIGHT_READY` 时，再输入 `go2 confirm-lock` 会返回
`GO2_JOINT_LOCK_ALREADY_CONFIRMED`，不再显示 `STATE_DENIED`。

安装：

```bash
sudo /opt/aerogo2/venv/bin/pip install --no-deps --force-reinstall ~/aerogo2/aerogo2-0.3.14-py3-none-any.whl
```
