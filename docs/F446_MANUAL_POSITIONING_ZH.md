# F446 人工定位与 HW-039 限位真机流程

适用于 Unitree 机载 aarch64 Ubuntu。所有 shell 命令均为单行；控制台命令每次输入一行。

## 安全边界

- 拆掉全部 X8 桨叶，固定机体，F446 电机使用限流供电，并准备独立断电。
- Pixhawk 必须 disarmed；CH5 必须 LOW；RC 不得 failsafe。
- Go2 必须稳定静止。
- 每个已配置 ESC 必须唯一、在线、遥测有限且 RPM 精确为 0。ESC 未供电时只能只读诊断。
- `s`/`stop` 不会 disarm 或停止 X8 旋翼；X8 始终由 RadioMaster/Pixhawk 独立安全链控制。

## 启动

先关闭只读监控服务，避免串口被占用：

~~~bash
sudo systemctl stop aerogo2-monitor.service
~~~

以一次性写权限启动：

~~~bash
set -a; source /etc/aerogo2/aerogo2.env; set +a; MAVLINK20=1 /opt/aerogo2/venv/bin/aerogo2 shell --hardware --enable-hardware-write --config /etc/aerogo2/hardware.yaml --confirm-hardware I_UNDERSTAND_HARDWARE_RISK
~~~

逐行执行：

~~~text
connect all
devices
preflight manual-position
manual enter
~~~

`manual enter` 的两阶段确认是先输入 `yes`，再输入 `ENTER_F446_MANUAL`。

## 人工收缩到 WALK

逐行执行：

~~~text
mr 500
s
confirm walk
~~~

`mr 500` 的两阶段确认是先输入 `yes`，再输入 `RUN_MANUAL_MOTOR`。`s` 是全局 `stop` 的快捷别名；它立即停止 F446、请求 Go2 停止并停止自动 setpoint，但在 `MANUAL_POSITIONING` 中不会退出人工定位会话。`confirm walk` 的精确确认文本是 `CONFIRM_MANUAL_WALK`。

若确认时提示电流或静止保持时间不足，保持不动，等待约 1 秒后重试 `confirm walk`。

## 人工展开到 FLIGHT

若当前已经退出人工定位，先执行：

~~~text
manual enter
~~~

然后逐行执行：

~~~text
mf 500
s
confirm flight
~~~

`mf 500` 的第二阶段确认同样是 `RUN_MANUAL_MOTOR`；`confirm flight` 的精确确认文本是 `CONFIRM_MANUAL_FLIGHT`。若提示保持时间不足，等待约 1 秒后重试。

人工确认只在当前进程中有效。重启、设备断开、故障或 `manual exit` 后构型重新视为未确认。

## 实时 HW-039 报告

电机 duty 非零时，控制台每 0.5 秒非阻塞显示：

~~~text
HW039 state=... duty=... R_IS=raw/mV L_IS=raw/mV used=raw/mV threshold=raw/mV over_active=...
~~~

这里是 ADC 原始值和毫伏。完成 HW-039 电流传递函数标定前，不能把它解释为安培值。

`mf`/`mr` 没有 F446 本地堵转停止，但仍受 AeroGo2 绝对过流联锁和主机运动超时保护。任何时候都可以输入 `s`；原命令 `stop` 仍保留。

## 自动堵转电流停止

先在停止状态设置并回读阈值：

~~~text
thr 1200
~~~

精确确认文本是 `CHANGE_F446_THRESHOLD`。主机会返回当前配置允许的 ADC 范围；如果 1200 不在范围内，以返回值为准。

自动收缩：

~~~text
limr 500
confirm walk
~~~

自动展开：

~~~text
limf 500
confirm flight
~~~

`limr`/`limf` 使用 F446 本地 HW-039 阈值、blank、overms 和 timeout 自动停止，同时仍受主机绝对过流与超时联锁。到位后 AeroGo2 继续要求 duty=0、电流恢复、方向匹配和人工确认。

## 故障处理

先输入：

~~~text
s
~~~

再逐行检查：

~~~text
faults active
status --full
motor current
~~~

不要自动发送 F446 `clear`。先断动力、排除机械卡滞和接线问题，再决定是否清故障。电流堵转只能表示负载变化，不能证明机械锁已啮合；最终飞行构型仍建议使用独立位置传感器或机械锁。
