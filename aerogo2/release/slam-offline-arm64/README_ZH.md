# AeroGo2 L1 Point-LIO 离线安装包

目标环境：Ubuntu 20.04、aarch64、ROS Noetic 与 ROS Foxy。

该包只用于第一阶段离线定位验证：

- 安装 ROS Foxy `ros1_bridge` 及其已解析出的 `gazebo_msgs` 依赖。
- 解包 Unitree 官方 `point_lio_unilidar` 源码到 `~/aerogo2_slam_ws`。
- 将输入话题从 `/unilidar/cloud`、`/unilidar/imu` 改为 Go2 已有的 `/utlidar/cloud`、`/utlidar/imu`。
- 在只加载 Noetic 的干净环境中编译，避免 Foxy/Noetic 路径污染。

它不会启动 Point-LIO、不会向 Pixhawk 发送 MAVLink、不会控制 Go2 或任何电机。

将整个目录传到 Go2 后，在目录内执行：

```bash
bash 01_install_and_build.sh
```

成功标志：

```text
POINT_LIO_BUILD=ok
ROS1_BRIDGE=installed
```

来源：

- Point-LIO: https://github.com/unitreerobotics/point_lio_unilidar
- ROS 2 package archive: https://repo.ros2.org/ubuntu/main/pool/main/r/

文件完整性由 `SHA256SUMS` 固定。源码归档下载于 2026-08-21；其归档 SHA-256 是该离线包的可复现版本标识。
