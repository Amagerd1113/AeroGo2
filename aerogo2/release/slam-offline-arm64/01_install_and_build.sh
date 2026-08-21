#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="${HOME}/aerogo2_slam_ws"
source_archive="${bundle_dir}/point_lio_unilidar-main.tar.gz"
gazebo_deb="ros-foxy-gazebo-msgs_3.5.3-1focal.20230527.060959_arm64.deb"
bridge_deb="ros-foxy-ros1-bridge_0.9.7-1focal.20230527.083653_arm64.deb"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: this bundle is only for aarch64."
    exit 2
fi

if [[ ! -f /opt/ros/noetic/setup.bash || ! -f /opt/ros/foxy/setup.bash ]]; then
    echo "ERROR: both ROS Noetic and ROS Foxy must already be installed."
    exit 2
fi

cd "${bundle_dir}"
sha256sum -c SHA256SUMS

sudo apt-get install -y "./${gazebo_deb}" "./${bridge_deb}"
test -x /opt/ros/foxy/lib/ros1_bridge/dynamic_bridge

if [[ -e "${workspace}/src/point_lio_unilidar" || -e "${workspace}/src/point_lio_unilidar-main" ]]; then
    echo "ERROR: ${workspace}/src already contains Point-LIO; nothing was overwritten."
    exit 3
fi

mkdir -p "${workspace}/src"
tar -xzf "${source_archive}" -C "${workspace}/src"
mv "${workspace}/src/point_lio_unilidar-main" "${workspace}/src/point_lio_unilidar"
sed -i 's#/unilidar/cloud#/utlidar/cloud#g; s#/unilidar/imu#/utlidar/imu#g' "${workspace}/src/point_lio_unilidar/config/unilidar_l1.yaml"

env -i HOME="${HOME}" USER="${USER:-unitree}" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 bash --noprofile --norc -c 'source /opt/ros/noetic/setup.bash; cd "$HOME/aerogo2_slam_ws"; catkin_make -DCMAKE_BUILD_TYPE=Release -j4'

test -x "${workspace}/devel/lib/point_lio_unilidar/pointlio_mapping"
echo "POINT_LIO_BUILD=ok"
echo "ROS1_BRIDGE=installed"
