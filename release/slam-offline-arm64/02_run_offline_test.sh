#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash 02_run_offline_test.sh /absolute/path/to/ros2_bag"
    exit 2
fi

bag_path="$(readlink -f -- "$1")"
workspace="${HOME}/aerogo2_slam_ws"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_stamp="$(date +%Y%m%d_%H%M%S)"
result_dir="${HOME}/aerogo2_slam_results/${run_stamp}"
ros_master_uri="http://127.0.0.1:11321"
ros_domain_id="42"
pids=()
cleaned=0

if [[ ! -f "${bag_path}/metadata.yaml" ]]; then
    echo "ERROR: ROS 2 bag metadata not found: ${bag_path}/metadata.yaml"
    exit 2
fi

if [[ ! -x "${workspace}/devel/lib/point_lio_unilidar/pointlio_mapping" ]]; then
    echo "ERROR: Point-LIO is not built in ${workspace}."
    exit 2
fi

if [[ ! -x /opt/ros/foxy/lib/ros1_bridge/dynamic_bridge ]]; then
    echo "ERROR: ros1_bridge is not installed."
    exit 2
fi

mkdir -p "${result_dir}"

ros1_env=(env -i HOME="${HOME}" USER="${USER:-unitree}" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 ROS_MASTER_URI="${ros_master_uri}")
bridge_env=(env -i HOME="${HOME}" USER="${USER:-unitree}" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 ROS_MASTER_URI="${ros_master_uri}" ROS_DOMAIN_ID="${ros_domain_id}" RMW_IMPLEMENTATION=rmw_cyclonedds_cpp)
ros2_env=(env -i HOME="${HOME}" USER="${USER:-unitree}" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 ROS_DOMAIN_ID="${ros_domain_id}" RMW_IMPLEMENTATION=rmw_cyclonedds_cpp)

cleanup() {
    if [[ ${cleaned} -eq 1 ]]; then
        return
    fi
    cleaned=1
    set +e
    for ((index=${#pids[@]}-1; index>=0; index--)); do
        kill -INT "${pids[index]}" 2>/dev/null || true
    done
    sleep 2
    for ((index=${#pids[@]}-1; index>=0; index--)); do
        kill -TERM "${pids[index]}" 2>/dev/null || true
        wait "${pids[index]}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

echo "RESULT_DIR=${result_dir}"
echo "Starting isolated ROS1 master on port 11321..."
"${ros1_env[@]}" bash --noprofile --norc -c 'source /opt/ros/noetic/setup.bash; exec roscore -p 11321' >"${result_dir}/roscore.log" 2>&1 &
pids+=("$!")

master_ready=0
for _ in {1..30}; do
    if "${ros1_env[@]}" bash --noprofile --norc -c 'source /opt/ros/noetic/setup.bash; rosparam list >/dev/null 2>&1'; then
        master_ready=1
        break
    fi
    sleep 0.2
done
if [[ ${master_ready} -ne 1 ]]; then
    echo "ERROR: isolated ROS1 master did not start."
    exit 4
fi

echo "Starting ros1_bridge in isolated ROS_DOMAIN_ID=42..."
"${bridge_env[@]}" bash --noprofile --norc -c 'source /opt/ros/noetic/setup.bash; source /opt/ros/foxy/setup.bash; exec ros2 run ros1_bridge dynamic_bridge --bridge-all-topics' >"${result_dir}/bridge.log" 2>&1 &
pids+=("$!")

echo "Starting Point-LIO without RViz..."
"${ros1_env[@]}" bash --noprofile --norc -c 'source /opt/ros/noetic/setup.bash; source "$HOME/aerogo2_slam_ws/devel/setup.bash"; exec roslaunch point_lio_unilidar mapping_unilidar_l1.launch rviz:=false' >"${result_dir}/point_lio.log" 2>&1 &
pids+=("$!")

sleep 4
if ! kill -0 "${pids[1]}" 2>/dev/null; then
    echo "ERROR: ros1_bridge exited early; inspect ${result_dir}/bridge.log"
    exit 5
fi
if ! kill -0 "${pids[2]}" 2>/dev/null; then
    echo "ERROR: Point-LIO exited early; inspect ${result_dir}/point_lio.log"
    exit 5
fi

echo "Recording /pointlio/odom..."
"${ros1_env[@]}" bash --noprofile --norc -c 'source /opt/ros/noetic/setup.bash; exec rostopic echo -p /pointlio/odom' >"${result_dir}/pointlio_odom.csv" 2>"${result_dir}/odom_record.log" &
pids+=("$!")

echo "Playing isolated ROS 2 bag: ${bag_path}"
"${ros2_env[@]}" bash --noprofile --norc -c 'source /opt/ros/foxy/setup.bash; exec ros2 bag play "$1"' _ "${bag_path}" >"${result_dir}/rosbag_play.log" 2>&1
sleep 3
cleanup
trap - EXIT INT TERM

pcd_path="${workspace}/src/point_lio_unilidar/PCD/scans.pcd"
if [[ -f "${pcd_path}" ]]; then
    cp -f "${pcd_path}" "${result_dir}/scans.pcd"
fi

python3 "${script_dir}/03_analyze_pointlio_csv.py" "${result_dir}/pointlio_odom.csv" | tee "${result_dir}/summary.txt"
echo "OFFLINE_TEST=complete"
echo "RESULT_DIR=${result_dir}"
