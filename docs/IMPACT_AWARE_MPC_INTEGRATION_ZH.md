# Impact-Aware 着陆算法：当前实现、参数与安全边界

本文只描述当前合并工程，不记录旧调试流水、废弃设计或固定测试数量。

- 上游基线：`Amagerd1113/AeroGo2` `main`，合并起点 `851146d975184f11cbdcf686451875c103dc2e1e`
- 算法范围：旋翼架完全展开并机械锁定后的下降、触地和接触后恢复
- 当前结论：离线模型、仿真和接口故障注入可运行；尚不具备自动着陆实机条件
- 安全结论：硬件写、LowCmd、正残差和自动着陆门禁必须保持关闭

折展电机不进入着陆控制环。着陆期间要求 F446 停止、四臂完全展开且机械锁定；程序只读固定力臂，不在线计算折展角或力臂。

## 1. 当前能力边界

| 能力 | 当前状态 | 不能误解为 |
|---|---|---|
| 法向一维动力学、标量冲量、地面几何和单调接触 | 已实现，可离线运行且结果永久标记 `hardware_output_permitted=false` | 已闭合真实足力执行链或可发送硬件 |
| 一维法向观测/导纳 | 已实现严格的三模式输入、法向会话绑定和有界状态机；仍是独立离线模块 | 已接入生产高频腿环、已完成牛顿标定或真实力跟踪 |
| AeroGo2 物理先验驱动的完整 NLP | 完整 SLSQP 使用可终止进程、求解后全约束审计，可作硬件禁用的 reference/hybrid 回归 | 已通过目标机实时资格或执行器已辨识 |
| 官方 Go2 URDF 质量、惯量、FK/Jacobian/IK | 哈希固定的离线先验 | 真机安全运动学 |
| SDK 标量足力适配和接触检测 | 已实现严格契约 | 三维 GRF、N 单位力或冲击峰值 |
| LowState observe-only | 已有纯订阅候选实现 | DDS、脚序、零位和实时性已验证 |
| LowCmd 唯一 owner | 接口、仲裁、CRC、限幅、TTL、watchdog、safe-hold 候选实现 | 已获准自由站立/空中执行 |
| 飞控 residual | 主机协议、fake、ACK/CLEAR/回读检查已实现 | Pixhawk 已有真实执行端 |
| 多速率控制 | reference/synthetic 结构已实现 | 已接入 `HardwareWorld` |
| FSM/触地恢复 | ownership 和退出守卫已实现 | 已有硬件 estimator/recovery producer |
| 真机自动着陆 | 代码级阻断 | 改 YAML 即可开启 |

非 DRY-RUN 自动着陆仍返回 `PHASE_NOT_AVAILABLE`；LowCmd 协同执行仍返回 `COORDINATED_ACTUATION_NOT_CONFIGURED`。这是应保留的 fail-closed 行为。

初步法向配置当前使用 schema v4：v4 增加满足指定 C 的一阶矩闭合与惯量复算，旧 v3 因语义不自洽而明确拒绝，v1/v2 仅保留兼容读取。独立的完整 `impact_aware_mpc` demo/template 使用 schema v3：动力学参考点为 C、腿运动学参考点为 B，旋翼力臂从 C 起算。两个 schema 版本号属于不同 loader，不能混用或静默继承。

## 2. B/O/C 与固定展开几何

坐标约定：世界系 ENU、z 向上；Go2 机体系 x 向前、y 向左、z 向上；单位为 m、s、rad、N、N·m、N·s；旋翼控制顺序固定为 `[RR, LF, LR, RF]`。

- `B`：Go2 机身/狗背中心参考点，是腿 FK/IK 的根；
- `O`：旋翼机架旋转中心；
- `C`：完整构型总质心，是刚体动力学位置、速度和惯量的参考点。

当前暂定关系：

```text
p_O^B             = [0, 0, 0.0972] m
p_C^B             = [0, 0, 0.0500] m
p_rotor-plane^O   = [0, 0, 0.0200] m
p_rotor-plane^C   = [0, 0, 0.0672] m
```

即 O 比 B 高 97.2 mm，C 比 B 高 50 mm，旋翼轴平面比 O 高 20 mm；B、O、C 暂在同一 z 轴。97.2/50/20 mm 分别由 YAML 的 `frame_center_O_from_body_origin_B_m`、`total_com_C_from_body_origin_B_m`、`rotor_plane_from_frame_center_O_m` 修改，并非 Python 硬编码；三者目前均是未测量先验，`frame_offsets_identified=false`，实机建模前必须测量。把“旋翼机架本体质心”简化在 O，不表示全部 10 kg 附件都集中在 O。

`B` 是机体固定点；“初始站姿狗背高度”只是 B 的初始世界位置，不是固定常数。若估计器输出 B，动力学必须转换到 C：

\[
p_C^W=p_B^W+R_{WB}p_C^B,\qquad v_C^W=v_B^W+\omega^W\times(R_{WB}p_C^B).
\]

`math_utils.py` 和多速率 sample 已显式区分 B/C：腿运动学以 B 为根，刚体动力学以 C 为参考。不能再用同一个含糊的 `body_position` 表示两者。

水平半径 0.665 m，`a=0.665/sqrt(2)=0.470226009 m`。首个几何旋翼位于 `+x,+y` 象限，即 LF=45°；周向每 90° 为 `LF→LR→RR→RF`。相对 C 的固定臂为：

| 控制项 | 方位角 | `p_rotor^C=[x,y,z]`，m |
|---|---:|---|
| RR | 225° | `[-0.470226009,-0.470226009,+0.0672]` |
| LF | 45° | `[+0.470226009,+0.470226009,+0.0672]` |
| LR | 135° | `[-0.470226009,+0.470226009,+0.0672]` |
| RF | 315° | `[+0.470226009,-0.470226009,+0.0672]` |

四个推力轴暂定 `[0,0,1]`。所有可编辑物理先验只放在 `configs/impact_aware_preliminary.yaml`；Python 仅校验并构造只读对象。

## 3. 官方 Go2 URDF 先验

工程内置 `configs/go2_description.unitree_ros.urdf`，运行时不联网读取 `master`：

| 身份 | 固定值 |
|---|---|
| 官方仓库 | `unitreerobotics/unitree_ros` |
| 文件提交 | `a3b70cae6fd4a82c0e1ece633d5c6f97e88c9d76` |
| 上游原始 SHA-256 | `7d19fe48e2e689ee1a032ab99f2a4a8b671d87e73de48d3e65811682a5b48b9e` |
| 工程副本 SHA-256 | `8f4571b49f35ce04b8833d561c403bddeb9cd8f7077e2ddbee82895726de487c` |
| 根 link / 质量合计 | `base` / `16.087 kg` |

副本只比上游原始字节多一个末尾换行。loader 每次校验副本 hash、robot 名称、树结构、惯性和关节限制，并拒绝 DTD/entity。

参考站姿来自 Unitree MuJoCo 提交 `4134cb5dc7ff1ba7f484deda48b5274b58694519`：SDK 腿序 `FR,FL,RR,RL`，每腿 `[hip,thigh,calf]=[0,0.9,-1.8] rad`。URDF 含 33 个 inertial 元素（其中 2 个零质量）；按该姿态合成全部正质量项得到：

```text
Go2 CoM 相对 URDF base = [-0.001693361, 0, -0.017589231] m
I_Go2@Go2-CoM, base axes =
[[ 0.165562021,  0.000121660, -0.015578542],
 [ 0.000121660,  0.501245124, -0.000031200],
 [-0.015578542, -0.000031200,  0.562125768]] kg·m²
```

当前暂把 URDF `base` 与 B 重合；`body_origin_B_alignment_identified=false`。如果 B 指机身上表面而非 URDF 根，必须实测并修改 `urdf_root_from_body_origin_B_m`。URDF 是仿真先验，不是这台真机的质量/惯量校准证书，也不能直接当 LowCmd 安全限位。

`go2_kinematics.py` 从同一 hash 固定 URDF 读取腿链、轴、原点和限制，提供离线 FK、Jacobian、有界数值 IK及前向代回校验。IK 可显式接收 `preferred_q_rad`（正常应为实测 q 或上一条已确认命令），枚举并复核全部收敛候选后选择离该种子最近的分支，避免沿连续笛卡尔轨迹无故跳支；不提供种子时为兼容旧调用而使用固定 home。种子只用于分支选择，仍须满足 URDF 关节界，不能把不可达或奇异目标变成可行。它明确标记 `OFFLINE_PRIOR_ONLY=True`、`HARDWARE_VALIDATED=False`，当前不接生产 runtime；连续性单元测试不等于实体支链、零位和限位已验证。多 seed 数值 IK 的 aarch64 高频 WCET 也未知，上机前必须测量并满足周期预算，否则应改成经验证的闭式 IK/有界实现。

## 4. 质量、质心和惯量

| 项目 | 当前值 | 性质 |
|---|---:|---|
| Go2 URDF 模型 | 16.087 kg | 官方仿真先验 |
| 旋翼、机架、电池、飞控、安装件、线束等 | 10.000 kg | 用户暂定合计 |
| 整机 | 26.087 kg | 暂定计算值 |
| 整机 C 相对 B | `[0,0,+0.050] m` | 用户暂定 |

水平悬停平均每轴推力约 `26.087×9.80665/4=63.96 N`，只用于离线量级检查，不是 Pixhawk 油门、PWM 或允许发送的命令。

现有 STEP/BOM 没有完整材料密度、部件质量、局部 CoM/惯量和装配误差，所以正式 `inertia.nominal_body_kg_m2` 仍为 `null`。离线模型另设隔离的工程先验：暂取 `p_URDF-root^B=0`，Go2 URDF CoM 为 `[-0.001693361041,0,-0.017589231219] m`；4 套 X8 各 1.095 kg，并把每套暂作位于半径 0.665 m、`z_B=0.1172 m` 的点质量。X8 整套自身 CoM 和本征惯量尚未获得，必须随完整 CAD/BOM 替换。

为同时满足总质量 26.087 kg 和用户给定 `p_C^B=[0,0,0.05] m`，剩余 5.62 kg 合并质量的一阶矩平衡等效 CoM 必须为 `[0.004847170654,0,0.191098213990] m`。这不是“机架本体 CoM”：机架本体仍简化在 O，但其质量尚未从电池、飞控、安装件和线束中拆出。若把全部 5.62 kg 错放在 O，复算 `z_C` 只有约 29.77 mm，与用户给定值不自洽；loader 会严格复算并拒绝此类配置。

惯量下界将剩余质量集中在上述等效 CoM；上界保持同一一阶矩、再在 xy 平面按 0.665 m 半径对称分布；名义取两者中点：

```text
离线名义 I_C =
[[ 1.960517510297,  0.000121660000, -0.021263416909],
 [ 0.000121660000,  2.296378784538, -0.000031200000],
 [-0.021263416909, -0.000031200000,  3.741901685208]] kg·m²
对角下界 = [1.339191385297, 1.675052659538, 2.499249435208] kg·m²
对角上界 = [2.581843635297, 2.917704909538, 4.984553935208] kg·m²
```

该区间不是统计置信区间，交叉项无界，也未证明暂定 C 与实际质量分布一致。代码标记 `PROVISIONAL_OFFLINE_ONLY`、`hardware_use_prohibited=true`；严禁复制到生产惯量或据此解锁硬件。最终需用完整 CAD/BOM、固定 12 关节姿态和平行轴定理合成，并传播质量、位置、姿态和材料不确定度。

参数维护分为两层：原始值是称量/测量或明确暂定的 Go2/附加质量、`p_URDF-root^B`、B/O/C、旋翼面/O、半径、方位角和 X8 质量；派生值是总质量、C/O、旋翼面/C、四个力臂、剩余质量等效 CoM，以及惯量下界/上界/名义值。派生值保存在同一 YAML 是为了可审查 diff，并不是第二组可独立调参的物理输入。

修改任一原始质量或几何量后，运行 `scripts/recompute_impact_aware_preliminary.py`，由程序按一阶矩和平行轴定理成组重算全部派生值，再运行 loader、测试和 `scripts/report_impact_aware_parameters.py`。例如修改 20 mm 会同时改变 X8 的 `z_B`、旋翼面/C、四臂 z 分量、剩余质量等效 CoM 和惯量；只改其中一项会被一致性检查拒绝，不能通过放宽容差绕过。重算工具默认只在 stdout 给出候选；`--output` 也只允许在原配置同目录创建一个不存在的新文件，拒绝覆盖、原地写和跨目录写。生成文件不保留原文件的逐行中文注释，建议人工审阅 diff 后把派生字段回填到带注释的主文件。

`aerogo2_offline.py` 现将这里的 26.087 kg、离线惯量和固定臂注入完整 MPC 数据结构，并可完成一次全 horizon 无接触悬停 NLP。它保留 synthetic fixture 的旋翼时间常数/变化率、气动偏航系数、接触上限、MPC 权重和导纳参数；未知电池时采用 12S/14S 厂家静态表中较低的 100% 端点 `147.1586 N/轴` 作为纯离线可行性上界，`κ` 固定为 0。profile 为 `aerogo2_provisional_offline_hybrid`，所有硬件输出永久为 false；该通过只证明数学问题在暂定物理量下可求解。

## 5. 简化 impact 模型与足力

缺少三轴足力时，当前实机研究主线只保留 C 的竖直运动与标量非弹性 reset：

\[
\dot z=v_z,\quad m\dot v_z=F_{rotor,z}+\sum_jF_{z,j}-mg,
\quad v_z^+=v_z^-+\frac{\sum_jJ_{z,j}}m,
\quad J_{stop}=m\max(0,-v_z^-).
\]

`J_stop` 是预测量，不是 SDK 实测冲量、峰值力或四脚分配。连续接触积分与瞬时 reset 不能在同一时段重复计入。模型不含切向力/冲量、摩擦锥、无滑移约束、角速度冲击 reset 或三维冲击峰值；只有飞控保持经批准的小倾角时，轴向推力才可近似为 z 向力。

`normal_only_mpc.py` 是这条主线的独立离线实现，不是把三维模型的摩擦系数设成零。完整 `ImpactAwareMPCProblem` 的 `landing_contact_geometry` 现在也是必填项，不存在默认水平地面；即使当前预测域尚未安排触地，也必须由调用方明确提供地面平面和守卫参数。模型显式检查地面高度、各足 signed distance、全预测域非穿透、触地位置容差、触地前最小下降速度、接触后的法向 sticking、每脚只允许 `0→1` 的单调接触表，以及旋翼/法向足力/冲量/平均冲击力/变量边界。为消除“一条总法向方程任意分配四个执行量”的欠定性，每个时刻必须由调用方分别给出 `rotor_force_allocation`、`contact_force_allocation` 和 `normal_impulse_allocation`：分配系数非负，非活动通道严格为零，活动通道之和严格为 1，并作为三组独立等式固定。AeroGo2 离线算例暂用四轴/四足各 `1/4`，这只是对称离线假设，不是由实体载荷分配辨识得到的结论。其足力输出字段是 `desired_contact_normal_forces_n`：这只是待跟踪的期望法向力，不能直接送入 Go2，也不能由未标定 counts 反证已经施加。

LowState 的 `foot_force[4]` 与 `foot_force_est[4]` 都是 `int16` 标量；字段名可理解为设备提供的足力通道与估计通道，但公开 SDK 不足以证明各自算法、脚序、工程单位或带宽，也没有切向分量。代码不根据名字猜测物理语义，必须显式选择一组并保留 `sdk_counts`：

```text
counts -> 时序/饱和检查 -> 脚序/符号映射 -> 滤波 -> 双阈值/驻留 -> contact boolean
```

不做 N 标定仍可研究“预测下降速度 + 接触事件切换”的简化 impact-aware 策略，但不能声称测得足力、冲量或峰值改善百分比。普通静载秤以后可做法向低频标定；它仍不能验证高带宽冲击峰值或切向力。

`normal_admittance.py` 另行冻结了三种互斥观测模式：

| 模式 | 可做什么 | 明确禁止 |
|---|---|---|
| `CONTACT_EVENT_ONLY_COUNTS` | 保留“仅接触事件”的类型身份；无接触时可触发状态 `RESET/FREEZE` | 接触为真时推进 N 单位导纳；该情况会被拒绝，counts 应留在外部接触检测链 |
| `CALIBRATED_NORMAL_ONLY_N` | 使用经本机标定的单个法向力标量 | 同时夹带 counts 或三维力 |
| `INDEPENDENT_3D_WORLD_N` | 把独立世界系三维力投影到指定单位地面法向 | 把切向分量变成切向修正 |

控制器输出的位置和速度修正严格平行于地面法向，采用正 `stance_stiffness_n_per_m`、力误差死区、后向欧拉、有界位置/速度、接触丢失 `RESET/FREEZE` 和 preview/commit/abort 事务。第一次**有效接触 transition 被 commit** 时绑定本会话单位地面法向；后续不同法向会被拒绝，只有 `reset()` 同时清除积分状态、旧 transition 和方向身份后才能重新绑定。若从非零/已接触历史状态恢复，则构造时必须显式给出 `initial_ground_normal_world`。无接触 preview/abort 不得偷换或建立方向身份。

这仍没有闭合物理执行链：该一维导纳当前未组装进 production `HardwareWorld`，`multirate.py` 只复用观测模式契约，其现有高频控制仍使用三维 `LegAdmittanceController`。即使以后接入，一维链也必须验证 `期望法向力→导纳修正→IK/LowCmd→地面→实际法向力` 的跟踪误差；未标定 counts 只能走接触事件路径，不能用来驱动 N 单位导纳。

## 6. 多速率、LowCmd 与飞控 residual

生产目标必须拆为：

```text
高频：LowState -> contact -> 法向恢复/导纳 -> FK/IK/限幅 -> 唯一 LowCmd writer
低频：一致状态快照 -> 异步有界求解 -> latest-only policy
飞控：原生高速基线 + 最新未过期 residual
安全：age/ACK/watchdog/limit/manual override -> FC CLEAR + Go2 safe-hold/revoke
```

snapshot、policy、LowCmd frame、FC tick、ownership/contact generation 必须独立编号。迟到、过期或 generation 不符的结果应丢弃，solver 完成不能替旧快照重置 TTL。完整三维 `nlp.py` 已把原生 SLSQP 放进独立 `spawn` 进程；父进程在收到正常、有限且结构合法的候选后，重新审计等式、不等式、`variable_bound_residual()` 和端到端实际耗时。超时后先 terminate/join，再在必要时 kill/join；工作进程无法停止会显式返回 `termination_failure`，不得把迟到可行解判为成功。超时、通信错误或异常退出走“不可信失败结果”，父进程不再重新调用可能无界耗时的目标函数/约束审计。这里的“硬超时”只指主机能够按 deadline 终止隔离的 SLSQP worker；`process.start()` 和成功结果返回后的父进程 post-audit 本身仍不在可杀子进程中，故整个 Python API 的 wall time 仍可能越过 `timeout_s`，但越时结果会被强制标为 `timeout/success=false` 且不提供 `first_input`。这不等于求解器已有真实时资格。SLSQP 仍只是 reference/shadow solver，尚未在最终 aarch64 完成 WCET、抖动、进程启动、父进程审计、BLAS 线程和资源隔离资格，因此不得进入固定周期生产链。法向一维 `normal_only_mpc.py` 目前仍在进程内调用原生求解器，deadline 只在求解返回后的完整审计阶段判定；它通过永久硬件禁用隔离，若将来要在线运行，必须同样移入可终止进程或替换成有确定时间上界的求解器。

### 6.1 LowState observe-only 与 LowCmd owner

`observe_only_enabled` 与 LowCmd `enabled` 分离。observe-only 只要求 LowState topic、最大 age、12 关节名称/motor ID/方向/零位和 mapping version/hash；它仅创建 subscriber，不创建 `rt/lowcmd` publisher，也不能 acquire ownership。

LowCmd 候选生命周期：

```text
HIGH_LEVEL_JOINT_LOCK -> LOWCMD_ACQUIRING -> LOWCMD_SAFE_HOLD -> LOWCMD_ACTIVE
-> LOWCMD_SAFE_HOLD -> HIGH_LEVEL_REACQUIRING -> HIGH_LEVEL_JOINT_LOCK
```

激活后检查 owner、`publisher_active`、writer/watchdog、LowState age、mapping 和关节跟踪，而不是持续要求 SportMode `mode=6`。target 过期时由同一 writer 进入已验证 safe-hold，不能在空中简单停发。writer 现在会在命令构造后、调用 DDS `Write` 前再次检查 TTL；如果阻塞式 `Write` 返回时目标已经过期，立即进入 fault/safe-hold，且绝不把该帧登记成有效 target ACK。算法腿序固定为 SDK 顺序 `[FR,FL,RR,RL]`，策略域、快照、高频采样和控制器必须携带同一腿序与准确 mapping hash，不能只检查 hash 非空。

LowCmd 的确认层级必须按下面的名字解释，禁止把较弱证据升级命名：

1. `submit()` 成功只表示容量为 1 的 owner mailbox 已接收/替换目标；
2. `writer_enqueue_generation`、`writer_enqueued_target_sequence` 和 `writer_enqueued_q_rad` 表示固定周期 writer 对同一序号完成软件位置、变化率和力矩包络限幅，并由 DDS `Write` 接收；它不是电机驱动器应用 ACK，也不是 LowState 实测关节角；
3. `actuator_applied_target_sequence` 只有目标侧真实提供 application ACK 时才允许出现，当前能力为 false。

`writer_enqueue_ack_available` 能力在 submit 前、等待期间和结果审计时必须保持稳定；能力翻转、`publisher_active=false`、generation 未前进、序号不符或 12 维限幅后 q 缺失都会 fail closed。同一 target sequence 的后续 writer 周期冻结第一次入队的 `writer_enqueued_q_rad`，不会继续向 mailbox 原始目标爬升；必须发布新 sequence 才允许命令继续变化。高频导纳只能用同 sequence 的 `writer_enqueued_q_rad` 做主机侧 anti-windup，不能把它表述为“实际应用关节角”。

safe-hold 的成功证据也是因果性的：revoke 后目标身份必须从 legacy/mailbox/writer/application 四层全部清空，`safe_hold_write_generation >= safe_hold_request_generation > 0`，并由该次写入之后的新鲜 LowState 证明 q/dq 已稳定；更早或仍在飞行中的 callback 不能结算本次 safe-hold。任何超时或证据不完整都保持 owner/fault，不得假定已经安全交还。已经进入 DDS 的过期帧无法由主机撤回，因此机器人端命令租约/watchdog 仍是强制实机门禁。当前 arbiter 没有生产 `network_exclusivity_verifier` 和运行期间持续 DDS publisher 监测，所以 acquire 故意拒绝；不得绕过。

### 6.2 Flight-controller residual

协议固定为：

\[
\Delta u_{applied}=\kappa\Delta u_{raw},\qquad u_{final}=u_{fc}+\Delta u_{applied}.
\]

`Delta_u_applied` 已乘一次 κ，单位 N、顺序 `[RR,LF,LR,RF]`。飞控只能在同一 mixer tick 叠加一次；请求须含 session/epoch、baseline version、target tick、sequence、timestamp、TTL，并返回 ACK、执行值、headroom 和饱和原因。超时/失联时 residual 原子归零，飞控自身基线继续工作。

不存在可从文献复制的本机 ACK/TTL/baseline tick：它们是自定义固件、链路和控制周期的接口契约，必须由同伴实现并测量。当前无真实执行端，实体 κ 只能为 0。

油门/PWM/RPM→推力可以只在飞控内辨识，前提是飞控向上层提供可信的每桨 N 单位 baseline/residual、正负 headroom、执行回读和 CLEAR。否则论文及上层模型必须包含 `T_i=g_i(command,V,...)` 的不确定映射，不能把 PX4 归一化 thrust 当 N。

### 6.3 跨设备一致性的当前边界

当前多速率代码已统一 session/epoch、snapshot/policy/contact/config/model identity、腿序和 TTL；对 FC 激活、mailbox 发布与 LowCmd writer evidence 任一失败都会 fail closed。它也不再把 LowCmd mailbox ACK 冒充电机 ACK，并只用软件限幅后且已由 DDS writer 入队的 `writer_enqueued_q_rad` 回灌导纳状态。首次高频 LowState 还必须把 12 个实测关节角与每腿控制器的 `previous_joint_command` 逐项比较；`initial_joint_alignment_tolerance_rad` 是必填的非负资格参数，代码没有生产默认值，超差时在控制器初始化和命令发送前终止。

但 Go2 与飞控尚没有共同未来生效时刻、两阶段 commit 或事务栅栏。当前顺序仍是先完成 FC residual 激活，再发布对应腿策略，因此存在“新 residual、旧腿策略”的时间窗；FC ACK、DDS `Write` 和 motor application 也分别是不同层级。只要此跨设备原子性未由真实协议闭合，正 κ 与 LowCmd 协同执行就必须保持门禁关闭。机器人端、飞控端本地 watchdog，以及独立于 Python/event loop 的 OS watchdog，也仍需真机实现和故障注入验收。

`multirate.py` 当前固定报告以下七个、名称和数量都不可弱化的硬件阻断项：

1. `cross_device_activation_transaction_unverified`：Go2/FC 跨设备激活事务未验证；
2. `go2_motor_side_application_ack_unavailable`：没有 Go2 电机侧应用确认；
3. `continuous_dds_owner_monitor_unavailable`：没有运行期持续 DDS 唯一发布者监测；
4. `independent_supervisor_watchdog_unverified`：独立监督 watchdog 未验证；
5. `production_atomic_force_sample_unavailable`：没有生产级原子足力/状态 sample；
6. `calibrated_normal_force_pipeline_unverified`：经标定的法向 N 管线未验证；
7. `normal_force_tracking_error_unvalidated`：法向期望力到实体响应的跟踪误差未验证。

这七项是 `MultiRateExecutionConfig` 硬件模式的显式门禁；求解器实时资格、运动学/映射、地面估计和现场风险批准仍是额外前置条件。即使未来把这七项逐条实现，也不能自动把当前离线对象变成真机许可。

## 7. 当前文件职责

| 文件 | 职责 |
|---|---|
| `configs/impact_aware_preliminary.yaml` | B/O/C、质量、固定臂、URDF、离线惯量及待确认项的唯一入口 |
| `preliminary.py` | preliminary schema v4 严格加载、一阶矩/惯量复算、法向模型和永久硬件拒绝 |
| `go2_urdf.py` / `go2_kinematics.py` | 固定 URDF 的质量合成与离线四腿 FK/Jacobian/IK；显式输出 B 到足端位置并用 `r_CF=r_BF-r_BC` 转成 C 力臂 |
| `aerogo2_offline.py` | 把 preliminary 物理先验注入完整 NLP；列明仍借用的 synthetic 数值并永久禁用硬件 |
| `normal_only_mpc.py` / `aerogo2_normal_only.py` | 法向一维着陆问题、三类固定分配、AeroGo2 对称暂定算例、完整约束审计和永久硬件输出拒绝 |
| `normal_admittance.py` | 三种互斥力观测语义、仅沿地面法向的有界导纳、法向会话绑定/reset；独立离线模块，尚未组装到 production 高频链 |
| `recompute_impact_aware_preliminary.py` | 修改原始质量/几何后生成经 strict loader 自检的新候选，不覆盖主配置 |
| `math_utils.py` | B→C 位置/速度刚体变换 |
| `contact_detection.py` / `go2_foot_force.py` | counts 契约与接触事件 |
| `types.py`、`config.py`、`dynamics.py`、`impact.py`、`nlp.py` | `FootPositionsFromBodyOriginB`、`FootLeverArmsFromComBody`/`Horizon` 显式 B/C 与腿序类型，完整三维 synthetic/reference 模型、地面/触地/倾角约束及可终止 SLSQP |
| `coordinator.py` / `multirate.py` | 无硬件权限的协调与多速率参考调度 |
| `integration.py` / `executor.py` | Go2/FC DTO、协议和已获权 owner 的提交规则 |
| `bridges/go2_lowlevel_sdk_bridge.py` | LowState 纯订阅及 LowCmd 唯一 writer 候选 |
| `bridges/go2_control_arbiter.py` | SportClient/LowCmd 互斥、epoch、进程锁 |
| `system_manager.py` / `safety_monitor.py` | ownership FSM、恢复退出事务和安全监视 |
| `hardware/runtime.py` | 仍组装 legacy `SafeDescentController`，尚未组装 impact 生产链 |

## 8. 参数责任表

### 8.1 可直接采用的安全/离线默认

| 参数 | 当前值 | 限制 |
|---|---|---|
| 重力 | `9.80665 m/s²` | 可直接使用 |
| B/O/C | 97.2/50/20/67.2 mm | 离线，实机前复核 |
| 臂/方位/顺序 | 0.665 m；LF 45°；`[RR,LF,LR,RF]` | 离线，实物逐轴复核 |
| Go2 URDF/home | hash 固定；16.087 kg；`[0,.9,-1.8]` | 离线先验 |
| 总质量/惯量 | 26.087 kg；第 4 节区间 | 只做初步仿真/敏感性 |
| 推力轴 | 四轴 `[0,0,1]` | 暂定小倾角模型 |
| κ / 硬件开关 | `0` / 全部 `false` | 当前必须保持 |

### 8.2 必须由本机测量或验证

| 参数组 | 必须得到的结果 |
|---|---|
| 质量/CoM/惯量 | 完整构型质量、三轴 C、完整 CAD/BOM 惯量和不确定度 |
| B/URDF/旋翼几何 | B 物理标记、`p_URDF-root^B`、四个 3D 力臂、轴倾角和误差 |
| LowState | topic、age/gap/tick、12 关节 ID/零位/方向、mapping hash |
| 足力 counts | 字段、脚序、符号、噪声、饱和、阈值、滤波、驻留 |
| LowCmd | 软限位、Δq、dq、Kp/Kd、tau、固件 torque clamp、温度、safe-hold |
| 首次关节对齐 | `initial_joint_alignment_tolerance_rad`、每腿初始 `previous_joint_command` 与首帧 LowState 的允许差；代码无生产默认值 |
| 目标机时序 | 周期、WCET、jitter、丢包、阻塞 Write、age/skew、watchdog |
| 推力能力 | 每桨 N 接口或 command/V/RPM→N、headroom、幅值/速率/jerk、失联归零 |

### 8.3 必须向同伴确认

1. PX4/ArduPilot/自定义固件、版本/build hash、airframe/mixer、参数文件 hash；
2. 板卡方向、输出坐标、四元数顺序、IMU 相对 C 的位置；
3. `RR/LF/LR/RF→Motor→物理输出/CAN node`、CW/CCW、桨手性、正推力方向；
4. PWM 或 Cyphal/CAN、输出率、disarmed/min/max、失联行为；
5. 是否已有同 tick N residual；ACK、TTL、baseline、回读、CLEAR、饱和语义；
6. 旋翼电池 12S/14S、低压/功率余量、实装桨型号；
7. LF=45° 是否与机械标注和飞控 mapping 一致；完整 10 kg BOM/CAD revision。

电池和桨可不进入上层 MPC，但影响质量、CoM、惯量、推力/headroom 和构型身份。固件、机架类型与安装方向决定坐标、mixer 和协议；即使由同伴调整，也须以 hash/映射契约交付。ENU/FLU 与 PX4 常见 NED/FRD 的转换必须在唯一适配层完成，不能靠板卡校准“自动消除”。

## 9. 全项目审计结论

已补强：`x8-bench` 只读白名单且未知/危险命令 fail closed；`x8-spin` 独立并要求拆桨、固定、精确确认和幅时上限；B/O/C、固定臂、pinned URDF、部署依赖、observe-only 已显式化；硬件模式拒绝外部伪造 landing estimate；Pixhawk heartbeat 异常关闭；离线 FK/IK 与质量属性共用同一 URDF。

本轮针对求解器、触地几何、足力执行、B/C 参考点、导纳漂移、姿态代价和多设备一致性重新审计。结果不能概括为“没有问题”：其中能由离线代码确定的问题已修复或 fail closed；需要真实传感、DDS/电机、飞控固件和目标机时序证据的问题仍然开放，并继续阻断硬件输出。

### 9.1 本轮七类问题的处理状态

| 问题 | 已完成的代码修复／离线可验证项 | 仍需真机完成；未完成前的门禁 |
|---|---|---|
| 1. 求解器误判成功与超时 | 完整 `nlp.py` 在 SLSQP 正常返回后独立复核等式、不等式、`variable_bound_residual()` 和端到端耗时；原生 SLSQP 位于独立 `spawn` 进程，超时会 terminate/kill，进程仍不退出则报告 `termination_failure`。超时/异常路径不在父进程重新执行可能无界的目标/约束审计，迟到可行解不能成为成功。 | 在 Go2 aarch64 上验证 `spawn`、SciPy/BLAS、CPU/内存隔离、终止可靠性、WCET、抖动和连续负载。当前可杀边界只包住 worker，`process.start()` 与成功后的父进程 post-audit 仍可能使 API wall time 越界；代码会拒绝该结果，但在线版本还需把整条预算做成确定有界。`normal_only_mpc.py` 当前仍是进程内 SLSQP，虽会在返回后判超时且永久禁止硬件输出，但若改成在线控制，必须先采用可终止进程或有确定上界的 production solver。 |
| 2. 缺触地几何约束 | 完整 NLP 增加必填且无默认值的 `landing_contact_geometry`：地面平面 `(n,h)`、世界系 signed distance、全结点非穿透、触地位置容差、触地前最小下降速度、严格小于 90° 的机体倾角锥，并拒绝着陆接触表中的 `1→0`；即使全域暂时无触地也不能省略地面。一维问题也显式检查地面高度、触地守卫和单调接触。 | 必须由真实状态估计器给出相干的 C 位姿/速度、足端世界位置和地面平面；实测选定位置/速度容差及远小于 90° 的运行倾角限值。地面法向或时间戳不可信时不得启用着陆。 |
| 3. MPC 足力与执行链未闭合 | 新增独立法向一维 MPC，明确区分期望法向足力、标量冲量和 contact boolean；旋翼力、接触力和冲量分别由非负、活动项和为 1 的固定 allocation 消除数值欠定。`normal_admittance.py` 以互斥模式拒绝 counts 进入 N 方程，并把输出限制在本会话绑定的法向。 | 这一维链仍未物理闭环，且独立一维导纳尚未组装进 production 高频环。当前无 WBC、`τ=Jᵀf` 或经辨识的法向接触刚度模型；`期望足力→导纳→IK→q/Kp/Kd→地面→实际足力` 尚无证据证明实际力可跟踪。必须在机械支撑和经批准的法向测力条件下辨识/验证；若没有 N 标定或外部参考，只能声称基于接触事件的柔顺着陆。 |
| 4. 足端力臂 B/C 与腿序语义 | 类型已拆成相对 B 的 `FootPositionsFromBodyOriginB`、相对 C 的 `FootLeverArmsFromComBody` 和预测域 `FootLeverArmsFromComBodyHorizon`，按 `{}^B r_CF={}^B r_BF-{}^B r_BC` 显式转换；动力学/冲击/NLP 拒绝未标注数组、B 类型误传和腿序不一致。IK 支持 `preferred_q_rad` 并从收敛候选中选择最近分支。策略域、快照、高频采样和控制器携带固定 SDK 腿序 `[FR,FL,RR,RL]`，mapping hash 必须与期望值完全相等。 | 实测 `p_C^B`、B 与 URDF 根的对齐、12 电机映射及实体脚序；用逐腿小幅运动/FK 回读验证符号、支链、奇异区和连续轨迹。production 必须把实测 q 或可信的上一条命令传为 IK 种子，不能依赖默认 home。恢复切向力前还要验证完整三维力臂。 |
| 5. 导纳长期漂移 | 三维 reference 导纳改为 `(1-η)K+ηK_stance` 且 `K_stance` 正定，并增加死区、修正位置/速度硬界、接触丢失 `reset/freeze`、workspace/IK/关节限幅和 writer 入队 q 的 anti-windup；独立一维导纳使用正驻留刚度、后向欧拉、有界状态及法向会话身份。多腿状态提交先全量验证后原子 commit。 | `K_stance`、M/D/K、deadband、修正界、释放策略和 FK 代回必须结合真实 counts/N 标定、噪声、接触刚度和 LowCmd 限幅调试；未取得同 sequence 的 `writer_enqueued_q_rad` 时高频环必须故障退出。该 q 仍不是电机应用/实测角，不能用它证明实体 anti-windup 或力跟踪。 |
| 6. 180° 姿态代价失效 | 姿态误差改为 SO(3) 主值 Log/geodesic，包含 π 附近的稳定分支，180° 不再得到零代价；着陆几何同时对全预测域施加倾角硬锥且配置必须 `<π/2`。 | 验证飞控/估计器坐标系、四元数顺序、连续性和 IMU-C 偏置；依据支撑试验给出实际 roll/pitch 上限。几何上“小于 90°”只是防止数学退化，不是可接受的实体着陆限值。 |
| 7. 多速率跨设备一致性 | command/policy 携带完整 identity、TTL 与腿序；LowCmd bridge 回报 DDS writer generation 和软件限幅后的 `writer_enqueued_q_rad`，高频环只有取得对应 sequence 的 writer evidence 才提交导纳状态；mailbox ACK 不再冒充电机 ACK。同一 sequence 冻结首次 writer 入队 q；capability、publisher、generation 或序号变化均 fail closed。writer 在构造后重检 TTL；`Write` 跨 TTL 返回会立刻 fault 且不产生 target ACK；safe-hold 要求请求、写入和后续 LowState 的因果证据。 | 仍缺 Go2 与 FC 的共同未来执行时刻或事务栅栏；现有顺序会留下“新 residual＋旧腿策略”窗口。主机无法撤回已经进入 DDS 的过期帧，DDS `Write` 也不是电机应用确认；还缺运行期间持续其他 `rt/lowcmd` publisher 监测、独立 OS watchdog、机器人端命令租约/watchdog 和飞控端 watchdog。上述链路及断网、阻塞、进程崩溃、重启/epoch 变化必须在硬件上验收。 |

X8 诊断脚本固定为 SHA-256 `7987dbf41d17e9c6d9dbd811b9be1fda0eea37c25028def17c4fca2986123dbb`，最大读取 512 KiB。入口在对齐检查和启动前均重新验证；最终通过 `python -` 的标准输入执行刚刚完成 hash 验证的同一字节快照，而不是再按路径打开可能已变化的文件。hash、白名单或映射任一不符都会拒绝执行；这只收窄诊断工具风险，不构成飞行或带桨许可。

### 9.2 P0：集成自动着陆前必须补齐

1. `HardwareWorld`/simulation 仍注入 legacy `SafeDescentController`；`AppConfig` 未组装 impact production 对象；
2. 缺真实同步的 LowState/state-estimator/contact/ground/kinematics input builder，以及硬件 landing estimator 和 post-touchdown recovery evidence producer；
3. 缺闭合并验证的法向执行控制器：至少要定义期望法向力如何变成安全的腿命令，并量化实际响应与 MPC 假设的偏差；
4. 缺有确定时间上界的 production solver；法向一维离线 SLSQP 不能因算例通过就接入实时链；
5. 缺真实 Pixhawk N residual transport/固件，现有仅 host 协议/fake；
6. 缺 Go2/FC 共同生效、两阶段 commit/事务栅栏或经证明的有界补偿协议；
7. 缺持续 DDS publisher 排他监测，以及独立 OS、机器人端和飞控端 watchdog；
8. 离线 FK/IK、B/C、腿序、关节映射/限制、首次关节对齐、连续 IK 分支和 safe-hold 未在机械支撑条件下验证；`initial_joint_alignment_tolerance_rad` 不得凭空采用通用默认值。

### 9.3 P1：执行器试验前的工程资格

- 实现并验证网络级唯一 LowCmd publisher；不仅 acquire 时检查，还要在 ACTIVE/SAFE_HOLD 全程监测其他 `rt/lowcmd` 发布者；当前缺生产 verifier，acquire 故意拒绝；
- 关键 watchdog 独立于 Python/event loop，评估 DDS Write 阻塞、跨 TTL 的已发送帧、进程崩溃和同步日志 flush；机器人端必须按 command lease 自主拒绝/替换过期命令，不能只依赖主机在 `Write` 返回后的补救；
- `ImpactAwareLowCmdExecutor` 目前依赖 owner 的同步 `status()` 和异步 `submit()/revoke()` 自身有界；恶意或失效 stub 永久阻塞时协程也会等待。生产 owner 必须证明这些本地边界不阻塞，或改成可终止 IPC/独立进程，不能仅在 executor 外层加一个会遗留后台写动作的软 timeout；
- 一维 production solver 移到可终止进程或换有界 solver，完整/一维方案都完成目标机 WCET/jitter/负载资格；多 seed 数值 IK 也必须纳入高频 WCET，超预算时替换为经验证的闭式/有界 IK；
- 冻结 NED/FRD↔ENU/FLU、四元数、IMU-COM、时钟同步和 generation；
- 完成故障注入、原始日志、配置/固件 hash、回退和风险评审。

单元测试通过只证明软件契约，不证明模型、机械强度、网络排他性、实时性或飞行稳定性。

## 10. 当前可测、禁测与部署顺序

当前可做：离线/仿真/参数敏感性；fake LowCmd/FC 故障注入；旋翼断电、Go2 机械稳定支撑时的 LowState observe-only；Pixhawk disarmed 的只读审计；全部拆桨且机架可靠固定后的单电机低油门 `x8-spin`。

当前禁止：自由站立/悬空/空中 LowCmd；带桨 motor-test；arm、起飞、悬停、自由落地；任何正 κ residual；集成自动着陆；把 counts 当 N 或把离线惯量当已辨识值。

用户没有力板、动捕、吊架和推力台，测试必须停止在“离线 + 只读 + κ=0 shadow”。若也无可靠固定工装，应跳过 `x8-spin`；不得用手扶代替支撑/系留。

阶段顺序：

1. **离线反例回归**：除正常算例外，必须注入越过变量边界但 `solver.success=true`、求解返回晚于 deadline、触地脚仍悬空、接触 `1→0`、180° 姿态、B/C 类型混传、腿序/mapping hash 错配、无 writer evidence、ACK 丢失和 TTL 过期；所有反例都应 fail closed；
2. **法向一维离线验证**：运行下述 normal-only 脚本，检查非穿透、触地速度、接触 sticking、等式/不等式/变量边界余量以及 `hardware_output_permitted=false`；对质量、C 高度、下降速度、触地步和力/冲量上限做扫参；
3. **aarch64 安装与 shadow**：运行 `deploy/install_aarch64.sh`；仅用记录/只读数据、`κ=0`、不 acquire LowCmd，测 full/normal-only 求解的 WCET 分布、deadline failure、进程创建、CPU/内存压力、jitter、age/skew 和日志影响；任何统计平均值都不能替代约定的最坏时延门限；
4. **HW-RO 与 LowState observe-only**：保持 `hardware_write_enabled=false`，检查 Pixhawk/F446/Go2 freshness、F446 duty=0、飞控 disarmed、ESC=0、网卡/时钟；确认系统没有创建 `rt/lowcmd` publisher，逐腿核对 `[FR,FL,RR,RL]`、motor ID、零位、方向、B→C 变换和两组原始 counts；
5. **机械支撑下的 Go2 owner 资格**：只有先实现持续 DDS publisher 监测和独立 watchdog，且四足重量由可靠支撑承担时，才测试 CheckMode/ReleaseMode、epoch、首次关节对齐、固定周期 Write、软件限幅后的 `writer_enqueued_q_rad`、同 sequence 冻结、safe-hold 因果证据、TTL、阻塞 Write、断网、杀进程和高层重获权。不得自由站立测试，也不得把 DDS Write 或 writer 入队 q 当电机 ACK/实测角；
6. **飞控 disarmed HIL**：电机输出物理禁止时验证 `[RR,LF,LR,RF]`、N 单位契约、baseline version/target tick、ACK/执行回读/headroom、重复包/乱序/过期、失联 CLEAR 和飞控基线保持；没有真实固件端时停在 fake 测试；
7. **跨设备事务故障注入**：先实现共同未来执行时刻或事务栅栏，再在输出禁止状态下注入 FC 先 ACK、腿策略迟到、Go2 writer 迟到、任一端重启/epoch 变化、时钟偏移和 watchdog 触发；证明不会出现可持续的“新 residual＋旧腿策略”；
8. **拆桨映射审计**：有可靠固定工装才逐轴核对 Motor/ESC/位置/旋向并运行受限 `x8-spin`；当前代码只允许 `rr/lf/lr/rf` 单臂并拒绝 `all`。这只能验证映射，不能验证 N 推力模型或带桨稳定性；
9. **强制停止点**：在取得机械支撑/系留、真实 residual、网络排他、法向力/推力辨识、P0/P1 验收和独立现场批准之前，不进入 LowCmd 承重、带桨、正 κ、悬停或着陆。若没有支撑/系留和经批准的测力手段，项目必须停在 shadow；不能用算法输出替代真实受力证据。

部署已固定的候选仅包括：CycloneDDS `9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd`、Unitree SDK2 Python `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5`、aarch64 `lxml==5.3.2`。安装器拒绝 vendor 源码中的 tracked/untracked 污染和 submodule；Cyclone 构建目录及 Unitree SDK 的 `git archive` 安装快照均置于源码树外。已有 YAML 与新版本不同时保留原件并写 `.dist` 候选。其余 Python 传递依赖、系统包、Python/Ubuntu/kernel 组合仍未形成带文件 hash 的完整目标机锁；当前只能复现上述三项身份，不能声称整个环境可复现或具有实时/固件资格。完成全量 lock 以前 impact 自动着陆和 LowCmd/正 residual 实机执行继续禁用；以后完成 lock 也仍需独立硬件资格测试。

本地回归：

```bash
python -m pip install -e ".[mpc,dev]"
python -m ruff check src tests scripts
python -m mypy src
python -m pytest -q
python scripts/recompute_impact_aware_preliminary.py --config configs/impact_aware_preliminary.yaml
python scripts/validate_aerogo2_normal_only_landing.py --output output/aerogo2_normal_only_landing_validation.json
python scripts/validate_impact_aware_mpc.py --output output/impact_aware_mpc_validation.json
python scripts/validate_aerogo2_offline_prior.py --output output/aerogo2_offline_prior_validation.json
```

`validate_aerogo2_normal_only_landing.py` 生成的 JSON 至少要满足：总体 `pass=true`、`solver.success=true`、等式/不等式/变量边界审计通过，并且顶层 `hardware_output_permitted` 必须严格为 `false`。该报告只证明暂定参数下的离线一致性；不得把其中的 N、冲量或第一步命令复制到硬件发送路径。

文档不固化测试数量和耗时，每次合并以当次日志为准。

## 11. 论文必须同步修改

1. 将近期实验模型写为法向一维竖直着陆，不再声称现有 Go2 标量通道支持三维 GRF、切向冲量、摩擦锥或三维峰值；三维 6-DoF 问题只能标为离线 reference/未来扩展；
2. 明确 `foot_force/foot_force_est` 未标定时只产生接触事件。MPC 的 `desired_contact_normal_forces_n` 是期望量；在没有 WBC、`τ=Jᵀf`、已辨识接触模型或真实误差验证时，不能假定 `f_actual≈f_MPC`，也不能报告 N、实测冲量或峰值改善；
3. 区分 B/O/C；0.665 m 是 O 到轴的水平半径，FK 输出相对 B，而动力学必须使用 `r_CF=r_BF-r_BC`。同时在论文固定脚序 `[FR,FL,RR,RL]` 与旋翼控制顺序 `[RR,LF,LR,RF]`，二者不可混写；
4. 把地面平面 `(n,h)`、signed distance、全域非穿透、触地位置容差、最小下降速度、接触 `0→1` 单调性和经实验确定的小倾角硬边界写入优化问题。impact reset 应写成理想标量非弹性预测，并列出忽略项；
5. 姿态代价应采用 SO(3) Log/geodesic 误差，说明 π 附近分支；不能继续使用在 180° 返回零的 `vee(skew(R_d^TR))` 作为全局误差；
6. 分开写三维 reference 导纳和一维法向导纳：三维律为 `K_eff=(1-η)K+ηK_stance` 且 `K_stance` 正定；一维实现没有 η 混合项，直接使用正 `stance_stiffness_n_per_m`。两者都要写清死区、位置/速度界、接触丢失 reset/freeze 和事务更新；当前所谓下游 anti-windup 只能使用同 sequence 的 `writer_enqueued_q_rad`，不能写成电机实际应用角。旧的 `η=1` 时零刚度公式应删除或仅作为被否决方案；
7. 分开 MPC 总推力、飞控基线与 residual，κ 只乘 residual 一次；无 N 接口时加入 `T=g(command,V,...)`，不能把归一化油门当 N；
8. 完整 SLSQP 可描述为“独立可终止进程＋求解后等式/不等式/变量边界/耗时审计”，但仍只能称 reference/shadow；一维进程内 SLSQP 更不能称实时。只有目标机 WCET/抖动和 production solver 资格完成后才可使用 real-time 表述；
9. 多速率实现应区分 mailbox 接收、DDS writer enqueue 和电机应用，并明确当前没有跨 Go2/FC 原子提交。加入 ownership、`publisher_active`、capability 稳定性、TTL、writer generation、同序号冻结、safe-hold 因果证据、ACK/CLEAR、共同生效时刻、独立 watchdog 和触地后恢复退出；
10. 区分 synthetic、回放、shadow、disarmed HIL、拆桨映射和实体闭环证据；当前条件下不能给出实体 impact reduction、足力跟踪或自由着陆结论。

## 12. 解除门禁的最低条件

- [ ] 质量、C、惯量、固定几何和不确定度绑定完整 BOM/revision；
- [ ] 地面平面、状态估计、B→C、脚序、Go2 mapping、counts、运动学、限制和 safe-hold 在可靠支撑下验证；
- [ ] 法向期望力到真实响应的模型/控制/误差边界已闭合；若无法测量，则收窄为不声称力跟踪的接触事件控制；
- [ ] 进程、服务、DDS 网络唯一 LowCmd publisher 已证明并在 ACTIVE/SAFE_HOLD 期间持续监测；
- [ ] 飞控实现同 tick N residual、TTL、ACK/回读、headroom、CLEAR；
- [ ] production runtime 组装原子 sample、多速率控制、硬件 estimator/recovery；
- [ ] Go2/FC 共同生效/同步补偿和 OS/机器人/飞控三层 watchdog 通过阻塞、网断、崩溃、重启测试；
- [ ] 最终 aarch64 通过 WCET/jitter/age/skew/丢包及分阶段风险评审；
- [ ] 论文声称与真实传感、接口和实验能力一致。

清单完成也不自动授权飞行，仍需独立代码审查和现场批准。在此前保留 `hardware_write_enabled=false`、`go2.low_level.enabled=false`、`allow_hardware_output=false`、`κ=0` 及所有代码级阻断。

## 13. 当前 project 节点图

![AeroGo2 当前工程节点与控制流图](AEROGO2_CURRENT_PROJECT_GRAPH_ZH.png)

高清矢量版为 [AEROGO2_CURRENT_PROJECT_GRAPH_ZH.svg](AEROGO2_CURRENT_PROJECT_GRAPH_ZH.svg)，可编辑源文件为 [AEROGO2_CURRENT_PROJECT_GRAPH_ZH.dot](AEROGO2_CURRENT_PROJECT_GRAPH_ZH.dot)。图是按当前 `SystemState`、`ALLOWED_TRANSITIONS`、控制权状态、LowCmd 桥、多速率控制和安全监视器重新绘制的，不是对旧图凭经验补线。

图中四个区域分别表示：

1. 运行时和外部设备：操作员/RC、Pixhawk、F446、Go2 经过各自 bridge 汇入唯一 `SystemManager`；
2. 顶层 `SystemState`：负责形态、飞行、着陆、触地确认、地面交还和故障；论文的三个触地阶段不是新的顶层状态；
3. 独立 Go2 控制权：高层 JOINT_LOCK、LowCmd 获取、激活、安全持姿和高层重获权，与顶层状态正交；
4. Impact-aware 多速率数据流：低频 MPC、高频腿环、飞控原生高速环和独立安全环，并明确画出尚未接通的跨设备提交器与真机门禁。

相对旧图，关键变化如下：

- 在 `TOUCHDOWN_VERIFY` 与 `FLIGHT_READY` 之间增加 `GO2_GROUND_HANDOVER`，表达“关闭 LowCmd endpoint 后等待新的高层 JOINT_LOCK”的事务；
- `AUTO_LANDING` 内部增加 `PRE_TOUCHDOWN → TOUCHDOWN → POST_TOUCHDOWN_RECOVERY`，首次触地不再等于退出控制器；
- LowCmd 激活后不再要求旧 `SportModeState.mode=6`，而检查 owner epoch、writer/watchdog、LowState 和关节误差；
- 腿与旋翼不再被画成一个单速率输出：腿侧必须经 LowState/接触/导纳/IK/LowCmd，旋翼侧必须经 baseline reservation/residual/ACK/执行回读；
- 图中红色虚线不是“可选优化”，而是当前真机自动着陆仍被阻断的缺失链路。

## 14. 相对同伴 GitHub `main` 的修改

合并基线固定为提交 `851146d975184f11cbdcf686451875c103dc2e1e`（当时 release 0.3.14）。本地目录没有 `.git`，因此复核方法是取得该 SHA 的固定 tree，对相对路径和 SHA-256 逐文件比较，而不是把会继续变化的 `main`、文件时间或本文中过期的文件数量当证据。

本文不再维护“修改/新增/相同文件数”统计：算法和安全修复继续演进后，这类数字会立即失真。准确表述是：合并时保留同伴工程，算法数学主体主要放在独立目录；但控制权和安全状态必须端到端传播，所以 runtime、配置、状态机、安全监视和 bridge/CLI 接缝存在必要修改。不能描述成“仅新增算法文件”“零侵入”，也不能用旧统计证明当前目录仍与基线一致。每次准备提交时应重新运行基于上述 SHA 的内容比较，并把机器生成的 diff 清单作为该次审查附件。

### 14.1 新增内容

| 类别 | 文件/范围 | 作用 |
|---|---|---|
| 算法主体 | `src/aerogo2/landing/impact_aware/*.py` | 固定旋翼模型、完整三维 reference NLP、法向简化模型、接触、导纳、IK 接缝、residual、coordinator 与多速率结构；以当前 tree 为准，不在文档固化文件数 |
| Go2/飞控接口 | `go2_control_arbiter.py`、`go2_lowlevel_interface.py`、`go2_lowlevel_sdk_bridge.py`、`fake_flight_controller_residual.py` | 唯一 LowCmd owner、CRC/限幅/TTL/watchdog/safe-hold 和可测试的飞控 residual 主机协议 |
| 安全辅助 | `pixhawk_freshness.py`、`signal_shutdown.py`、`go2_lowcmd.py` | 多源时间一致性、owner 未安全交还时进程驻留、受限 CLI |
| 参数与模型 | 3 个 Impact-aware YAML、固定 Go2 URDF 与许可证 | 暂定 B/O/C、固定力臂、质量/惯量先验、synthetic 与 hardware 配置隔离 |
| 工具 | `scripts/*impact_aware*` 和 κ 检查脚本 | 派生参数重算、配置报告和离线求解验证 |
| 验证 | 专项单元测试与离线验证脚本 | 边界、单位、epoch、TTL、故障注入、恢复退出和模型回归 |
| 图与说明 | 本文以及 PNG/SVG/DOT | 保存当前实现真值和可编辑节点图 |

### 14.2 修改同伴文件的范围

| 层 | 主要文件 | 为什么必须修改 |
|---|---|---|
| 数据/配置 | `common/{enums,models,config}.py`、默认和站点 YAML | 新增 `GO2_GROUND_HANDOVER`、独立控制权、LowState 电机/足力反馈、LowCmd 限值和恢复证据；严格拒绝未知/重复配置 |
| 运行时 | `hardware/runtime.py`、`main.py`、`simulation/world.py` | 实例化共享 arbiter/LowState-LowCmd bridge，并把状态注入 manager；尚未注入 production Impact-aware 控制链 |
| Go2/Pixhawk bridge | `go2_sdk_bridge.py`、`pixhawk_mavlink_bridge.py`、`fake_pixhawk.py` | 共享 Sport/LowCmd 仲裁，读取 LowState 足力，记录独立源时间戳并强化取消/超时语义 |
| FSM/安全 | `system_manager.py`、`transition_guards.py`、`state_machine.py`、`safety_monitor.py`、`interlocks.py`、`landing/safety_filter.py` | 获取/激活/撤权/交还、触地后退出屏障，以及“高层锁定”和“LowCmd owner 健康”分支判据 |
| CLI/进程生命周期 | command service、注册/dispatcher/shell/renderer 等 | 只暴露受守卫的 ownership 操作；owner 未交还时普通退出不得遗弃 writer |
| 部署/依赖 | `deploy/install_aarch64.sh`、`pyproject.toml`、requirements、MANIFEST | 固定 SDK/Cyclone 版本，加入 NumPy/SciPy 离线依赖与安装检查 |
| 非论文附带强化 | `x8_bench.py`、README、硬件文档 | X8 诊断白名单、脚本 hash、单轴拆桨测试和部署审计；应与论文算法修改分开记账 |

合并时同伴已调通的形态/F446 主体没有被算法目录重写，但状态机和安全接缝确实已经扩展。哪些文件仍逐字节一致，必须以准备提交时对固定 SHA 的实际比较结果为准，不再沿用早期审计数字或静态名单。

版本号仍是上游 `0.3.14`，发布或提交时建议标记为“Upstream 0.3.14 + Impact-Aware downstream integration”，同时保留上述 base commit，避免把本地扩展误认为同伴原版 0.3.14。

## 15. 当前程序与论文不一致的地方

先区分三个容易混淆的“实现层”：

| 实现层 | 当前真实含义 |
|---|---|
| 顶层 runtime | 自动着陆仍是 legacy `SafeDescentController` 的 DRY-RUN 速度 setpoint；硬件模式直接拒绝 |
| 论文数学 reference | 三维 6-DoF、3D GRF/冲量/摩擦锥、导纳和 SLSQP 均有独立离线实现 |
| 近期可实施实验模型 | 法向一维动力学 + 标量接触事件；仍是 offline/shadow，尚未接入执行器 |

因此“某个公式已有 Python 实现”不等于“顶层程序正在运行该公式”，更不等于“已由真机验证”。逐项差异如下。

| 论文位置/声称 | 当前程序实际实现 | 必须如何改论文 |
|---|---|---|
| 摘要和贡献称完成原型着陆实验，并给出峰值力、姿态误差改善 | 实验章节为空，摘要仍有 `XX.X%/YY.Y%` 占位；当前只有离线、synthetic 和 fake 回归 | 删除结果百分比和“prototype validated”；改成“提出并完成离线软件验证”，等真实试验后再补 |
| 式 (1)–(7) 在线计算折展机构、部署角和旋翼位置 | 着陆假设四臂完全展开且机械锁定；代码直接读取四个固定力臂，不控制折展电机 | 机构运动学可保留为设计章节；控制章节令 `θ=θ_land`、`θdot=0`，把 `r_i^C` 写成常量 |
| 式 (8)–(17) 假设已知 `k_f/k_m`、转速、推力一阶动态和偏航反扭矩 | 完整 reference 有方程，AeroGo2 配置仍借 synthetic 时间常数/变化率/旋向；真实 Pixhawk 没有 N residual 端 | 标明这些只属于离线参考；实机前必须辨识 `T=g(command,V,RPM,...)` 或由飞控提供校准 N 接口，不可把油门直接当 N |
| 式 (18)–(29) 包含 12 腿关节和折展自由度的全模型 | 当前控制预测只使用总刚体 reduced model；腿用独立 URDF FK/IK，折展自由度不进着陆环 | 全模型改为建模背景/离线推导；实现模型明确是固定构型的分层 reduced model |
| 式 (20) 将 `v_B` 写为 B 系，式 (21) 用 `pdot=Rv_B`；式 (32) 又直接用 `pdot=v_B` | 程序统一用总质心 C 的世界系位置/线速度，角速度用 B 系 | 全文改为 `p_C^G,v_C^G,R_GB,ω^B`，并统一写 `pdot_C^G=v_C^G`；不要再让 `v_B` 同时表示两种坐标 |
| Reduced state 文中又假设 body 原点与 CoM 重合 | 用户已定义 B、O、C 不重合；代码显式换算，当前暂定 `p_C^B=[0,0,0.05] m` | 删除重合假设，加入 B→C 位置/速度刚体变换；所有力矩臂从 C 起算 |
| 旋翼面相对 CoM 的高度在早期描述中暂定 150 mm | 最新 B/O/C 数据给出 `O-B=97.2 mm`、旋翼面比 O 高 20 mm、`C-B=50 mm`，故程序为 `z_rotor^C=67.2 mm` | 论文改用 67.2 mm 暂定值并注明待 CAD/BOM 更新，不能同时保留 150 mm |
| 式 (26)、(38)–(44) 使用三维足力、切向冲量、摩擦锥、三维 sticking reset | 完整 reference 实现；当前 Go2 只有每脚一个未标定整数通道，可实施主线只保留法向接触事件和一维理想冲量 | 正文实验模型改成一维法向；三维模型可放“扩展 reference/未来工作”，不得写成已由 Go2 数据验证 |
| 一维方程只约束总旋翼力、总足力和总冲量，却同时优化四个分量 | 程序要求三组逐时刻固定 allocation；非活动项为零，活动项非负且和为 1，当前算例各用 `1/4` 对称先验 | 把 allocation 明确写成外部给定参数/假设，不得把 `1/4` 描述成 MPC 自主求得或实测最优载荷分配；实体阶段须辨识或改用可观测分配模型 |
| Go2 counts 被直接代入法向导纳 | `normal_admittance.py` 将 counts、已标定标量 N、独立世界系 3D N 冻结为互斥模式；counts 接触为真时不能推进导纳，输出只沿会话绑定法向 | 论文必须区分接触检测与 N 单位力控制；未标定时删除力跟踪声称，只保留接触事件逻辑，并写明法向在 reset 前不可切换 |
| 式 (44) 的 `Λ_z/Δt_imp` | 这是“假设冲击时长内的等效平均法向力”，代码也按 average 检查 | 全文把 peak/峰值改成 equivalent average；没有力板或高带宽标定传感器时不能报告真实冲击峰值 |
| 接触表由外部直接指定即可触发冲量，未约束足端是否到地 | 程序要求地面平面、signed distance、非穿透、触地位置/下降速度守卫和单调接触表；完整模型还加全域倾角硬锥 | 把这些约束正式写入 NLP，而不是只把接触表当已知离散参数；说明地面/状态估计误差如何进入容差 |
| 姿态代价使用 `vee(skew(R_d^TR))` | 程序已改用 SO(3) 主值 Log/geodesic；原式在 180° 会给出零误差，不能作为全局着陆姿态代价 | 修改代价函数和误差定义，并加入远小于 90°、由实验批准的 roll/pitch/地面法向夹角硬约束 |
| 式 (45)–(50) 被写成在线实时 MPC | 完整 direct multiple-shooting SLSQP 已使用可终止独立进程和求解后全审计，但尚无 aarch64 WCET；一维求解器仍是进程内离线 SLSQP并永久禁止硬件输出 | 改称 offline/reference 或 shadow MPC；完成 production solver、进程/资源隔离和目标机 WCET 后才可称 real-time |
| MPC 的首个 GRF 可直接送腿 | Go2 SDK 实际接收关节 `q/dq/Kp/Kd/tau`；程序须经接触检测→导纳→workspace→IK→关节限幅→唯一 LowCmd owner | 论文补全该执行链，并说明 MPC 足力是期望量，不是 SDK 可直接发送的命令 |
| 式 (51)–(56) 的三维导纳被描述为实机控制器 | 数学实现与式 (55) 一致：`δ_m(f_est-ηf_des)`；但 3D GRF、真机 IK/零位/方向均未验证 | 只可称离线 reference；近期实验若无 N 标定，应改成基于接触事件的 z 向/姿态恢复，不声称三维力跟踪 |
| 式 (55) 在 `η=1` 时恢复刚度变为零 | 三维 reference 已改为 `K_eff=(1-η)K+ηK_stance`；独立一维法向实现直接使用正 `stance_stiffness_n_per_m`，并冻结三种互斥观测模式、法向会话身份、死区、有界修正和 reset/freeze | 论文应分开给出两套控制律，删除旧零刚度公式；`writer_enqueued_q_rad` 只支持主机侧 anti-windup，不能写成电机实际应用反馈，参数和实体有界性仍须实机辨识 |
| 旋翼命令在多处交替称“总推力”和“修正量” | NLP 变量是经 κ 后实际作用的一组总推力；传输 payload 只有 `Δu_applied=u_applied-u_fc`，raw 仅作审计代数重构 | 固定三种变量名：`u_fc`、`Δu_raw`、`Δu_applied=κΔu_raw`；飞控只加一次，不再乘 κ |
| 普通 onboard FC 被视为能直接执行上述量 | 当前只有 host sink 和 fake；普通 PX4/MAVLink 不具备同 tick baseline、每桨 N、TTL、执行回读及原子 CLEAR | 论文把它列为待实现的专用飞控接口，不得写成 Pixhawk 6X 板卡天然提供 |
| 图 2 表现为一条单速率闭环 | 程序设计是异步 MPC、高频 Go2 腿环、飞控原生高速环和独立安全环四个时间域 | 用本节节点图重画论文控制框图，增加 snapshot/policy/epoch/TTL/ACK/watchdog |
| 腿与旋翼被视为同时执行同一 MPC 控制量 | 当前代码先完成 FC residual 激活再发布腿策略；LowCmd mailbox、DDS writer 与电机应用也不是同一 ACK，跨设备仍非原子 | 论文必须把共同未来执行时刻/事务栅栏列为实现条件；在完成前不得声称严格同步协同控制，只能称带已知限制的分层参考架构 |
| 检测到触地即可结束 AUTO_LANDING | LowCmd 路径必须完成 `POST_TOUCHDOWN_RECOVERY`、residual 零值 ACK+执行+持续状态、setpoint 停止、Go2 safe-hold 和稳定驻留 | 把恢复阶段和退出屏障写入算法/伪代码；`LANDING_COMPLIANT` 是另一条上游高层流程，不能混用 |
| 质量、CoM、惯量和限值作为确定参数 | 26.087 kg、C 高 50 mm、离线惯量区间和 X8 曲线均是先验；正式惯量仍为 null | 参数表增加“来源、revision、标定状态、不确定度、允许用途”；估计值只用于仿真和敏感性分析 |

导纳的力误差项仍按 `f_est-ηf_des` 定义；这与非零 `K_stance`、deadband 和 anti-windup 是不同层次的修改，论文应分别给出符号、单位和状态更新，避免把安全限幅误写成原始线性模型本身。

## 16. 以当前程序为准的论文改写建议

### 16.1 先收窄论文的完成度

当前可以成立的表述是“完成算法公式、严格软件边界、离线算例和故障注入验证”；不能成立的是“完成旋翼—四足协同真机着陆”“实时 MPC 已部署”或“降低实测冲击峰值”。标题、摘要、贡献和结论都应使用同一完成度，不要只在实验章节加一句限制。

建议把贡献改成：

1. 固定展开构型下、显式区分 B/O/C 的着陆 reduced model；
2. 一套含三维冲击约束的离线 reference NLP，以及面向现有传感器的法向简化模型；
3. 面向 Go2 LowCmd 和飞控 residual 的多速率、唯一所有权、TTL/ACK/safe-hold 安全架构；
4. 当前仅软件/离线验证，实体闭环验证作为下一阶段。

### 16.2 用下面的实现模型替换含糊公式

固定几何写成：

\[
\theta=\theta_{\mathrm{land}},\qquad \dot\theta=0,\qquad
{}^B r_i^C=\mathrm{const.}
\]

其中水平半径为 0.665 m，顺序固定为 `[RR,LF,LR,RF]`，当前相对 C 的 z 分量为 0.0672 m。第一根几何臂位于 `x>0,y>0` 象限并对应 LF；几何周向顺序与控制数组顺序必须分开写。

状态统一成：

\[
x_r=\left[p_C^G,\ v_C^G,\ R_{GB},\ \omega^B,\ T\right],
\qquad \dot p_C^G=v_C^G .
\]

若估计器给出 B，则先做：

\[
p_C^G=p_B^G+R_{GB}p_C^B,\qquad
v_C^G=v_B^G+\omega^G\times(R_{GB}p_C^B).
\]

近期实验用的 impact 模型写成：

\[
\dot z_C=v_z,\qquad
m\dot v_z=F_{\mathrm{rotor},z}+\sum_i F_{z,i}-mg,
\]
\[
v_z^+=v_z^-+\frac{\sum_iJ_{z,i}}{m},\qquad
J_{\mathrm{stop}}=m\max(0,-v_z^-).
\]

必须紧接着声明 `J_stop` 是理想预测量，不是 SDK 实测冲量；未标定 counts 只决定接触布尔量。若继续保留三维式 (38)–(50)，应明确标成离线 reference 问题，并与近期实体方法分小节。

着陆几何至少补为：

\[
d_{i,k}=n^T\left(p_{C,k}^G+R_{GB,k}\,{}^B r_{CF_i}\right)-h\ge 0,
\]
\[
0\le d_{i,k_{td}}\le\varepsilon_d,qquad
n^Tv_{F_i,k_{td}}^-\le-v_{\min},qquad
s_{i,k+1}\ge s_{i,k},
\]
\[
e_R=\operatorname{Log}(R_d^TR)^\vee,qquad
(R_{GB}e_z)^Tn\ge\cos\theta_{\max},quad 0<\theta_{\max}<\frac{\pi}{2}.
\]

其中数学上 `<90°` 只排除翻转分支；实验使用的 `θ_max` 必须明显更小且由状态估计噪声和支撑测试确定。论文还必须把 `f_{MPC}` 称为期望足力，并把 `f_{actual}\approx f_{MPC}` 明确列为待辨识/验证假设，而不是由导纳和 IK 的存在直接推出。

接触后导纳改写为：

\[
M(\eta)\ddot x+D(\eta)\dot x+
\left[(1-\eta)K+\eta K_{stance}\right]x
=\operatorname{deadband}\!\left(f_{est}-\eta f_{des}\right),
\quad K_{stance}\succ0.
\]

死区、位置/速度投影、接触释放 reset/freeze 和下游 `writer_enqueued_q_rad` 主机侧 anti-windup 都是实际实现的一部分；它只证明 writer 已接受软件限幅后的 q，不证明电机已应用。论文的稳定性或有界性讨论必须纳入这些非线性和反馈层级限制，而不是继续分析已经删除的零驻留刚度系统。

旋翼接口只保留一套语义：

\[
\Delta u_{\mathrm{applied}}=\kappa\Delta u_{\mathrm{raw}},\qquad
u_{\mathrm{final}}=u_{\mathrm{fc}}+\Delta u_{\mathrm{applied}},
\qquad 0\le\kappa\le1.
\]

当前硬件实验必须取 `κ=0`；今后只有 N 单位、顺序、baseline tick、headroom、TTL、ACK/执行回读和 CLEAR 全部验收后，才可按风险评审逐级提高。不能在论文中先给一个脱离实测链路的“推荐正 κ”。

### 16.3 重写算法流程

论文伪代码建议与程序一致：

1. 低频线程形成带 session/ownership/contact/config/model identity、准确腿序/mapping 和地面几何的相干快照；
2. reference/production solver 生成短寿命策略，独立复核等式、不等式、变量边界、姿态锥和端到端 deadline，不合格则丢弃；
3. 生产实现先准备 FC residual 和腿策略，但两端都不能立即形成不可撤回的新作用；
4. 经共同未来执行时刻、两阶段 commit 或等价事务栅栏同时生效；当前程序缺这一能力，所以本步保持硬件禁用；
5. 高频腿环首先验证首帧关节角与控制器历史在经实机确定的 `initial_joint_alignment_tolerance_rad` 内；随后每个新鲜 LowState 周期做接触迟滞、导纳、IK 和限幅，提交给唯一 owner；只有收到同 sequence 的 writer generation 与 `writer_enqueued_q_rad` 后才提交导纳状态，并明确该证据不是电机 ACK；
6. 飞控在自己的高速环中保持基线，只替换并叠加一个未过期 residual register；ACK 必须区分“接收、排程、实际执行”；
7. 接触 epoch 变化、ACK/age/owner/跟踪异常、其他 DDS publisher 出现或人工接管时，先清 residual，再让同一 LowCmd writer safe-hold；
8. 触地后继续处于 `AUTO_LANDING/POST_TOUCHDOWN_RECOVERY`；只有退出屏障全部满足才到 `TOUCHDOWN_VERIFY`，随后在地面事务中交还高层。

同时明确：串行 `ImpactAwareLandingCoordinator` 是单周期离线参考，不是生产调度器；“成对命令对象”也不是已经实现的跨设备原子提交。

### 16.4 重写验证章节

在没有力板、动捕、吊架、推力台和已标定足力的当前条件下，可报告：

- 方程残差、约束满足率、求解成功/失败分布和参数敏感性；
- synthetic/fake 下 TTL、乱序、过期、ACK 丢失、接触 epoch 变化和 owner 故障注入；
- HW-RO 下 LowState/Pixhawk/F446 的 age、jitter、脚序趋势和时间一致性；
- `κ=0` shadow 计算的求解时延与候选 residual，但不得执行；
- 拆桨并可靠固定条件下的映射/遥测审计。

不能报告真实足端 N、切向力、冲量、瞬时峰值、协同着陆改善率或自由落地成功率。取得相应仪器/标定、真实 residual 固件、cross-device commit、有界 solver 和分阶段风险许可后，再新增实体结果；不要用离线图替代实体证据。

### 16.5 推荐论文结构

1. 引言：把贡献限定为固定展开着陆和当前验证层级；
2. 系统设计：保留折展机构，但说明着陆中锁止；
3. 坐标与参数：先定义 G/B/O/C、旋翼顺序和不确定度；
4. 离线完整 reference 模型：6-DoF、3D impact/NLP；
5. 当前可实施模型：法向一维、接触事件和 κ=0 shadow；
6. 多速率实现与安全状态机：LowCmd ownership、FC residual、恢复退出；
7. 离线/故障注入结果；
8. 限制和真机前置条件；
9. 结论：只总结已经验证的内容。

## 17. 参考入口

- [上游工程](https://github.com/Amagerd1113/AeroGo2/tree/main)
- [Unitree Go2 URDF](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/go2_description)
- [Unitree LowState 示例](https://github.com/unitreerobotics/unitree_ros2/blob/master/example/src/src/read_low_state.cpp)
- [Hobbywing X8 G2](https://www.hobbywing.com/products/xrotor-x8-g2)
- [PX4 VehicleThrustSetpoint](https://docs.px4.io/main/en/msg_docs/VehicleThrustSetpoint)
- [状态机](STATE_TRANSITIONS_ZH.md)；[aarch64 部署](HARDWARE_AARCH64_ZH.md)
