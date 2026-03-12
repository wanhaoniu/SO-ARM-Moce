---
name: soarmmoce-real-con
description: 使用本技能通过本地 Python 脚本或本地 SDK 控制真实 soarmMoce 机械臂；适用于自然语言动作控制、连续小步笛卡尔移动、回零与状态查询。2/3 号关节按减速比 5.3/5.6 适配并运行在多圈模式，无夹爪舵机。
metadata:
  openclaw:
    emoji: "🤖"
    requires:
      bins: ["python3"]
---

# soarmmoce-real-con

## 功能概览

- 本技能用于直接控制本机串口连接的真实 `soarmMoce` follower arm。
- 代码结构分成两层：
  - `scripts/soarmmoce_sdk.py`：SDK 风格控制逻辑
  - `scripts/soarmmoce_state.py` / `scripts/soarmmoce_move.py`：命令行入口
- 当前 TCP 笛卡尔控制按 `5DOF position-only IK` 工作，只解末端位置 `x/y/z`，不再强行约束完整 6D 姿态。
- 对 `delta/xyz` 这类笛卡尔移动，默认锁住 `wrist_roll`，避免横移时为了凑位置把末端滚转甩开。
- 轨迹下发默认按时间频率连续插值，并使用平滑缓入缓出，避免单步跳变太大导致动作发卡。
- 2 号关节 `shoulder_lift` 与 3 号关节 `elbow_flex` 已分别按减速比 `5.3 / 5.6` 做换算，底层运行在多圈模式。
- 当前对 2/3 号多圈关节采用 `方案 A`：读状态时临时使用 `position mode`，发多圈步进命令时切回 `step mode`，这样才能既读到位置又继续用步进控制。
- 当前机械臂没有安装 6 号夹爪舵机，不要调用夹爪脚本或夹爪 API。

## 何时使用

- 用户说：`把机械臂抬高一点`、`往前一点`、`再来一点`、`回零`
- 用户要求状态查询：`当前机械臂在哪`
- 用户要执行多步动作，但不要求你生成正式项目代码

## 核心规则

1. 不要用 node 工具。
2. 脚本返回 JSON 仅供内部判断。
3. 默认最终回复只给用户自然语言。
4. 对普通空间动作，不要反问哪个关节、多少度。
5. 不要调用任何 gripper/open/close/set_gripper 相关命令。

## 推荐调用方式

### 1) 读取状态

```bash
python3 ~/.openclaw/skills/soarmmoce-real-con/scripts/soarmmoce_state.py
```

标定完多圈关节后，`state.joint_state.shoulder_lift / elbow_flex` 会返回用于运动和 IK 的关节角；如果你还想看多圈软件零点下的连续量，读 `state.multi_turn_state.relative_raw / joint_deg_from_home`。

### 1.1) 只读 IK 诊断

```bash
python3 ~/.openclaw/skills/soarmmoce-real-con/scripts/soarmmoce_diag_ik.py --dx 0.02 --frame base
```

### 1.2) 自动标定

运行前先把机械臂摆到你认定的 `home` 姿态。脚本会以当前姿态作为 `home` 参考位；单圈关节自动向正负两个方向寻限位，多圈关节也会自动向正负两个方向寻限位，但实际位置读取统一走 `position mode + 软件展开`。多圈结果不再依赖 `homing_offset`，而是写入 `home_wrapped_raw + min_relative_raw/max_relative_raw`。

```bash
python3 ~/.openclaw/skills/soarmmoce-real-con/scripts/soarmmoce_auto_calibrate.py
```

如果只想先生成 JSON，不回写寄存器：

```bash
python3 ~/.openclaw/skills/soarmmoce-real-con/scripts/soarmmoce_auto_calibrate.py --apply-registers false
```

### 1.3) 手动标定

运行方式对齐 `lerobot-calibrate` 的参数风格。脚本会在交互过程中让你手动把机械臂摆到目标 `home` 姿态；单圈和多圈关节都要再手动掰到整段活动范围。多圈关节在这里会把当前 `home` 姿态记成软件零点，再把整段范围记录成相对 `home` 的连续 raw。捕获 `home` 时脚本会实时打印 `HOME_RAW / WRAPPED / REF`，其中多圈关节的 `REF` 固定为 `0`；随后会再打印一次 `CAPTURED_RAW / HOME_ZERO`，需要你输入 `ok` 才继续。

```bash
python3 ~/.openclaw/skills/soarmmoce-real-con/scripts/soarmmoce_manual_calibrate.py \
  --robot.type=soarmmoce \
  --robot.port=/dev/ttyACM0 \
  --robot.id=soarmmoce
```

如果只想预览结果，不回写寄存器也不保存 JSON：

```bash
python3 ~/.openclaw/skills/soarmmoce-real-con/scripts/soarmmoce_manual_calibrate.py \
  --robot.type=soarmmoce \
  --robot.port=/dev/ttyACM0 \
  --robot.id=soarmmoce \
  --robot.apply_registers=false \
  --robot.save_json=false
```

### 1.4) 多圈极限位测试

这个脚本会读取当前多圈状态和标定 JSON，基于 `min_relative_raw/max_relative_raw` 规划相对 `home` 的目标位，并在范围看起来不可信时直接拒绝执行。

```bash
python3 ~/.openclaw/skills/soarmmoce-real-con/scripts/soarmmoce_test_calibrated_limits.py
```

确认计划没问题后，再执行真实运动：

```bash
python3 ~/.openclaw/skills/soarmmoce-real-con/scripts/soarmmoce_test_calibrated_limits.py --execute=true
```

### 2) 小步笛卡尔移动

```bash
python3 ~/.openclaw/skills/soarmmoce-real-con/scripts/soarmmoce_move.py delta --dz 0.01 --frame base
```

需要排查多圈关节执行误差时，给运动命令加 `--trace`：

```bash
python3 ~/.openclaw/skills/soarmmoce-real-con/scripts/soarmmoce_move.py delta --dy 0.05 --frame base --duration 2.0 --trace
```

### 3) 绝对 XYZ 移动

```bash
python3 ~/.openclaw/skills/soarmmoce-real-con/scripts/soarmmoce_move.py xyz --x 0.22 --y 0.00 --z 0.18
```

### 4) 关节级低层修正

```bash
python3 ~/.openclaw/skills/soarmmoce-real-con/scripts/soarmmoce_move.py joint --joint wrist_roll --delta-deg 5
```

### 5) 回零

```bash
python3 ~/.openclaw/skills/soarmmoce-real-con/scripts/soarmmoce_move.py home
```

## SDK 直接调用

```python
from soarmmoce_sdk import SoArmMoceController

arm = SoArmMoceController()
arm.move_delta(dz=0.01, frame="base")
print(arm.get_state())
```

运行临时脚本时建议显式带上：

```bash
PYTHONPATH=~/.openclaw/skills/soarmmoce-real-con/scripts python3 /tmp/soarmmoce_sequence.py
```

## 可用 API

- `SoArmMoceController().read()`
- `SoArmMoceController().get_state()`
- `SoArmMoceController().move_delta(...)`
- `SoArmMoceController().move_to(...)`
- `SoArmMoceController().move_joint(...)`
- `SoArmMoceController().move_joints(...)`
- `SoArmMoceController().home()`
- `SoArmMoceController().stop()`

以下接口会直接报错，因为当前硬件没有夹爪舵机：

- `SoArmMoceController().set_gripper(...)`
- `SoArmMoceController().open_gripper()`
- `SoArmMoceController().close_gripper()`

## 环境变量

- `SOARMMOCE_PORT`：串口，默认 `/dev/ttyACM0`
- `SOARMMOCE_ROBOT_ID`：标定 ID，默认优先 `soarmmoce`，找不到再回退 `follower_moce`
- `SOARMMOCE_CALIB_DIR`：标定目录
- `SOARMMOCE_URDF_PATH`：URDF 路径，默认优先使用 `sdk/src/soarmmoce_sdk/resources/urdf/soarmoce_urdf.urdf`
- `SOARMMOCE_TARGET_FRAME`：末端 frame，默认 `wrist_roll`（按当前 5DOF 链截断）
- `SOARMMOCE_HOME_JOINTS_JSON`：覆盖 home 目标关节
- `SOARMMOCE_JOINT_SCALE_JSON`：覆盖关节减速比/方向，默认 `{"shoulder_pan":1.0,"shoulder_lift":5.3,"elbow_flex":5.6,"wrist_flex":1.0,"wrist_roll":1.0}`
- `SOARMMOCE_LINEAR_STEP_M`：笛卡尔插值步长，默认 `0.01`
- `SOARMMOCE_JOINT_STEP_DEG`：关节插值步长，默认 `5.0`
- `SOARMMOCE_CARTESIAN_UPDATE_HZ`：笛卡尔轨迹下发频率，默认 `20.0`
- `SOARMMOCE_JOINT_UPDATE_HZ`：关节轨迹下发频率，默认 `25.0`
- `SOARMMOCE_MAX_EE_POS_ERR_M`：笛卡尔动作最终位置误差容忍，默认 `0.01`
- `SOARMMOCE_IK_TARGET_TOL_M`：5DOF IK 收敛阈值，默认 `0.001`
- `SOARMMOCE_IK_MAX_ITERS`：5DOF IK 最大迭代次数，默认 `200`
- `SOARMMOCE_IK_DAMPING`：5DOF IK 阻尼系数，默认 `0.05`
- `SOARMMOCE_IK_STEP_SCALE`：5DOF IK 每轮步进缩放，默认 `0.8`
- `SOARMMOCE_IK_JOINT_STEP_DEG`：5DOF IK 单轮单关节最大步长，默认 `8.0`
- `SOARMMOCE_IK_SEED_BIAS`：5DOF IK 保持当前姿态的偏置强度，默认 `0.02`
- `SOARMMOCE_ARM_P_COEFFICIENT`：单圈关节 P 参数，默认 `16`
- `SOARMMOCE_ARM_D_COEFFICIENT`：单圈关节 D 参数，默认 `8`

## 执行策略

默认优先级：
1. `soarmmoce_state.py`
2. `soarmmoce_move.py delta`
3. `soarmmoce_move.py xyz`
4. `soarmmoce_move.py home`
5. SDK 临时脚本
6. `joint` / `joints` 仅作低层兜底

对 `上/下/左/右` 默认使用 `frame="base"`。
只有用户明确要求沿末端当前方向前进/后退时，才优先使用 `frame="tool"`。

串口相关脚本不要并行运行；同一时刻只保留一个 `state/move/diag` 进程，否则容易出现假性的缺电机 ID 或端口占用。

自动标定脚本同样不要和其它串口脚本并行运行。它的限位判定优先看 `Present_Velocity + Moving`，`Present_Current` 只作为兜底异常保护。

## 参考文件

- `skills/soarmmoce-real-con/scripts/soarmmoce_sdk.py`
- `skills/soarmmoce-real-con/scripts/soarmmoce_auto_calibrate.py`
- `skills/soarmmoce-real-con/scripts/soarmmoce_manual_calibrate.py`
- `skills/soarmmoce-real-con/scripts/soarmmoce_state.py`
- `skills/soarmmoce-real-con/scripts/soarmmoce_diag_ik.py`
- `skills/soarmmoce-real-con/scripts/soarmmoce_move.py`
- `skills/soarmmoce-real-con/agents/openai.yaml`
