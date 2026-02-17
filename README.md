# SOARM-Moce 🤖
> An enhanced robotic arm based on SOARM101: higher payload, larger workspace, and the same control workflow and precision experience  
> **Open-source plan: March 2026** (code and hardware materials will be released on the open-source date)

[English](README.md) | [中文](README_ZH.md)

![SOARM-Moce Model Overview](images/1.JPG)
<!-- TODO: Replace with your "model overview" image path, e.g. docs/media/overview.jpg -->

---

## 1. Project Overview
**SOARM-Moce** is our enhanced version built on top of **SOARM101**. While keeping the same **5-DOF architecture** and **Python + ROS control workflow**, we reinforce key joints with **metal reduction modules** to significantly improve payload capacity and structural stiffness, while also expanding workspace coverage.

This project is designed for:
- Makers and open-source hardware developers (rapid secondary development and feature extensions)
- Education and lab teaching (ROS/kinematics/control/vision course support)
- Lightweight applications and prototyping (pick-and-place, interaction demos, etc.)

---

## 2. Appearance and Structure (Image Slots)
### 2.1 Model Overview
![Model Overview](images/4.JPG)
<!-- TODO: Model overview image -->

### 2.2 SOARM101 vs SOARM-Moce Comparison
![SOARM101 vs SOARM-Moce Comparison](images/3.JPG)
<!-- TODO: Comparison image (recommended: payload/workspace/structural reinforcement points) -->

### 2.3 Core Module Close-up (Metal Reduction Module on Key Joint)
![Key Module Close-up](images/2.JPG)
<!-- TODO: Core close-up (recommended labels: key joint, reduction module, mounting position) -->

---

## 3. Key Upgrades (Compared to SOARM101)
- **Major payload boost**: Reinforced key joints with metal reduction modules, resulting in a significant payload increase (validated by experiments).
- **Larger workspace**: Based on public URDF simulation evaluation, workspace area increases by nearly 30%.
- **Higher stiffness and stability**: Reinforced structure provides stronger torsion and deformation resistance, improving overall system stability.
- **Same precision and control habits**: Repeatability remains 1 mm, and control stays Python + ROS, keeping the learning cost low.
- **More complete ecosystem**: Compatible with the upstream **LeRobot** ecosystem and extended with **Moce-specific ecosystem support**.

---

## 4. Core Metrics Comparison (SOARM101 vs SOARM-Moce)
> The following data is summarized from project comparison materials: payload values come from experiments, workspace-related values come from URDF simulation results.

| Metric | SOARM101 | SOARM-Moce | Change |
|---|---:|---:|---:|
| Rated max payload (kg) | 0.5 | 1.5 | **3x** increase |
| Limit payload (kg) | – | 2.0 | Higher payload headroom |
| Repeatability (mm) | 1.0 | 1.0 | Unchanged |
| Max horizontal reach Rmax (mm) | 380.6 | 433.1 | +13.8% |
| Max 3D reach Dmax (mm) | 447.2 | 516.2 | +15.4% |
| Max Z height (mm) | 428.7 | 502.9 | +17.3% |
| XY workspace area (m²) | 0.3255 | 0.4226 | +29.8% |
| Structural material | Standard 3D-printed structure | Reinforced 3D print + metal reduction modules | Higher stiffness |
| Key joint design | Conventional drive structure | Dual-joint metal reduction reinforced design | Torque amplification |
| Degrees of freedom (DOF) | 5 | 5 | Same architecture |
| End-effector support | Generic end-effector interface | Modular custom end-effector interface | Better extensibility |
| Control method | Python + ROS | Python + ROS | Same |
| Ecosystem support | LeRobot | LeRobot compatible + Moce ecosystem | More complete |
| Modular maintenance | Standard structure maintenance | Upgradable/replaceable key joints | Better maintainability |

---

## 5. Repository Contents (To Be Completed After Open Source)
> **Note: This repository will be completed on the open-source date in March 2026.**

Expected contents:
- `hardware/`: BOM, structural part list, machining/printing recommendations, assembly instructions
- `urdf/`: URDF files, mesh models, inertia/joint parameters
- `ros/`: ROS packages (launch, control, examples)
- `sdk/`: Python control interface, example scripts, API docs
- `docs/`: Calibration workflow, FAQ, development guide
- `examples/`: Trajectory following, teaching record, grasping demo (optional)

---

## 6. Quick Start (Placeholder: To Be Added After Open Source)
### 6.1 Requirements
- Ubuntu 20.04/22.04 (recommended) or macOS/Windows (some features may be limited)
- Python 3.8+
- ROS (Noetic/Humble, depending on release version)

### 6.2 Installation (Placeholder)
```bash
# TODO: pip/rosdep/colcon installation steps will be provided after open source
```
