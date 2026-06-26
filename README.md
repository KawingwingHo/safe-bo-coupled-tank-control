# 耦合双水箱安全贝叶斯优化自动整定

这是一个纯软件在环项目：使用安全贝叶斯优化自动整定两个耦合水箱的四个 PI 参数，同时控制在线试验中的液位越限和泵长期饱和风险。

项目不是对 SafeCtrlBO 仓库的直接复制。当前实现是一个透明的安全 UCB 基线：高斯过程下置信界定义可认证安全集合，安全域以有限步长逐步扩张，最终控制器还需通过独立数字孪生扰动工况资格测试。

当前主实验保留四维 PI 参数空间，并没有“加参数”。四维设置用于验证安全 BO 的在线风险控制逻辑；若要研究加性 GP、多任务 GP 或 SafeCtrlBO 类复杂结构，必须先把控制器扩展为有真实控制意义的 8–12 维空间，并重新完成安全约束、消融和未见工况验证。

## 研究问题

普通贝叶斯优化能够快速找到高性能参数，但可能在调参过程中执行危险试验。本项目检验：

> 在接近普通 BO 性能的前提下，安全 BO 能否显著减少在线整定中的安全违规？

被优化参数为：

```text
[Kp1, Ki1, Kp2, Ki2]
```

代价函数由归一化 IAE、泵能耗和控制量总变差组成：

```text
J = 0.72 * IAE + 0.20 * energy + 0.08 * total_variation
```

安全要求为：

- 液位保持在 0.055 m 至 0.36 m。
- 两个泵指令处于饱和区的总时间比例不超过 50%。

## 最终实验结果

20 个独立整定工况，每种方法 30 次在线试验；每个候选控制器经过 5 个资格工况，之后在 20 个未见工况验证。

| 方法 | 在线违规次数/工况 | 未见工况安全率 | 验证成本 |
|---|---:|---:|---:|
| 人工保守 PI | 0.00 | 100.00% | 0.3527 |
| 随机搜索 | 14.85 | 98.75% | 0.3417 |
| 普通 BO | 19.05 | 99.00% | 0.3380 |
| 安全 BO | 0.90 | 98.75% | 0.3412 |

相对普通 BO，安全 BO 将在线违规减少 95.28%，在线最优安全成本增加 1.39%。配对 Wilcoxon 检验的违规减少单侧 p 值为 `4.18e-05`。结果来源于 `results/final/summary.json`，不是手工填写。

资格门使最终部署安全率很高，但不能抹去在线整定风险。494 个由 GP 下置信界认证的安全试验中，仍有 18 个实际违规。因此本项目不声称绝对安全保证。

## 环境与运行

已验证环境：Apple M1 Pro、macOS 15.7.7、Python 3.12.13。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

运行测试与静态检查：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests
```

GitHub Actions 在全新 macOS + Python 3.12 环境执行同一组检查。

快速实验：

```bash
.venv/bin/python -m safebo_tanks.experiment --quick
```

完整复现实验：

```bash
.venv/bin/python -m safebo_tanks.experiment \
  --output results/final \
  --seeds 20 \
  --budget 30 \
  --candidates 2048 \
  --qualification-scenarios 5 \
  --validation-scenarios 20
```

主要输出：

- `trials.csv`：每次在线试验的参数、性能、安全裕度和预测下置信界。
- `qualified_controllers.csv`：独立资格门选出的部署参数。
- `validation.csv`：未见工况逐次验证结果。
- `summary.json`：配置、汇总指标和统计检验。
- `learning_and_safety.png`：性能收敛与累计违规。
- `safety_calibration.png`：预测安全下界与实际安全裕度。
- `representative_response.png`：双水箱液位和泵指令响应。

信任步长消融：

```bash
.venv/bin/python -m safebo_tanks.ablation \
  --output results/ablation \
  --seeds 20 \
  --budget 30 \
  --candidates 2048
```

消融结果表明，只有 GP 下置信界时平均在线违规为 3.25 次；加入 0.18 信任步长后为 0.90 次，配对单侧检验 `p=5.83e-05`，而成本差异不显著（`p=0.189`）。

部署资格门消融：

```bash
.venv/bin/python -m safebo_tanks.deployment_ablation \
  --output results/ablation \
  --validation-scenarios 20
```

直接部署训练工况中的最优安全点时，未见工况安全率仅 84.50%；加入 5 工况独立资格门后提升至 98.75%（配对单侧 `p=0.000705`），成本差异不显著（`p=0.191`）。因此项目的前沿工程价值由两项独立证据支撑：信任步长约束在线探索风险，资格门约束部署过拟合。

## 演示

- [45 秒数据驱动演示视频](demo/safe_bo_coupled_tank_demo.mp4)：1280×720、20 fps、H.264。
- [系统架构图](assets/system_architecture.png)：在线安全搜索与离线资格门的职责分离。

## 代码结构

```text
src/safebo_tanks/
  plant.py         非线性耦合双水箱、泵滞后、PI 和安全评价
  space.py         四维对数参数空间与保守初始安全集
  optimization.py  人工、随机、普通 BO 和安全 BO
  experiment.py    多种子实验、资格门、统计与绘图
  ablation.py      信任步长消融与统计检验
  deployment_ablation.py  部署资格门消融与统计检验
  demo.py          架构图和 MP4 演示生成
tests/              单元与端到端测试
docs/               工程审计与方法边界
assets/             系统架构图
demo/               演示视频
```

## 方法边界

- 当前方法是透明的安全 BO 工程基线，不是 SafeOpt 或 SafeCtrlBO 的逐行复现。
- 数字孪生被当作昂贵黑箱；若直接利用完整模型预计算所有安全参数，研究问题会被人为简化。
- 在线整定使用固定标准试验工况，未见扰动只用于资格和最终验证，避免把工况变化混入参数效应。
- “高概率安全”依赖 GP 核、置信系数和数据覆盖，不能替代真实设备上的独立保护逻辑。
- 当前只整定 `Kp1, Ki1, Kp2, Ki2` 四个 PI 参数，因此结论限于四维控制器空间，不能用来证明加性 GP、多任务 GP 或完整 SafeCtrlBO 的必要性。

## 下一阶段扩维路线

扩维不能只是把参数名加进搜索向量。合理的 8–12 维版本应来自真实控制自由度，例如：

- 两个回路的前馈修正系数；
- 抗积分饱和回算系数；
- 测量滤波时间常数；
- 设定值权重或参考轨迹平滑参数。

扩维后需要重新定义初始安全集、安全边界和资格门，并在相同在线预算下比较全维 Matérn GP、按回路分组的加性 GP、以及多任务/分组 GP。只有当这些实验重新完成在线违规、性能代价、信任步长消融和未见工况验证后，才可以讨论复杂 GP 结构是否真正必要。

主要参考：[SafeCtrlBO 论文](https://arxiv.org/abs/2408.16307)、[公开代码](https://github.com/hxwangnus/SafeCtrlBO)。
