# 腋窝淋巴结 CT 智能分析工作台

将医学影像课题中的预处理、腋窝淋巴结分割、骨骼定位、单体拆分、影像组学分类和解剖分级串为一个可追踪的 Web 工作流。输入是静脉期胸部 CT 的三维 NIfTI、脱敏病例编号和侧别；输出逐颗淋巴结的 JSON、Markdown、CSV 报告、原始 CT 网格上的 NIfTI 标签图及预览图。

## 演示视频

[观看产品演示（飞书妙记）](https://my.feishu.cn/minutes/obcn43hg83kpz5kojbz21dj3?from=from_copylink)

## 进度与范围

- Streamlit 网页支持上传病例、后台运行、查看进度和历史结果。
- 已在 Linux + RTX 4090 环境完成单病例网页入口端到端验收：15 颗预测淋巴结，生成 8 个最终文件，运行用时 515.4 秒。该数字是这一次运行记录，不是平均耗时或临床性能结论。
- 论文阶段的单模块研究结果：UNETR++ 分割在 380 例独立验证病例上平均 Dice 0.8364、敏感度 0.8476；LightGBM 在 85 颗淋巴结测试样本上 AUC 0.9683；骨骼分级模型在 679 颗研究样本上与专家标注一致率 90.28%。其中 679 颗样本也用于拟合分级边界。完整 Web 工作流尚未做多病例独立评测。

## 代码结构

- `app.py`：网页入口、后台任务和历史结果。
- `main.py`：命令行入口和流水线编排。
- `modules/`：输入校验、空间变换、预处理、TotalSegmentator 调用、UNETR++ 推理适配、后处理、影像组学、LightGBM 分类、几何分级、报告和可视化。
- `configs/`：流水线参数、影像组学参数、特征归一化参数和模型文件指纹。
- `envs/`：主环境与 UNETR++ 环境的依赖清单。
- `tests/`：特征顺序、空间还原、分级、后处理和报告等测试。
- `third_party/unetr_plus_plus/`：基于公开 UNETR++ 工程的推理依赖及项目适配；保留原项目说明和 Apache 2.0 许可证。

## 运行条件

本仓库仅提供源码和配置。模型权重、患者影像、医生标注、运行结果、云端环境备份均未公开。完整推理需要按 `configs/model_manifest.json` 配置 5 折 UNETR++ 权重、LightGBM 模型和 TotalSegmentator 权重，并在 Linux GPU 环境配置两个 Python 环境。原项目使用的 PyRadiomics 扩展曾针对 NumPy 1.20.1 重新编译；仅执行 `pip install -r` 不保证能直接复现完整运行环境。

命令行示例（模型及环境就绪后）：

```bash
python main.py --input /path/to/venous_ct.nii.gz --patient-id demo_case --side left --run-id run_001
```

网页示例（在主环境中）：

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

## 来源与贡献

研究算法和原始脚本来自课题组既有工作；本项目独立完成系统化梳理、接口适配、流程编排、网页入口、结果留存和工程验收。UNETR++ 相关代码保留在 `third_party/`，来源与许可证见该目录文件。模型输出用于科研分析演示，不代表已完成整套系统的临床验证。
