# runs — 训练结果目录

每次训练会自动创建一个时间戳子目录，例如：

```
runs/20261025-213500-阻尼振荡/
├── result.json            # 完整结果（表达式、指标、配置、Pareto、训练历史）
├── expression.txt         # 人类可读的最佳表达式
├── config.json            # 本次训练参数
├── predictions.csv        # 训练集 + 评估集的真实值 / 预测值 / 残差
├── dataset_used.csv       # 实际参与建模的列与 split 标记
├── fit_train.png          # 训练集 预测 vs 真实
├── residuals_train.png    # 训练集残差
├── fit_eval.png           # 评估集（若开启留出集）
├── residuals_eval.png
├── history.png            # 奖励/R² 曲线
├── pareto.png             # 精度–复杂度前沿
├── SR.log                 # PhySO 逐 epoch 候选日志（CSV）
└── train.log              # GUI 运行日志
```

`result.json` 可以通过「结果」或「推理」页的「载入结果…」重新打开；载入后仍可直接输入新样本推理。
此目录中的文件属于用户实验产物，不应手工删除正在使用的模型。
