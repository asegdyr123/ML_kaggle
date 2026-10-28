# 2026ML 广告点击率预测：v2 小组评审代码

本仓库整理用于小组交流、检查和复现的 **v2 六模型融合**。任务是结合广告域与资讯域数据，预测测试广告的点击概率，指标为 ROC AUC。保留实现、配置及小型结果证据；原始比赛数据、模型、预测文件和提交 CSV 均不随仓库分享。

| 方案 | 11 月 15 日本地 AUC | 11 月 16 日本地 AUC | Public AUC |
|---|---:|---:|---:|
| v1 四模型等权 | 0.78165440 | 0.79155650 | — |
| v2 六模型固定权重 | **0.78573345** | **0.79552483** | **0.79757（用户确认已提交）** |

15/16 日均参与模型选择与早停，不是独立盲测。Public 成绩来自用户确认和截图，未通过 API 核验上传文件字节；Private 成绩未知。详细证据见 `evidence/`。历史导出清单中的 `not_uploaded` 和空分数是导出当时状态，后续用户反馈记录在 `public_result_user_reported.json`。

v2 保留两种负采样比例的 LightGBM、普通类别 CatBoost、兴趣增强 CatBoost，并加入 lowcard 与 numeric 两种 CatBoost 表示。旧四模型权重各 0.09375，lowcard 为 0.375，numeric 为 0.25。实现说明见 [V2_METHOD.md](docs/V2_METHOD.md)，从零复现见 [V2_REPRODUCE.md](docs/V2_REPRODUCE.md)。

目录：

```text
src/           特征构建、训练、融合、CSV 导出
configs/       六模型最终训练及 15/16 日验证配方
tests/         小型合成数据检查，不训练模型
evidence/      本地指标、历史导出清单、用户反馈和源码哈希
docs/          方法与复现说明
requirements.txt  原 v2 工作环境实测版本
```

快速阅读顺序：方法说明 → `configs/submission_v2.json` 的成员和权重 → `src/run_experiment.py` 的训练/目标编码 → 复现说明。`final_v2.json` 只训练新增两个模型；从空目录复现必须先运行 `final_v1.json`，生成原四个模型。

数据须由获授权的队员从 Kaggle 比赛 `2026ml` 自行下载，接受规则与访问权限由队员自行处理。四个文件是 `train.csv`、`train_feeds.csv`、`final_test.csv`、`test_feeds.csv`，统一放在 `2026ml/`。不使用外部数据。本方法采用课堂允许的当天行为及测试特征统计，依赖所提供资讯数据中的标签；具体边界见方法说明，不直接套用于严格在线预测情景。

环境基线为 Windows、Python 3.12.4、NVIDIA GPU；CatBoost 训练固定使用 GPU 0。完整命令和限制见复现说明。无需比赛数据的快速检查：

```powershell
python tests/check_submission.py
python tests/check_validation.py
python tests/check_te_rate_only.py
```

仓库排除：比赛 ZIP/原始 CSV、特征缓存、逐行预测、模型权重、提交 CSV、虚拟环境、运行日志、密钥、个人路径、会话记录，以及无关的后续神经网络实验。共享训练模块保留少量显式可选功能，v2 配方未启用它们；源码按当前工作树逐字节保存，不冒充历史 Git 提交。

原 v2 CSV 为 404,182 行，格式 `id,pctr`；`id` 为测试原始行号 0..404181。原文件 SHA256：

```text
fcd0e300d640565131127ec7e1ae75574d8a2d635174063c3e7bf6b2b08dcea3
```

本次整理未重新训练或提交。导出脚本只在本地生成文件，已有 CSV 会拒绝覆盖。**只有队长负责最终 Kaggle 提交**；仓库不包含自动提交入口。
