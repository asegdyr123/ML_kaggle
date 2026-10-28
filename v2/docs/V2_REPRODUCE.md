# v2 从原始数据复现

从 Git 仓库根目录先执行 `cd v2`；以下命令均在 `v2/`（本快照根目录）执行。整理时仅做源码、配置和小型测试验证，未执行这些完整训练命令。原项目产物保留不动；这是独立的共享快照。

## 1. 环境

原工作环境：Windows、Python 3.12.4、NVIDIA GPU（CatBoost 使用 GPU 0）。依赖版本取自原项目虚拟环境的已安装包。`requirements.txt` 固定本流程直接依赖及主要数值计算依赖，不是操作系统/CUDA 或全部间接依赖的完整锁文件。GPU 驱动和硬件差异可能影响训练结果。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

后续统一使用该 Python。Linux 可将 `.\.venv\Scripts\python.exe` 替换成 `.venv/bin/python`，但本快照未在 Linux 验证。CatBoost 训练设备在代码中固定为 GPU；CPU 改法属于另一个实验，不是原配方。

## 2. 数据

获授权后从 Kaggle 比赛 `2026ml` 下载原始 ZIP。将 ZIP 放到仓库外，执行（将示例路径替换成自己的路径）：

```powershell
.\.venv\Scripts\python.exe src/download_data.py --zip ../2026ml.zip --output 2026ml
```

或在自己的终端交互认证下载：

```powershell
.\.venv\Scripts\python.exe src/download_data.py --login --output 2026ml
```

脚本不代为接受比赛规则，也不提交预测。不要向仓库提交凭据。这里必须明确使用 `--output 2026ml`：下载脚本通用默认目录是 `data/`，而 v2 构建和导出脚本使用 `2026ml/`。

最终直接目录结构：

```text
2026ml/train.csv
2026ml/train_feeds.csv
2026ml/final_test.csv
2026ml/test_feeds.csv
```

原运行共有 2,526,320 条广告训练记录和 404,182 条广告测试记录。数据未随代码分发，无法仅从仓库进行真实训练。

## 3. 无训练检查

```powershell
.\.venv\Scripts\python.exe tests/check_submission.py
.\.venv\Scripts\python.exe tests/check_validation.py
.\.venv\Scripts\python.exe tests/check_te_rate_only.py
```

分别检查重复业务 ID 下的行对齐及非法预测拒绝、LODO 目标编码的留出标签隔离及加权 AUC、默认目标统计特征保持完整。这些检查使用合成数据，不证明真实比赛成绩。

## 4. 从零构建所有必要特征

```powershell
.\.venv\Scripts\python.exe -u src/build_features.py --data 2026ml --output cache/features_v1.parquet
.\.venv\Scripts\python.exe -u src/build_snapshot_features.py
.\.venv\Scripts\python.exe -u src/build_interest.py
```

按此顺序执行：第一步建立 `cache/` 并生成 `features_v1.parquet` 与同名 `.json` 元数据；后两步输出 `snapshot_rows.parquet` 和 `interest_rows.parquet`。兴趣构建另存 `interest_transform.joblib`，训练直接读取行特征，不依赖外部预训练模型。

基础构建读取全部四个原始文件；快照读取两份广告文件；兴趣构建读取四份文件。均无需未分发的历史缓存。资讯域的 `train_feeds.csv` 与 `test_feeds.csv` 所提供标签会参与资讯统计/兴趣表示。广告测试标签未知；广告目标编码仅从划分前训练日期计算。

## 5. 可选：重建本地验证结果

以下会正式训练 12 个验证模型，耗时和资源取决于硬件。验证配置从各成员保存的实验 `args` 重建，保持种子、采样、迭代上限和早停参数，路径改为相对仓库路径。融合直接使用六个成员、固定权重，与原嵌套融合数学等价；浮点加法顺序可能产生末位差异。

```powershell
.\.venv\Scripts\python.exe -u src/run_batch.py configs/validation_day15.json
.\.venv\Scripts\python.exe -u src/run_batch.py configs/validation_day16.json
.\.venv\Scripts\python.exe src/run_blends.py configs/blend_day15.json
.\.venv\Scripts\python.exe src/run_blends.py configs/blend_day16.json
```

融合写入 `outputs/v2_review_day15/ensemble.json` 和 `outputs/v2_review_day16/ensemble.json`。对照 AUC 为 0.785733450485114、0.7955248334672058。结果不是保证逐位相同；GPU训练、Polars 类别编码顺序和硬件环境均可能影响结果。两天参与了选择，不可当作独立盲测；原 bootstrap 区间见证据文件，未校正选择偏差。

## 6. 全日期训练并导出 v2

最终训练不依赖第 5 步的验证预测，但使用已冻结的最佳迭代数。必须先训旧四成员，再训新增两成员：

```powershell
.\.venv\Scripts\python.exe -u src/run_batch.py configs/final_v1.json
.\.venv\Scripts\python.exe -u src/run_batch.py configs/final_v2.json
.\.venv\Scripts\python.exe src/export_submission.py configs/submission_v2.json
```

`final_v1.json` 包含四个旧成员（135/130/1594/1828 轮）；`final_v2.json` 只包含 lowcard、numeric（各 1594 轮）。六者都使用 17 日为测试日、全部广告训练日期；先确定完整训练行，再按配置保留全部正样本与 40% 或 15% 负样本。因此“全部日期训练”不等于“保留全部负样本”。

输出为 `outputs/submission_20261026_v2/submission.csv` 及清单。脚本检查成员是最终重训、全训练日期行数一致、测试行身份一致、概率有限且落在 [0,1]；按测试原始行序输出 `id,pctr`，保留重复 `log_id` 对应的独立行。无需先导出 v1 CSV，`submission_v1.json` 只用于理解旧融合。

`run_batch.py` 遇已完成的实验会跳过，`export_submission.py` 拒绝覆盖已有 CSV。干净克隆运行可避免误复用历史输出；不要通过删除原项目产物来追求复现。

原文件哈希记录在 `evidence/v2_export_manifest_historical.json` 和 `evidence/source_snapshot.json`。重新训练不保证重建相同 CSV 字节，原始哈希仅标识原交付文件。本仓库不包含该文件或逐行预测。只有队长决定并执行 Kaggle 提交。
