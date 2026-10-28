# v2 方法说明与组内核对稿

本文仅整理已经完成的 v2，不增加实验，也不把后续版本的结果算入 v2。依据实际脚本、配置、实验 JSON 与导出清单核对；用于组内理解和讨论，不是报告或 PPT。

共享快照内可直接查看[本地融合指标与 bootstrap 摘要](../evidence/v2_local_summary.json)、[成员验证指标](../evidence/validation_metrics.json)、[历史导出清单](../evidence/v2_export_manifest_historical.json)和[后续用户 Public 确认](../evidence/public_result_user_reported.json)。本文提到的原实验 `outputs/`、`cache/` 和未保留的历史配置路径是审计出处，不表示它们已分发；从零复现请先在 Git 仓库执行 `cd v2`，再参照 [V2_REPRODUCE.md](V2_REPRODUCE.md)。

**v2 已确认结果：**11 月 15 日留出 AUC 为 **0.785733450485114**，16 日为 **0.7955248334672058**；Public AUC **0.79757** 来自用户明确确认；Private AUC **未知**。Public 记录注明未独立核验线上上传字节。导出时的 `manifest.json` 仍保留 `not_uploaded`、分数为空的历史状态，应结合后写入的 `public_result_user_reported.json` 理解，不能据此否定用户确认。

## 1. 任务、数据与身份

目标是对每条广告曝光预测点击概率 `pctr=P(label=1|特征)`，以 ROC AUC 评价排序能力。资讯域提供同一用户的另一类行为信息，v2 将其转为广告模型的输入，并未把资讯行直接混入广告训练标签。

| 数据文件（实际输入目录 `2026ml/`） | 行数 | 用途 |
|---|---:|---|
| `train.csv` | 2,526,320 | 广告特征与广告点击标签；2019-11-10 至 16 日 |
| `final_test.csv` | 404,182 | 17 日广告特征；广告标签隐藏 |
| `train_feeds.csv` | 827,820 | 提供的资讯曝光、点击及用户兴趣等字段 |
| `test_feeds.csv` | 142,962 | 提供的测试期资讯行为；其中资讯标签是可用输入 |

主要广告字段可分为用户画像（`user_id,age,sex,residence,city,city_rank`）、设备网络（`series_dev,series_group,device_dev,device_name,device_size,net_type`）、广告与位置（`task_id,adv_id,creat_type_cd,adv_prim_id,inter_type_cd,slot_id,site_id,spread_app_id,hispace_app_tags,app_second_class,app_score`）、日期与日志（`pt_d,log_id`）、用户刷新/生命周期（`u_refreshTimes,u_feedLifeCycle`）和行为序列。字段名的具体业务含义以数据提供方定义为准，不能从匿名 ID 推导真实业务语义。

`log_id` 不唯一，不能用作行主键，也不能简单去重。审计记录显示训练集 `(pt_d,log_id)` 唯一数为 2,521,466，少于总行数；全字段唯一行数却等于总行数。代码给训练、测试分别按原文件顺序生成从 0 开始的 `row_index`，再以 **`(row_index,is_test)`** 标识缓存行。额外特征用该组合键 `1:1` 左连接，避免把相同 `log_id` 的不同曝光合并。`log_id` 同时保留为模型数值特征和导出核对字段，不能将其存在误解为唯一性保证。

## 2. 预处理与特征

### 2.1 先处理原始 token，再编码类别

六个广告历史序列为 `ad_click_list_v001/002/003` 和 `ad_close_list_v001/002/003`，另有 `u_newsCatInterestsST`；以 `^` 分隔。001、002、003 在代码中分别与当前 `task_id`、`adv_prim_id`、`spread_app_id` 比较。每个序列提取长度、不同 token 数、整串类别 ID、前五个 token；广告序列额外计算“包含当前对象”和“首 token 等于当前对象”。

例如，虚构原始序列 `12^7^12`、当前原始 `task_id=7`，得到长度 3、不同 token 数 2、`match=true`、`first_match=false`，前五位为 `12,7,12,-1,-1`。比较发生在**原始字符串/原始对象 ID**空间。随后类别列统一转为字符串类别的连续整数编码；编码后的 `7` 不保证还表示原始 ID 7，也不保证不同列的相同整数有相同语义。

类别词表使用提供的训练和测试特征值，不使用广告标签。数值列转 `Float32`、缺失填 `-1`；类别缺失也先填 `-1` 再编码。序列 null 先填空串后 split，因此代码的“空串长度”不应直接解读为真实有效历史长度；token 转整数失败或越界填 `-1`。原数据中的哨兵值不会自动都转成缺失。训练阶段删除训练子集中的常量列；`pt_d` 不直接进入模型，加入 `day_number=pt_d-20191110`。

### 2.2 基础、序列与当天曝光统计

所有六个成员均使用 `base,sequence,counts,source` 四组，以及快照和 14 组目标统计。当天曝光统计不读广告标签，包括：

- 单键 `user_id,task_id,adv_id,adv_prim_id,spread_app_id,slot_id` 的当天曝光数。
- 组合键 `(user_id,task_id)`、`(user_id,adv_prim_id)`、`(user_id,spread_app_id)`、`(user_id,slot_id)`、`(task_id,slot_id)` 的当天曝光数。
- 每用户每天不同 `task_id,adv_prim_id,spread_app_id,slot_id,net_type,device_size` 的数量。
- `log_id/max_day(log_id)`，以及 `(log_id-min_user_day)/(max_user_day-min_user_day+1)`；这是日志数值位置特征，不是经验证的真实时间戳。

### 2.3 资讯域同日聚合与兴趣压缩

`train_feeds.csv` 与 `test_feeds.csv` 合并后按 `(u_userId,e_et)` 聚合，再与广告 `(user_id,pt_d)` 对齐。特征包括资讯曝光数、点击数、CTR、`cillabel=1` 数量；`u_phonePrice,u_browserLifeCycle,u_browserMode,u_feedLifeCycle,u_refreshTimes,i_dislikeTimes,i_upTimes,e_po,e_rn,pro` 的均值和最大值；`i_docId,i_s_sourceId,i_cat,e_ch,e_m,e_pl,i_dtype` 的不同值数，以及若干字段的众数。全体提供的资讯中频率最高的 20 个 `i_cat` 还生成主题曝光占比与“主题且点击”占比。后者分母是该用户当日**全部资讯曝光数**，不是该主题曝光数，不能称为该主题 CTR。

仅兴趣增强成员额外使用 25 列：资讯主题曝光前缀 `v`、点击主题前缀 `c`、长/短期兴趣及点击类别序列前缀 `l/s/r` 组成用户日文档，TF-IDF（`min_df=3,sublinear_tf=True`）后做 24 维 TruncatedSVD（`n_iter=7,random_state=2026`），再加非零词数 `interest_distinct_tokens`。词表与 SVD 只在资讯日期 ≤20191114 上拟合，其余日期仅变换；缺少匹配向量填 0。资讯标签来自提供的资讯文件，不是隐藏的广告测试标签。

### 2.4 当天历史快照（30 列）

对每个用户、每个日期、每种广告 click/close 序列，将不同整串历史视为不同快照。每种序列生成快照数、token 并集大小、当前对象是否出现在并集、成员关系是否变化、出现比例，共 6×5=30 列。重复曝光中的相同整串快照先去重，同一快照中的重复 token 也去重，所以出现比例是“包含该对象的不同快照数/不同快照总数”，不是曝光比例。

虚构例：同用户同日有两种 task 点击快照 `12^7`、`12^7^9`，当前 task 为 9，则 `count=2,union_size=3,union_match=1,membership_changed=1,match_fraction=1/2`。若当前 task 为 12，最后两项分别为 0 和 1。

这会使用当前曝光之后、但同一天的已提供特征行，属于**教师允许的穿越/传导式特征设置**。快照不读广告标签，也不跨到后续日期；当天曝光统计和资讯同日聚合也依赖完整当天输入。它们不能直接宣称是严格在线、逐条即时可用特征；课堂授权不是生产可部署性的证据。

## 3. 标签统计：未采样历史上的整日 LODO

实现为 `src/run_experiment.py:add_target_stats`。先从截止日之前所有广告训练行提取 `fit_labels`，**再**抽样训练行；因此 TE 计数与先验基于未采样完整历史。验证日/测试日广告标签不进入 TE。v2 使用基础 `TE_KEYS`，没有启用 `extended_te`、`past` 或 `te_rates_only`。

准确的 14 组键如下，每组产生 `_n`、`_clicks`、`_ctr` 三列，共 42 列：

```text
[user_id]                  [task_id]             [adv_id]
[adv_prim_id]              [spread_app_id]       [slot_id]
[app_second_class]         [user_id,adv_prim_id]
[user_id,spread_app_id]     [user_id,slot_id]     [user_id,task_id]
[task_id,slot_id]           [age,spread_app_id]   [sex,adv_prim_id]
```

设截止日为 D，完整历史 H={广告训练行：日期<D}；某训练行日期为 d、键值为 k。N、S 是 H 总行数和点击总数，N_d、S_d 是其中整日 d 的数量；n_k、s_k 是键 k 的历史数量，n_kd、s_kd 是键 k 在日 d 的数量。

```text
训练行：n = n_k - n_kd；s = s_k - s_kd；prior = (S-S_d)/(N-N_d)
验证/测试行：n = n_k；s = s_k；prior = S/N
TE_ctr = (s + α × prior)/(n + α)
α = 100（单键），30（双键）
```

缺失键计数置 0，因此无历史时平滑 CTR 回退到对应 prior。特征名例如 `te_user_id_adv_prim_id_n`、`te_user_id_adv_prim_id_clicks`、`te_user_id_adv_prim_id_ctr`。若双键去除该训练日后有 20 次曝光、2 次点击、其他日期先验为 0.01，则 CTR 为 `(2+30×0.01)/(20+30)=0.046`，而不是未平滑的 0.1。

LODO 排除的是**整天**，不只是当前一行，减少同日重复用户/对象造成的标签自泄漏。训练行可使用截止日前、其自身日期之后的其他训练日标签；这符合课堂允许穿越的设定，但不是严格过去式 TE。代码断言历史最大日期小于截止日。CatBoost 自身类别统计与这里显式构造的 TE 是两个层次，不能混称为同一过程。

## 4. 负样本抽样与概率修正

所有正样本保留，负样本以固定种子 2026 做独立均匀随机保留。五个成员保留负样本比例 q=0.4，另一个 LightGBM 为 q=0.15；验证/测试集完整保留。无时间衰减（`half_life=0`），v2 无额外正类权重。完整重训历史为 2,526,320 行；q=0.4 实际训练 1,022,942 行，q=0.15 为 396,972 行。“全部日期重训”指使用全部日期的训练来源，不能表述为每个模型都拟合全部未采样行。

设模型给出采样分布概率 p_s，采样后的 odds 是总体 odds 的 1/q 倍，代码对单模型验证和测试预测各修正一次：

```text
p = q × p_s / (1-p_s+q × p_s)
```

例如 q=0.4、p_s=0.1，则修正为约 0.042553。单模型这一单调变换不改变 AUC，但影响 Log loss、概率尺度及不同抽样率成员的概率融合，因此先各自修正再加权。公式依赖均匀负采样等假设，不等于已完成独立概率校准。

## 5. 六个最终成员、表示和实际参数

共同输入为前述四组+快照+LODO，训练日期 20191110–20191116，预测 17 日；常量列删除后普通成员 261 列、兴趣成员 286 列。下表类别列数取各个 final `experiment.json`，不是原始数据字段数。

| 最终成员（`outputs/` 下目录名） | 类别处理 / 类别列数 | 负采样 q | 固定树数 | 特有设置 | v2 权重 |
|---|---|---:|---:|---|---:|
| `f01_e16_lgb_snapshots` | LightGBM 类别 / 70 | .4 | 135 | min_child_samples=400 | .09375 |
| `f01_e26_lgb_snapshots_small_sample` | LightGBM 类别 / 70 | .15 | 130 | min_child_samples=150 | .09375 |
| `f01_e17_cat_snapshots` | CatBoost categorical / 70 | .4 | 1594 | 标准快照模型 | .09375 |
| `f01_e22_cat_interest_snapshots` | CatBoost categorical / 70 | .4 | 1828 | 额外 25 列兴趣 | .09375 |
| `f02_cat_lowcard` | CatBoost lowcard / 48 | .4 | 1594 | 仅训练子集中不同值数 ≤512 的已标记类别列声明为类别 | .375 |
| `f02_cat_numeric` | CatBoost numeric / 0 | .4 | 1594 | 所有缓存列按数值输入 | .25 |

两个 LightGBM 共同实际参数：`objective=binary,metric=auc,learning_rate=.035,num_leaves=15,max_depth=-1,reg_lambda=30,reg_alpha=0,colsample_bytree=.85,subsample=.85,subsample_freq=1,max_bin=127,max_cat_to_onehot=8,cat_smooth=100,cat_l2=10,random_state=2026,n_jobs=10,force_col_wise=True,deterministic=True,verbosity=-1`；`n_estimators` 与 `min_child_samples` 见表。类别值还按训练子集 levels 重映射，验证/测试未见类别记为 -1。

四个 CatBoost 共同实际参数：`learning_rate=.05,depth=7,l2_leaf_reg=10,loss_function=Logloss,eval_metric=AUC,random_seed=2026,task_type=GPU,devices=0,gpu_ram_part=.75,border_count=127,thread_count=8,max_ctr_complexity=1,one_hot_max_size=16,random_strength=1,allow_writing_files=False,verbose=100`；`iterations` 见表。未显式设置的库参数沿用库默认值。CLI 中的 `leaves,min_child,colsample,cat_smooth,l1` 不进入 CatBoost 的实际参数字典，不能把它们宣传为 CatBoost 生效的调参。

三种 CatBoost 表示来自**同一编码缓存**：categorical 声明所有保留的已标记类别列；lowcard 只声明其中 ≤512 类的列，其余编码整数作为数值；numeric 不声明类别列。numeric 不是恢复原始数值，编码整数的顺序不具备已证实业务含义，重建缓存可能改变编码。这里的理由是验证上的互补性，而不是声称编码顺序具有因果意义。

最终模型均不对隐藏测试标签早停。旧四成员的树数沿用 16 日选择结果，新增两成员固定为相同基准树数 1594。`configs/final_v2.json` 仅包含新增两模型，旧四个来自 `configs/final_v1.json` 并复用已有产物；复现六成员不能只运行 `final_v2.json`。

## 6. 时间验证、指标与选择证据

15 日留出：训练日期<D=20191115，未采样历史 1,679,557 行，验证完整 424,780 行、3,523 个正样本。16 日留出：训练日期<D=20191116，未采样历史 2,104,337 行，验证完整 421,983 行、3,724 个正样本。不是随机 K 折，也不是每个样本都严格只使用过去信息的在线回测。

主指标 ROC AUC 衡量随机正例预测高于随机负例的概率；同时记录 Average Precision（AP）和 Log loss。样本正例不足 1%，AP 有助于观察稀有点击检索，Log loss 用于观察概率质量。最终 v2 的 AP 为 0.08009397371495207 / 0.08149794758933063，Log loss 为 0.04241679891396335 / 0.04430567768045923（15/16 日）。最终全日期模型没有本地测试指标，实验 JSON 的空 `metrics` 是因为标签隐藏。

| 已记录方案 | 15 日 AUC | 16 日 AUC | 证据与解释 |
|---|---:|---:|---|
| 原 LightGBM + CatBoost 等权 | .7675067172935426 | .7802651668874175 | `b02/b01`；早期基线 |
| 增加当天快照后的两模型等权 | .7800203555393169 | .7903653493405735 | `b04/b03`；不应将整个差值都认定为单一机制的因果效应 |
| 旧四模型等权 | .7816543996986982 | .7915565005668003 | `b15/b14`；采样率与兴趣表示的组合 |
| 50% lowcard + 50% 旧四模型 | .784753129171898 | .7949452376022135 | `b16/b17` |
| v2：25% numeric + 75% 上一行 | **.785733450485114** | **.7955248334672058** | `b18/b19` |

lowcard 单模型 15/16 日 AUC 为 .781896728590518 / .7921197343638641；numeric 为 .7833652570042203 / .7911978367573596。numeric 在 16 日并非最强单模型，但加入已有融合后提升，支持的是预测互补性。对 lowcard 已比较 .25/.5 固定权重；numeric 也比较 .25/.5，16 日相对 lowcard 融合增益分别约 .000580/.000181，属于小幅确认，不能夸大为单独的重大突破。

精确展开式为：

```text
p_old4 = (p_LGB40 + p_LGB15 + p_Cat + p_CatInterest)/4
p_lowblend = .5×p_lowcard + .5×p_old4
p_v2 = .25×p_numeric + .75×p_lowblend
     = .09375×(p_LGB40+p_LGB15+p_Cat+p_CatInterest)
       + .375×p_lowcard + .25×p_numeric
```

这里均是修正后的概率加权平均，没有训练二层 stacking 模型，没有使用 rank 平均。更复杂 lowcard 兴趣/depth8 扩展在 15 日的固定融合均负增益，记录为停止、不继续 16 日（`repr_extension_day15.json`），没有加入 v2。

两日都参与过历史模型选择、部分早停及融合比较，不是独立盲测。表示实验中的固定树数、预设权重控制了部分自由度，却不能消除此前选择影响。`b18/b19` 的 300 次用户簇 Poisson bootstrap 相对旧四模型增益区间分别为 [.0023407544,.0059782080] 与 [.0020550933,.0057027271]；记录明确不校正模型选择偏差，也不证明隐藏集必然提升。不同日标签分布、用户重复及完整当天特征可用性都限制外推。

## 7. 导出与交付身份

`configs/submission_v2.json` 指定六成员和权重，`src/export_submission.py` 按 `row_index` 排序并逐成员核对 `row_index,log_id,pt_d`、隐藏标签占位 -1、概率有限且在 [0,1]、完整历史行数与 final 标记。CSV 列为 **`id,pctr`**：`id` 是原 `final_test.csv` 行位置 0..404181，绝不是 `log_id`。格式依据队长先前已接受的提交包；本文没有重新访问比赛网页核实规则。

正式文件为 `outputs/submission_20261026_v2/submission.csv`，404,182 行；无 BOM，概率写到小数点后 10 位并回读检查。清单 SHA256：`fcd0e300d640565131127ec7e1ae75574d8a2d635174063c3e7bf6b2b08dcea3`。清单记载概率最小 .0002531709627597118、最大 .8355831755313949、均值 .008106176279634931。这里引用已有清单，未为本次文档重新训练或重新导出；Public .79757 的依据是用户确认，Private 仍未知。

## 8. 组内讨论与来源索引

当前选择依据是两日记录上的收益、不同表示与抽样率的互补以及可复用产物。需要讨论的边界是：允许穿越的课堂收益与真实在线收益有多大差别；匿名类别整数作为数值的收益是否依赖当前缓存编码；负采样修正后的概率是否还需要独立校准；频繁选择后哪些提升能在真正独立数据上保持。现有材料不能回答这些问题，也不应补写未经执行的实验证据。

以下为原实验目录审计路径，其中 `outputs/`、`cache/` 和 `configs/repr_blends.json` 未随共享快照分发。表中 `src/`、`final_v1/final_v2/submission_v2` 配置可在本快照中查看；共享结果证据见本文顶部链接。

| 要核对的内容 | 原实验目录审计路径 |
|---|---|
| 行数、字段与身份审计 | `outputs/data_audit.json`、`outputs/identity_audit.json` |
| 基础、序列、计数、资讯聚合及编码 | `src/build_features.py`、`cache/features_v1.json` |
| 快照定义及 30 列 | `src/build_snapshot_features.py`、`cache/snapshot_rows.json` |
| 兴趣文档、拟合时间边界及 25 列 | `src/build_interest.py`、`cache/interest_rows.json` |
| LODO 键、平滑公式、抽样/修正、类别模式与模型实际调用 | `src/run_experiment.py` |
| 旧四成员 / 新两成员重训配置 | `configs/final_v1.json`、`configs/final_v2.json` |
| 六模型实际参数、列名单、采样行数、树数 | `outputs/<上表最终成员>/experiment.json` |
| 表示比较与 numeric 小幅确认 | `outputs/recovery_analysis/cat_representation_day15.json`、`cat_representation_day16.json`、`cat_numeric_day15.json`、`cat_numeric_day16.json` |
| 被拒绝的表示扩展 | `outputs/recovery_analysis/repr_extension_day15.json` |
| 早期方案 progression | `outputs/ensemble_summary.csv`、`outputs/b01_original_day16/ensemble.json` 至 `outputs/b15_sampling_interest_day15/ensemble.json` 的相应目录 |
| 表示融合权重及 v2 验证 | `configs/repr_blends.json`、`outputs/b16_lowcard_day15/ensemble.json`、`outputs/b17_lowcard_day16/ensemble.json`、`outputs/b18_repr_day15/ensemble.json`、`outputs/b19_repr_day16/ensemble.json` |
| 最终权重与导出检查 | `configs/submission_v2.json`、`src/export_submission.py`、`outputs/submission_20261026_v2/manifest.json` |
| Public 用户确认与未知 Private | `outputs/submission_20261026_v2/public_result_user_reported.json` |

本次文档核验只读取源码、配置与已有 JSON/汇总，未重算全部预测、未训练、未提交 Kaggle；不复制原始数据行或任何凭据。
