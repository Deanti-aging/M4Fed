# OpenNeuro 数据集统计（openneuro_metadata.csv）

> **数据来源**：[AnonRes/OpenMind](https://huggingface.co/datasets/AnonRes/OpenMind)
> **统计时间**：待生成（运行 `python scripts/compute_dataset_stats.py` 后自动填充）
> **预处理**：`preprocess_table()` 已将 `n/a`、`NA`、`nan`、空字符串等统一替换为 NaN；
> 连续字段（age、weight、bmi）已转换为 float。

---

## 一、总体概况

| 统计项 | 数量 |
|--------|-----:|
| CSV 总行数（扫描条目） | 待生成 |
| 唯一受试者数（ds_id/sub-id 去重） | 待生成 |

---

## 二、受试者级统计（按 ds_id/sub-id 去重）

| 统计项 | 数量 |
|--------|-----:|
| 总受试者数 | 待生成 |
| **age 字段非空**的受试者数 | 待生成 |
| **age 和 sex 字段均非空**的受试者数 | 待生成 |
| **age 和 health_status 字段均非空**的受试者数 | 待生成 |

---

## 三、扫描条目级统计（每行 = 一个 NIfTI 文件）

| 统计项 | 数量 |
|--------|-----:|
| 总扫描条目数 | 待生成 |
| **age 字段非空**的扫描条目数 | 待生成 |
| **age 和 sex 字段均非空**的扫描条目数 | 待生成 |
| **age 和 health_status 字段均非空**的扫描条目数 | 待生成 |

---

## 四、模态分布（全量）

| 模态（modality） | 扫描条目数 |
|------------------|----------:|
| 待生成 | — |

---

## 五、T1w 结构像专项（M4Fed 实际使用的模态）

| 统计项 | 数量 |
|--------|-----:|
| T1w 扫描条目总数 | 待生成 |
| T1w 中 **age 非空**的扫描条目数（可纳入训练） | 待生成 |
| T1w 中 **age 非空**的唯一受试者数 | 待生成 |

---

## 说明

- **受试者标识**：从 `image_path`（BIDS 格式）提取 `ds_id/sub-id`（如 `ds000001/sub-01`）作为唯一受试者键，跨数据集去重。
- **训练过滤规则**：M4Fed 的 domain task 为脑龄预测，`_sample_loader_fn` 中会跳过 `age` 为空的行，这些受试者的 NIfTI 文件**不会被下载**，从而节省磁盘和网络开销。
- **实际训练样本数**：取决于 `n_cross_ds`、`n_private_ds_per_node`、`n_public_ds_max` 等配置参数对子数据集的进一步划分。

---

## 生成方法

在项目根目录下运行：

```bash
python scripts/compute_dataset_stats.py
# 或指定缓存目录和输出路径：
python scripts/compute_dataset_stats.py --cache_dir /path/to/cache --output openneuro_dataset_statistics.md
```

脚本会自动下载 `openneuro_metadata.csv`（首次运行需联网访问 HuggingFace），
计算统计数据后**覆盖写入**本文件。
