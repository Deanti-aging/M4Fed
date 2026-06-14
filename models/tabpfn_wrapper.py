"""
基于 TabPFN 的表格缺失值填充器。

在 M4Fed 热身阶段的"表格补全"步骤中，
对每个节点的元数据表格中缺失的字段值进行预测性填充，
以减少因元数据缺失导致的文本条件质量下降。

填充策略：
  - 对每一列有缺失值的字段，以其他字段作为特征，用已有行训练 TabPFN
  - 连续型字段（年龄、BMI 等）：使用 TabPFNRegressor
  - 类别型字段（性别、诊断等）：使用 TabPFNClassifier
  - TabPFN 不可用时：退回中位数（连续型）或众数（类别型）填充
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional


class TabPFNImputer:
    """
    逐列填充 DataFrame 中缺失值的封装器。

    TabPFN（Tabular Prior-data Fitted Networks）是一种用于小数据集的
    in-context 学习模型，在少量有标注行上直接推理，无需 epoch 级别训练，
    特别适合元数据表格（通常样本量不大）的缺失值预测。
    """

    # 需要用回归模式（数值预测）的字段名集合
    CONTINUOUS_FIELDS = {"age", "BMI", "weight", "height"}

    def __init__(self, device: str = "cpu", seed: int = 42):
        """
        参数：
          device : 计算设备（TabPFN 可使用 GPU 加速）
          seed   : 随机种子（保证填充结果可复现）
        """
        self.device = device
        self.seed = seed
        self._tabpfn = None    # 懒加载标志，None 表示尚未尝试加载

    def _load_tabpfn(self) -> None:
        """
        懒加载 TabPFN 分类器和回归器。
        仅在第一次调用 fit_transform 时加载，避免不必要的导入开销。
        加载失败时设置 _tabpfn=False，后续使用中位数/众数填充。
        """
        if self._tabpfn is not None:
            return
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            self._clf = TabPFNClassifier(device=self.device, seed=self.seed)
            self._reg = TabPFNRegressor(device=self.device, seed=self.seed)
            self._tabpfn = True
        except Exception as e:
            print(f"[TabPFN] 加载失败: {e}。将使用中位数/众数填充作为替代。")
            self._tabpfn = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对 DataFrame 中所有含缺失值的列进行预测性填充。

        处理流程（逐列）：
          1. 跳过无缺失值的列
          2. 若有观测值 < 5 行，直接用中位数/众数填充（样本太少，模型无意义）
          3. 将特征矩阵转为数值型（类别字段做标签编码）
          4. 用 TabPFNRegressor/Classifier 拟合有值行，预测缺失行
          5. TabPFN 推理失败时退回中位数/众数填充

        参数：
          df : 含缺失值的元数据 DataFrame

        返回：
          填充后的 DataFrame（深拷贝，不修改原始数据）
        """
        self._load_tabpfn()
        df = df.copy()
        df = df.reset_index(drop=True)

        # 只处理非标识符字段（participant_id 和 filename 不参与填充）
        feature_cols = [c for c in df.columns if c not in ("participant_id", "filename")]

        for target_col in feature_cols:
            missing_mask = df[target_col].isna()
            if missing_mask.sum() == 0:
                continue  # 该列无缺失，跳过

            observed_mask = ~missing_mask

            if observed_mask.sum() < 5:
                # 有效样本太少，TabPFN 泛化能力不可靠，退回简单填充
                df = self._simple_fill(df, target_col)
                continue

            # 以其他所有字段作为特征（排除当前目标列）
            feature_cols_for_target = [c for c in feature_cols if c != target_col]
            # 将特征矩阵转为纯数值型（TabPFN 要求 float32 输入）
            X_all = self._to_numeric_matrix(df[feature_cols_for_target])
            X_train = X_all[observed_mask.values]           # 有值行作为训练集
            y_train = df.loc[observed_mask, target_col].values
            X_pred = X_all[missing_mask.values]             # 缺失行作为预测目标

            if self._tabpfn:
                try:
                    # 根据字段类型选择回归还是分类
                    is_cont = (target_col in self.CONTINUOUS_FIELDS or
                               df[target_col].dtype in [np.float32, np.float64])
                    if is_cont:
                        # 连续字段：使用回归预测
                        y_train_f = pd.to_numeric(y_train, errors="coerce").astype(float)
                        valid = ~np.isnan(y_train_f)
                        if valid.sum() < 3:
                            df = self._simple_fill(df, target_col)
                            continue
                        self._reg.fit(X_train[valid], y_train_f[valid])
                        preds = self._reg.predict(X_pred)
                    else:
                        # 类别字段：使用分类预测
                        self._clf.fit(X_train, y_train.astype(str))
                        preds = self._clf.predict(X_pred)
                    df.loc[missing_mask, target_col] = preds
                    continue
                except Exception as e:
                    print(f"[TabPFN] 列 '{target_col}' 预测失败: {e}。退回简单填充。")
            # TabPFN 不可用或推理失败，使用简单填充
            df = self._simple_fill(df, target_col)

        return df

    @staticmethod
    def _to_numeric_matrix(df_sub: pd.DataFrame) -> np.ndarray:
        """
        将 DataFrame 转换为 float32 数值矩阵，供 TabPFN 使用。

        处理方式：
          - 数值型字段：直接转 float，无法转换的设为 -1
          - 类别型字段（object 类型）：标签编码（category.cat.codes），NaN 设为 -1
        """
        numeric = df_sub.copy()
        for col in numeric.columns:
            if numeric[col].dtype == object:
                # 标签编码：将字符串类别映射为整数（-1 代表 NaN）
                cats = numeric[col].astype("category").cat.codes.astype(float)
                cats[numeric[col].isna()] = -1
                numeric[col] = cats
            # 强制转为数值，无法解析的填 -1（作为缺失值占位符）
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce").fillna(-1)
        return numeric.values.astype(np.float32)

    @staticmethod
    def _simple_fill(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        简单统计填充（TabPFN 的退回方案）：
          - 数值型字段：用中位数填充（对异常值不敏感）
          - 类别型字段：用众数填充（最频繁出现的类别）
          - 众数为空时填充 "unknown"
        """
        if df[col].dtype in [np.float32, np.float64, float]:
            fill_val = df[col].median()
        else:
            mode = df[col].mode()
            fill_val = mode.iloc[0] if len(mode) > 0 else "unknown"
        df[col] = df[col].fillna(fill_val)
        return df
