"""联邦服务器参与者：代表协调方，持有公开数据且负责模型聚合和热身训练。"""
from __future__ import annotations   # 允许注解中使用尚未定义的类名
import copy                           # 标准库：深拷贝（备用，供子类扩展使用）
from typing import Dict, List, Optional   # 类型注解工具
from torch.utils.data import ConcatDataset   # PyTorch 数据集拼接工具

from .base_participant import BaseParticipant                       # 参与者基类（含模型接口）
from models.medmd.medmd import MedMD                                # MedMD 生成模型
from models.swin.swin_multitask import SwinMultiTask                 # SwinUNETR 多任务预测模型
from data.openmind_dataset import OpenMindDataset, make_dataloader  # 数据集和 DataLoader 工厂


class Server(BaseParticipant):
    """
    M4Fed 联邦学习中的服务器节点。

    特征（与客户端的区别）：
      - 同时持有私有数据和公开数据
      - train_loader 合并了公开数据和私有数据（全量训练用）
      - 另有独立的 public_train_loader 和 private_train_loader（分阶段训练用）
      - 热身阶段：服务器在公开数据上预训练 MedMD，然后将 U-Net 分发给所有客户端
      - 联邦循环：服务器接收各客户端的合成数据，用于更新 MedMD
      - 本地训练：服务器在自身私有数据上训练 SwinMultiTask

    数据访问策略（论文设计）：
      - MedMD 热身：只用公开数据（public_train_loader）
      - 领域模型本地训练：私有数据 + 合成数据（add_synthetic_data_to_loader）
      - 评估：只用私有数据（test_loader；公开数据全量用于训练，不保留测试集）
    """

    node_id = "server"   # 服务器的固定节点标识符（客户端为 "client_{id}"）

    def __init__(self, cfg, private_datasets: dict, public_datasets: dict,
                 device: str = "cpu"):
        """
        参数：
          cfg              : 全局配置对象（M4FedConfig）
          private_datasets : 服务器私有数据集 {'train': Dataset, 'test': Dataset}
          public_datasets  : 服务器公开数据集 {'train': Dataset}（全量训练，无测试集）
          device           : 计算设备（"cuda" 或 "cpu"）
        """
        super().__init__(cfg, device)            # 调用基类初始化（设置 cfg、device 等公共属性）
        self.private_datasets = private_datasets  # 保存私有数据集（联邦协议中的私有部分）
        self.public_datasets = public_datasets    # 保存公开数据集（MedMD 热身训练使用）

        # ---- 混合 DataLoader（公开 + 私有训练，用于全量训练） ----
        # 将公开训练集和私有训练集合并（服务器特有权限，可访问两类数据）
        # 公开数据全量作为训练集，不保留测试集（see build_node_datasets）
        combined_train = ConcatDataset([
            public_datasets["train"],   # 公开训练数据集（全量）
            private_datasets["train"],  # 服务器私有训练数据集
        ])
        # 全量混合训练 DataLoader（公开 + 私有训练集，用于全量训练）
        self.train_loader = make_dataloader(
            combined_train,                         # 合并的训练数据集
            cfg.training.batch_size,                # 批大小
            shuffle=True,                           # 训练时打乱顺序
            num_workers=cfg.training.num_workers,   # 并行 worker 数
            seed=cfg.training.seed,                 # 随机种子（保证可复现）
        )
        # 测试 DataLoader：只用私有测试集（公开数据无测试集，评估口径与其他节点一致）
        self.test_loader = make_dataloader(
            private_datasets["test"],               # 只使用私有测试数据集
            cfg.training.batch_size,                # 批大小
            shuffle=False,                          # 测试集不打乱
            num_workers=cfg.training.num_workers,   # 并行 worker 数
            seed=cfg.training.seed,                 # 随机种子
        )

        # ---- 独立 DataLoader（分阶段训练专用，细粒度控制） ----
        # 热身阶段 MedMD 只在公开数据上训练（公开数据质量高，适合预热 U-Net）
        self.public_train_loader = make_dataloader(
            public_datasets["train"],               # 只使用公开训练数据
            cfg.training.batch_size,                # 批大小
            shuffle=True,                           # 热身训练打乱顺序
            num_workers=cfg.training.num_workers,   # 并行 worker 数
            seed=cfg.training.seed,                 # 随机种子
        )
        # 本地训练阶段领域模型只在私有数据上训练（严格保护私有数据不外泄）
        self.private_train_loader = make_dataloader(
            private_datasets["train"],              # 只使用私有训练数据
            cfg.training.batch_size,                # 批大小
            shuffle=True,                           # 本地训练打乱顺序
            num_workers=cfg.training.num_workers,   # 并行 worker 数
            seed=cfg.training.seed,                 # 随机种子
        )

    def build_models(
        self,
        task_specs: dict,                 # 预测任务规格字典（任务名 → 类别数/连续）
        random_init_swin: bool = False,   # 是否随机初始化 SwinUNETR 骨干（消融实验用）
    ) -> None:
        """
        构建服务器的 MedMD 和 SwinMultiTask 模型实例。

        与客户端构建逻辑基本一致，但 MedMD 的 VAE 校准使用公开数据。
        """
        self.medmd = MedMD(self.cfg.model, device=self.device).to(self.device)
        if self.cfg.model.calibrate_vae_scale:
            self.medmd.calibrate_scale_factor(
                self.public_train_loader,
                n_batches=self.cfg.model.vae_calibration_batches,
            )
        self.swin = SwinMultiTask(
            pretrain_url=self.cfg.model.swinunetr_pretrain_url,
            img_size=tuple(self.cfg.data.mri_shape),
            feature_size=self.cfg.model.swinunetr_feature_size,
            task_specs=task_specs,
            device=self.device,
            random_init=random_init_swin,
            age_loss_weight=self.cfg.training.age_loss_weight,
        ).to(self.device)

    def n_private_train(self) -> int:
        """返回服务器私有训练集的样本数（用于日志和合成数据量计算）。"""
        return len(self.private_datasets["train"])   # 取 Dataset 长度

    def n_public_train(self) -> int:
        """返回服务器公开训练集的样本数（用于热身阶段数据量统计）。"""
        return len(self.public_datasets["train"])    # 取 Dataset 长度

    def add_synthetic_data_to_loader(
        self,
        synthetic_samples: list,        # 合成样本字典列表（与 OpenMindDataset 格式兼容）
        merge_with: str = "private",    # 合并基础数据集的选择（"private" 或其他）
    ):
        """
        将合成样本与服务器现有数据合并，返回新的 DataLoader。

        本地训练阶段：服务器将自身合成的数据（从私有数据生成）
        与现有数据合并，以增加训练多样性，缓解服务器数据分布偏差。

        参数：
          synthetic_samples : 合成样本字典列表
          merge_with        : 合并策略：
                              "private" - 只与私有数据合并（默认）
                              其他值   - 与公开+私有数据全部合并

        返回：
          合并后的 DataLoader（shuffle=True，用于本地训练）
        """
        # 将合成样本包装为 OpenMindDataset 对象（统一数据接口格式）
        syn_ds = OpenMindDataset(
            synthetic_samples,                         # 合成样本字典列表
            mri_shape=tuple(self.cfg.data.mri_shape),  # MRI 目标分辨率（与真实数据一致）
        )
        if merge_with == "private":
            base_ds = self.private_datasets["train"]   # 只与私有训练数据合并
        else:
            # 非 "private" 策略：合并公开+私有训练数据作为基础数据集
            base_ds = ConcatDataset([self.public_datasets["train"],
                                     self.private_datasets["train"]])
        merged = ConcatDataset([base_ds, syn_ds])   # 合并基础数据集和合成数据集
        # 返回合并后的 DataLoader（训练模式，shuffle=True）
        return make_dataloader(merged, self.cfg.training.batch_size,
                               shuffle=True, seed=self.cfg.training.seed)
