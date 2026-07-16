"""联邦客户端参与者：代表一个持有私有医学数据的医院/机构节点。"""
from __future__ import annotations   # 允许注解中使用尚未定义的类名
import copy                           # 标准库：深拷贝（备用，供子类扩展使用）
from typing import Optional           # 类型注解：可选参数
from torch.utils.data import ConcatDataset   # PyTorch 数据集拼接工具

from .base_participant import BaseParticipant                       # 参与者基类（含模型接口）
from models.medmd.medmd import MedMD                                # MedMD 生成模型
from models.swin.swin_multitask import SwinMultiTask                 # SwinUNETR 多任务预测模型
from data.openmind_dataset import OpenMindDataset, make_dataloader  # 数据集和 DataLoader 工厂


class Client(BaseParticipant):
    """
    M4Fed 联邦学习中的客户端节点。

    特征：
      - 只持有私有数据（不含公开数据）
      - MedMD 和 SwinMultiTask 初始权重来自服务器分发
      - 联邦循环中使用 QLoRA 微调（降低通信和计算开销）
      - 可接收合成数据并与私有数据合并训练
    """

    def __init__(self, client_id: int, cfg, private_datasets: dict,
                 device: str = "cpu"):
        """
        参数：
          client_id        : 客户端编号（0, 1, ...），用于日志和文件命名
          cfg              : 全局配置对象（M4FedConfig）
          private_datasets : 私有数据集字典，格式：{'train': Dataset, 'test': Dataset}
          device           : 计算设备（"cuda" 或 "cpu"）
        """
        super().__init__(cfg, device)           # 调用基类初始化（设置 cfg、device 等公共属性）
        self.node_id = f"client_{client_id}"    # 节点唯一标识符（日志、文件命名用）
        self.client_id = client_id              # 客户端编号（整数，0-indexed）
        self.private_datasets = private_datasets  # 保存私有数据集字典（含 train/test）

        # 初始化私有数据的训练 DataLoader（shuffle=True 保证每轮顺序不同）
        self.train_loader = make_dataloader(
            private_datasets["train"],             # 私有训练集 Dataset 对象
            cfg.training.batch_size,               # 批大小（全局配置）
            shuffle=True,                          # 训练集打乱顺序（防止模型记住顺序）
            num_workers=cfg.training.num_workers,  # 数据加载并行 worker 数
            seed=cfg.training.seed,                # 随机种子（保证 shuffle 可复现）
        )
        # 初始化私有数据的测试 DataLoader（shuffle=False 保证评估结果稳定可比）
        self.test_loader = make_dataloader(
            private_datasets["test"],              # 私有测试集 Dataset 对象
            cfg.training.batch_size,               # 批大小
            shuffle=False,                         # 测试集不打乱（评估结果可复现）
            num_workers=cfg.training.num_workers,  # 并行 worker 数
            seed=cfg.training.seed,                # 随机种子
        )

    def build_models(
        self,
        task_specs: dict,                 # 预测任务规格字典（任务名 → 类别数/连续/离散）
        random_init_swin: bool = False,   # 是否随机初始化 SwinUNETR 骨干（消融实验 C 使用）
    ) -> None:
        """
        构建客户端的 MedMD 和 SwinMultiTask 模型实例。

        权重初始化：
          - MedMD 的 U-Net 随机初始化（热身阶段后由服务器覆盖为预训练权重）
          - SwinUNETR 骨干：正式模式下载 MONAI 官方 SSL 预训练权重；
                           random_init_swin=True 时随机初始化（消融实验 C 用）
        """
        self.medmd = MedMD(self.cfg.model, device=self.device).to(self.device)
        if self.cfg.model.calibrate_vae_scale:
            self.medmd.calibrate_scale_factor(
                self.train_loader,
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
        """
        返回私有训练集的样本数。
        用于计算合成数据量 = synthesis_ratio × n_private（按比例生成合成数据）。
        """
        return len(self.private_datasets["train"])   # 直接取 Dataset 长度

    def add_synthetic_data_to_loader(self, synthetic_samples: list):
        """
        将合成样本与私有训练数据合并，返回新的 DataLoader。

        本地训练阶段使用：将本节点合成的数据（或收到的服务器合成数据）
        与私有数据合并，扩充训练集以减少 Non-IID 带来的局部过拟合。

        参数：
          synthetic_samples : 合成样本字典列表（每个元素与 OpenMindDataset 格式兼容）

        返回：
          包含私有数据 + 合成数据的 DataLoader（shuffle=True，用于本地训练）
        """
        # 将合成样本列表包装为 OpenMindDataset 对象（统一数据接口）
        syn_ds = OpenMindDataset(
            synthetic_samples,                       # 合成样本字典列表
            mri_shape=tuple(self.cfg.data.mri_shape),  # MRI 目标分辨率（与私有数据一致）
        )
        # 使用 ConcatDataset 合并私有数据集和合成数据集（不复制数据，节省内存）
        merged = ConcatDataset([self.private_datasets["train"], syn_ds])
        # 返回合并后的 DataLoader（训练模式：shuffle=True）
        return make_dataloader(merged, self.cfg.training.batch_size,
                               shuffle=True, seed=self.cfg.training.seed)
