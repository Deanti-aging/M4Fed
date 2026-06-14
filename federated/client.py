"""联邦客户端参与者。"""
from __future__ import annotations
import copy
from typing import Optional
from torch.utils.data import ConcatDataset

from .base_participant import BaseParticipant
from models.medmd.medmd import MedMD
from models.neurofm.neurofm_multitask import NeuroFMMultiTask
from data.openmind_dataset import OpenMindDataset, make_dataloader


class Client(BaseParticipant):
    """
    M4Fed 联邦学习中的客户端节点。

    特征：
      - 只持有私有数据（不含公开数据）
      - MedMD 和 NeuroFM 初始权重来自服务器分发
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
          device           : 计算设备
        """
        super().__init__(cfg, device)
        self.node_id = f"client_{client_id}"
        self.client_id = client_id
        self.private_datasets = private_datasets

        # 初始化私有数据的 DataLoader（训练集 shuffle，测试集不 shuffle）
        self.train_loader = make_dataloader(
            private_datasets["train"], cfg.training.batch_size,
            shuffle=True, num_workers=cfg.training.num_workers, seed=cfg.training.seed,
        )
        self.test_loader = make_dataloader(
            private_datasets["test"], cfg.training.batch_size,
            shuffle=False, num_workers=cfg.training.num_workers, seed=cfg.training.seed,
        )

    def build_models(
        self,
        task_specs: dict,
        random_init_neurofm: bool = False,
    ) -> None:
        """
        构建客户端的 MedMD 和 NeuroFM 模型实例。

        调用时机：在每次实验开始前（或重置实验时），由实验脚本调用。
        权重初始化：
          - MedMD 的 U-Net 随机初始化（热身阶段后由服务器覆盖）
          - NeuroFM 骨干：正式模式从 HuggingFace 加载预训练权重；
                         random_init_neurofm=True 时随机初始化（消融实验 C）

        参数：
          task_specs           : 预测任务规格字典（见 NeuroFMMultiTask 说明）
          random_init_neurofm  : 是否随机初始化 NeuroFM 骨干（消融实验用）
        """
        self.medmd = MedMD(self.cfg.model, device=self.device).to(self.device)
        if self.cfg.model.calibrate_vae_scale:
            self.medmd.calibrate_scale_factor(
                self.train_loader,
                n_batches=self.cfg.model.vae_calibration_batches,
            )
        self.neurofm = NeuroFMMultiTask(
            repo_id=self.cfg.model.neurofm_repo,
            task_specs=task_specs,
            device=self.device,
            random_init=random_init_neurofm,
            age_loss_weight=self.cfg.training.age_loss_weight,
        ).to(self.device)

    def n_private_train(self) -> int:
        """返回私有训练集的样本数（用于计算合成数据量 = ratio × n_private）。"""
        return len(self.private_datasets["train"])

    def add_synthetic_data_to_loader(self, synthetic_samples: list):
        """
        将合成样本与私有训练数据合并，返回新的 DataLoader。

        本地训练阶段使用：将本节点合成的数据（或收到的服务器合成数据）
        与私有数据合并，扩充训练集以减少 Non-IID 带来的局部过拟合。

        参数：
          synthetic_samples : 合成样本字典列表（与 OpenMindDataset 格式兼容）

        返回：
          包含私有数据 + 合成数据的 DataLoader（shuffle=True）
        """
        # 将合成样本包装为 Dataset 对象
        syn_ds = OpenMindDataset(
            synthetic_samples,
            mri_shape=tuple(self.cfg.data.mri_shape),
        )
        # 使用 ConcatDataset 合并两个数据集（不复制数据，节省内存）
        merged = ConcatDataset([self.private_datasets["train"], syn_ds])
        return make_dataloader(merged, self.cfg.training.batch_size,
                               shuffle=True, seed=self.cfg.training.seed)
