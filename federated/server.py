"""联邦服务器参与者。"""
from __future__ import annotations
import copy
from typing import Dict, List, Optional
from torch.utils.data import ConcatDataset

from .base_participant import BaseParticipant
from models.medmd.medmd import MedMD
from models.neurofm.neurofm_multitask import NeuroFMMultiTask
from data.openmind_dataset import OpenMindDataset, make_dataloader


class Server(BaseParticipant):
    """
    M4Fed 联邦学习中的服务器节点。

    特征（与客户端的区别）：
      - 同时持有私有数据和公开数据
      - train_loader 合并了公开数据和私有数据（全量训练用）
      - 另有独立的 public_train_loader 和 private_train_loader（分阶段训练用）
      - 热身阶段：服务器在公开数据上预训练 MedMD，然后将 U-Net 分发给所有客户端
      - 联邦循环：服务器接收各客户端的合成数据，用于更新 MedMD
      - 本地训练：服务器在自身私有数据上训练 NeuroFM

    数据访问策略（论文设计）：
      - MedMD 热身：只用公开数据（public_train_loader）
      - NeuroFM 本地训练：私有数据 + 合成数据（add_synthetic_data_to_loader）
      - 评估：公开数据 + 私有数据（test_loader）
    """

    node_id = "server"

    def __init__(self, cfg, private_datasets: dict, public_datasets: dict,
                 device: str = "cpu"):
        """
        参数：
          cfg              : 全局配置对象
          private_datasets : 服务器私有数据集 {'train': Dataset, 'test': Dataset}
          public_datasets  : 服务器公开数据集 {'train': Dataset, 'test': Dataset}
          device           : 计算设备
        """
        super().__init__(cfg, device)
        self.private_datasets = private_datasets
        self.public_datasets = public_datasets

        # ---- 混合 DataLoader（公开 + 私有，用于评估和全量训练） ----
        combined_train = ConcatDataset([
            public_datasets["train"],
            private_datasets["train"],
        ])
        combined_test = ConcatDataset([
            public_datasets["test"],
            private_datasets["test"],
        ])
        self.train_loader = make_dataloader(
            combined_train, cfg.training.batch_size,
            shuffle=True, num_workers=cfg.training.num_workers, seed=cfg.training.seed,
        )
        self.test_loader = make_dataloader(
            combined_test, cfg.training.batch_size,
            shuffle=False, num_workers=cfg.training.num_workers, seed=cfg.training.seed,
        )

        # ---- 独立 DataLoader（分阶段训练用） ----
        # 热身阶段 MedMD 只用公开数据
        self.public_train_loader = make_dataloader(
            public_datasets["train"], cfg.training.batch_size,
            shuffle=True, num_workers=cfg.training.num_workers, seed=cfg.training.seed,
        )
        # 本地训练阶段 NeuroFM 只用私有数据（保证私有性）
        self.private_train_loader = make_dataloader(
            private_datasets["train"], cfg.training.batch_size,
            shuffle=True, num_workers=cfg.training.num_workers, seed=cfg.training.seed,
        )

    def build_models(
        self,
        task_specs: dict,
        random_init_neurofm: bool = False,
    ) -> None:
        """
        构建服务器的 MedMD 和 NeuroFM 模型实例。

        与客户端相同，NeuroFM 可以选择预训练权重或随机初始化（消融实验用）。
        """
        self.medmd = MedMD(self.cfg.model, device=self.device).to(self.device)
        if self.cfg.model.calibrate_vae_scale:
            # 服务器 MedMD 用公开数据进行热身训练，校准也使用公开数据
            self.medmd.calibrate_scale_factor(
                self.public_train_loader,
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
        """返回服务器私有训练集的样本数。"""
        return len(self.private_datasets["train"])

    def n_public_train(self) -> int:
        """返回服务器公开训练集的样本数。"""
        return len(self.public_datasets["train"])

    def add_synthetic_data_to_loader(
        self,
        synthetic_samples: list,
        merge_with: str = "private",
    ):
        """
        将合成样本与服务器现有数据合并，返回新的 DataLoader。

        本地训练阶段：服务器将自身合成的数据（从私有数据生成）
        与现有数据合并，以增加训练多样性。

        参数：
          synthetic_samples : 合成样本字典列表
          merge_with        : 合并策略：
                              "private" - 只与私有数据合并
                              其他值   - 与公开+私有数据合并

        返回：
          合并后的 DataLoader（shuffle=True）
        """
        syn_ds = OpenMindDataset(
            synthetic_samples,
            mri_shape=tuple(self.cfg.data.mri_shape),
        )
        if merge_with == "private":
            base_ds = self.private_datasets["train"]
        else:
            # 合并公开数据和私有数据
            base_ds = ConcatDataset([self.public_datasets["train"],
                                     self.private_datasets["train"]])
        merged = ConcatDataset([base_ds, syn_ds])
        return make_dataloader(merged, self.cfg.training.batch_size,
                               shuffle=True, seed=self.cfg.training.seed)
