"""
将 MRI 图像配准到 MNI152 标准模板空间的预处理工具。

用途：M4Fed 补充实验（experiments/healthy_subset_registered.py）——在
healthy_subset.py 的基础上，验证"配准到标准模板"这一预处理步骤对训练/评估
结果的影响，其余训练流程（数据筛选、划分、轮数、超参）与 healthy_subset.py
完全一致。

实现要点：
  - 标准模板：MNI152 2009a 非线性对称模板（ICBM152，经 nilearn 自动下载并
    本地缓存），使用带颅骨的全头 T1 模板而非脑提取模板——OpenMind 样本是否
    已做脑提取（is_brain_extract 字段）并不统一，全头模板对两种输入都更稳健。
  - 配准方式：仅做刚体+仿射（12 自由度）配准，不做非线性形变（如 ANTs SyN）。
    非线性配准单张需要几分钟到几十分钟，对 OpenMind 健康子集（约 7000+ 张
    T1w 扫描）不可行；仿射配准足以体现"空间归一化"这一变量，且单张约
    10~30 秒，可在合理时间内跑完全量数据。
  - 直接用 SimpleITK 读取 NIfTI 文件（sitk.ReadImage），不经过 nibabel 的
    get_fdata() 数组再手工转换——SimpleITK 的 NIfTI reader 已正确解析
    sform/qform 仿射与物理空间信息，避免手工处理 nibabel(RAS+)/ITK(LPS+)
    坐标系差异引入的错误。
"""
from __future__ import annotations    # 允许注解中使用尚未定义的类名
import os                              # 路径操作
from typing import Optional            # 类型注解工具

import numpy as np                     # 数值数组

# 标准模板本地缓存目录（未指定 cache_dir 时使用）
_DEFAULT_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      "data", "mni_template")
# 配准结果磁盘缓存目录（未指定时使用）：训练脚本反复 __getitem__ 会命中同一批
# 样本，配准是整条流水线里最耗时的一步，必须落盘缓存，否则每个 local_round /
# epoch 都要重新配准一遍
DEFAULT_REGISTERED_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                             "data", "mri_registered_cache")

_TEMPLATE_PATH_CACHE: dict = {}   # {cache_dir: 模板文件本地路径}，进程内只下载/解析一次


def get_template_path(cache_dir: Optional[str] = None) -> str:
    """返回（必要时下载）MNI152 2009a 全头 T1 标准模板的本地文件路径。"""
    key = cache_dir or _DEFAULT_TEMPLATE_DIR
    if key in _TEMPLATE_PATH_CACHE:
        return _TEMPLATE_PATH_CACHE[key]
    from nilearn import datasets   # 局部导入：只有启用配准时才需要
    os.makedirs(key, exist_ok=True)
    bunch = datasets.fetch_icbm152_2009(data_dir=key, verbose=0)
    _TEMPLATE_PATH_CACHE[key] = bunch["t1"]
    return bunch["t1"]


def register_to_template(mri_path: str, template_path: Optional[str] = None) -> np.ndarray:
    """
    将 mri_path 指向的 NIfTI 文件仿射配准到标准模板空间，返回配准后的体积。

    流程：几何中心初始化 → 多分辨率（4x/2x/1x 下采样）仿射配准（Mattes 互信息
    + 正则化梯度下降）→ 用最终变换把原图重采样到模板网格。

    参数：
      mri_path      : 待配准的 NIfTI 文件本地路径
      template_path : 标准模板文件路径（None 时用 get_template_path() 的默认模板）

    返回：
      float32 numpy 数组，形状与模板体素网格一致（配准+重采样后的体素值，
      未做百分位裁剪/归一化——交由 data.preprocessing.preprocess_mri 继续处理）
    """
    import SimpleITK as sitk   # 局部导入：只有启用配准时才需要安装该依赖

    fixed = sitk.ReadImage(template_path or get_template_path(), sitk.sitkFloat32)
    moving = sitk.ReadImage(mri_path, sitk.sitkFloat32)

    initial_tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_tx, inPlace=False)
    # 多分辨率金字塔：先在粗分辨率上快速收敛到大致对齐，再逐级细化，
    # 既加速又降低陷入局部极小值的概率
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_tx = reg.Execute(fixed, moving)

    resampled = sitk.Resample(
        moving, fixed, final_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32,
    )
    return sitk.GetArrayFromImage(resampled).astype(np.float32)


def registered_cache_path(identifier: str, cache_dir: Optional[str] = None) -> str:
    """
    把样本标识符（如 hf_image_path，形如 "ds000001/sub-01/anat/xxx.nii.gz"）
    映射为配准结果的磁盘缓存文件路径（.npy）。

    训练脚本（data.openmind_dataset.OpenMindDataset）与预处理脚本
    （experiments/register_healthy_subset_mri.py）共用本函数，保证同一个
    样本在两处算出的缓存路径一致。
    """
    cache_dir = cache_dir or DEFAULT_REGISTERED_CACHE_DIR
    safe_name = identifier.replace(os.sep, "__").replace("/", "__")
    return os.path.join(cache_dir, safe_name + ".npy")
