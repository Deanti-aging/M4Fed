"""
从项目根目录下的 .env.local 加载本地密钥（如 TABPFN_TOKEN）到环境变量，
并将各库的预训练模型下载缓存目录重定向到项目内的 data/model_cache。

.env.local 不纳入版本控制（见 .gitignore），Windows 开发机和 HPC 服务器上
各自维护自己的一份，内容形如：
    TABPFN_TOKEN=tabpfn_sk_xxxxxxxx
"""
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 预训练模型统一缓存目录：HuggingFace Hub（VAE/BioMedCLIP/TabPFN 等）与
# torch.hub（SwinUNETR SSL 预训练权重）都落在此目录下的各自子目录中，
# 不再散落到宿主用户目录 ~/.cache，便于随项目迁移、清理和限额管理。
MODEL_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "model_cache")


def load_local_env() -> None:
    """
    读取项目根目录下的 .env.local（若存在），将其中的 KEY=VALUE 写入 os.environ；
    并重定向预训练模型缓存目录（见 _setup_model_cache_dir）。
    已存在的环境变量不会被覆盖（优先使用外部显式设置的值）。
    .env.local 不存在时静默跳过该部分，不报错。
    """
    env_path = os.path.join(_PROJECT_ROOT, ".env.local")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    _setup_model_cache_dir()


def _setup_model_cache_dir() -> None:
    """
    将 HuggingFace Hub（HF_HOME）与 torch.hub（TORCH_HOME）的默认缓存目录
    （均为 ~/.cache/...）重定向到 MODEL_CACHE_DIR 下的子目录。

    必须在 huggingface_hub / torch / open_clip / tabpfn 等库被导入之前调用，
    因为这些库在导入时即读取环境变量确定缓存路径。已被外部显式设置的
    HF_HOME / TORCH_HOME 不会被覆盖。
    """
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(MODEL_CACHE_DIR, "huggingface"))
    os.environ.setdefault("TORCH_HOME", os.path.join(MODEL_CACHE_DIR, "torch"))
