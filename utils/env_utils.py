"""
从项目根目录下的 .env.local 加载本地密钥（如 TABPFN_TOKEN）到环境变量。

.env.local 不纳入版本控制（见 .gitignore），Windows 开发机和 HPC 服务器上
各自维护自己的一份，内容形如：
    TABPFN_TOKEN=tabpfn_sk_xxxxxxxx
"""
import os


def load_local_env() -> None:
    """
    读取项目根目录下的 .env.local（若存在），将其中的 KEY=VALUE 写入 os.environ。
    已存在的环境变量不会被覆盖（优先使用外部显式设置的值）。
    文件不存在时静默跳过，不报错。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root, ".env.local")
    if not os.path.exists(env_path):
        return
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
