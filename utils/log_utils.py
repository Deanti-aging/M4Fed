"""终端输出日志工具：将运行期间的 stdout/stderr 同步写入结果目录下的日志文件，
便于训练结束（或意外中断）后仍能在结果文件夹里查看完整的终端输出。
"""
from __future__ import annotations  # 允许注解中使用尚未定义的类名
import datetime                     # 记录日志起始时间戳
import os                           # 路径与目录操作
import sys                          # 重定向 sys.stdout / sys.stderr


class _Tee:
    """将一次 write 同时转发给多个底层流（如真实终端 + 日志文件），实现"终端可见 + 落盘"双写。"""

    def __init__(self, *streams):
        self._streams = streams   # 待转发的目标流列表（顺序无关）

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except UnicodeEncodeError:
                # 真实终端流的编码取决于系统区域设置（如中文 Windows 默认 GBK），
                # 而项目输出里常出现 GBK 无法编码的数学符号（R²、±、Σ 等）；
                # 日志文件本身固定用 UTF-8 打开不受影响，这里只需让终端这一路
                # 降级为替换非法字符显示，不能让打印失败中断整个训练/评估流程
                encoding = getattr(s, "encoding", None) or "utf-8"
                s.write(data.encode(encoding, errors="replace").decode(encoding))
            s.flush()   # 立即刷新：训练中途崩溃或被强制终止时，已产生的日志不会因缓冲而丢失
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        # 部分库（如 tqdm）会用 isatty() 判断是否为交互终端，Tee 对象本身不是，统一返回 False
        return False


def setup_run_logger(results_dir: str, log_name: str = "run.log") -> str:
    """
    将当前进程的 stdout/stderr 重定向为"终端 + 文件"双写，此后所有 print()/异常回溯
    都会自动同步保存到 {results_dir}/{log_name}，终端显示效果不变。

    以追加模式打开日志文件，多次运行（如重复执行 main.py）的日志会依次累积，
    每次运行开头写入分隔线和时间戳，便于区分。

    幂等：若当前进程已经重定向过（sys.stdout 已是 _Tee），不会重复叠加包裹。

    参数：
      results_dir : 结果保存目录（日志文件写入此目录下，自动创建）
      log_name    : 日志文件名（默认 run.log）

    返回：
      日志文件的完整路径
    """
    os.makedirs(results_dir, exist_ok=True)   # 确保结果目录存在
    log_path = os.path.join(results_dir, log_name)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)   # 行缓冲，配合逐次 flush 及时落盘

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"\n{'=' * 70}\n运行开始：{ts}\n{'=' * 70}\n")

    if not isinstance(sys.stdout, _Tee):   # 避免重复调用时多层嵌套包裹
        sys.stdout = _Tee(sys.stdout, log_file)
    if not isinstance(sys.stderr, _Tee):
        sys.stderr = _Tee(sys.stderr, log_file)

    return log_path
