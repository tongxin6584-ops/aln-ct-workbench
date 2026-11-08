"""每一步统一记录开始、完成、失败和耗时。"""

import logging
import time
from contextlib import contextmanager

LOG = logging.getLogger("aln.pipeline")


@contextmanager
def stage(name):
    started = time.monotonic()
    LOG.info("[开始] %s", name)
    try:
        yield
    except Exception:
        LOG.exception("[失败] %s；请检查该阶段输入和上方报错", name)
        raise
    else:
        LOG.info("[完成] %s；耗时 %.1f 秒", name, time.monotonic() - started)
