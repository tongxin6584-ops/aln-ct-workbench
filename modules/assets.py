"""上传云端后核对模型资产，避免权重缺失或复制损坏。"""

import hashlib
import json
from pathlib import Path


def verify_model_assets(project_root):
    root = Path(project_root)
    manifest = json.loads((root / "configs" / "model_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"模型资产缺失: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"模型资产SHA256不匹配: {path}")
    return len(manifest["files"])
