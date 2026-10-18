#!/usr/bin/env python3
"""把 conda install 的新包合并进 environment.yml，保留手动编辑的内容。"""
import subprocess
import sys
from pathlib import Path

ENV_FILE = Path("environment.yml")
TMP_FILE = Path("/tmp/current-env.yml")

# 导出当前环境的 conda 包（--from-history 只记录显式安装的）
subprocess.run(
    ["conda", "env", "export", "-p", "./.venv", "--from-history"],
    stdout=TMP_FILE.open("w"),
    check=True,
)

# 读取现有 environment.yml 的 dependencies
import yaml

with ENV_FILE.open() as f:
    env_data = yaml.safe_load(f)
with TMP_FILE.open() as f:
    current_data = yaml.safe_load(f)

existing = set(env_data.get("dependencies", []))
current = set(current_data.get("dependencies", []))

# 找出新增的 conda 包
new_pkgs = current - existing
if new_pkgs:
    print("发现新包：", new_pkgs)
    # 在 pip 之前插入新包
    deps = env_data["dependencies"]
    pip_idx = deps.index("pip") if "pip" in deps else len(deps)
    for pkg in sorted(new_pkgs):
        deps.insert(pip_idx, pkg)
        pip_idx += 1
    with ENV_FILE.open("w") as f:
        yaml.dump(env_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print("已更新 environment.yml")
else:
    print("没有新包")
