import os
# 1. 强制禁用不稳定的 Xet 协议
os.environ["HF_HUB_DISABLE_XET_DOWNLOAD"] = "1"
# 2. 如果你想尝试极致加速，可以取消下面这行的注释（前提是 pip install hf_transfer）
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

repo_id = "lerobot/metaworld_mt50" #HuggingFaceVLA/libero lerobot/metaworld_mt50
local_dir = "/root/autodl-tmp/lerobot/datasets/metaworld_mt50" #/root/autodl-tmp/lerobot/datasets/libero /root/autodl-tmp/lerobot/datasets/metaworld_mt50

print(f"开始安全下载数据集到: {local_dir}")

try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False, # 确保文件直接下载到目标目录
        resume_download=True,
        max_workers=8 # 增加并发数提高速度
    )
    print("\n✅ 下载完成！")
except Exception as e:
    print(f"\n❌ 下载过程中出错: {e}")
    print("提示：如果遇到连接超时，请尝试开启/关闭 AutoDL 的学术加速后重新运行脚本。")