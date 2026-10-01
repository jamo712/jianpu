import os
from huggingface_hub import snapshot_download

# 国内镜像（必须放前面）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
SAVE_DIR = r"F:\WorkSpace\ai\jianpu\Qwen2-VL-2B-Instruct"

print(f"正在下载: {MODEL_ID}")
print(f"保存到: {SAVE_DIR}")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=SAVE_DIR,
    local_dir_use_symlinks=False,   # Windows 别用软链
    resume_download=True,           # 断点续传
)

print("下载完成 ✅")
print(f"路径: {SAVE_DIR}")