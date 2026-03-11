from huggingface_hub import snapshot_download


# # IDEA-Research/Rex-Omni
# snapshot_download(
#     repo_id="IDEA-Research/Rex-Omni",
#     local_dir="/root/wanjie-data/hf_weights/Rex-Omni",
#     resume_download=True,  # 启用断点续传
#     force_download=True    # 强制下载（即使缓存中存在）
# )

# # IDEA-Research/grounding-dino-base
# snapshot_download(
#     repo_id="IDEA-Research/grounding-dino-base",
#     local_dir="/root/wanjie-data/hf_weights/GD_base",
#     resume_download=True,  # 启用断点续传
#     force_download=True    # 强制下载（即使缓存中存在）
# )

# snapshot_download(
#     repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
#     local_dir="/root/wanjie-data/hf_weights/Qwen2_5_VL",
#     resume_download=True,  # 启用断点续传
#     force_download=True    # 强制下载（即使缓存中存在）
# )

# snapshot_download(
#     repo_id="openbmb/MiniCPM-V-4_5",
#     local_dir="/root/wanjie-data/hf_weights/MiniCPM_VL",
#     resume_download=True,  # 启用断点续传
#     force_download=True    # 强制下载（即使缓存中存在）
# )

# snapshot_download(
#     repo_id="Qwen/Qwen3-VL-30B-A3B-Thinking",
#     local_dir="/root/wanjie-data/hf_weights/Qwen3_VL",
#     resume_download=True,  # 启用断点续传
#     force_download=True    # 强制下载（即使缓存中存在）
# )

# Qwen/Qwen3-4B
snapshot_download(
    repo_id="Qwen/Qwen3-4B",
    local_dir="/root/wanjie-data/hf_weights/Qwen3-4B",
    resume_download=True,  # 启用断点续传
    force_download=True    # 强制下载（即使缓存中存在）
)

