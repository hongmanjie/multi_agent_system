import subprocess
import re

def get_gpu_memory_usage_nvidia_smi():
    """使用 nvidia-smi 获取每张 GPU 的显存使用情况"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free', 
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            # 解析每一行，例如: 0, NVIDIA GeForce RTX 3090, 24576, 12345, 12231
            parts = [p.strip() for p in re.split(r',\s*', line)]
            if len(parts) >= 5:
                idx = int(parts[0])
                name = parts[1]
                total_mem = int(parts[2])  # MB
                used_mem = int(parts[3])   # MB
                free_mem = int(parts[4])   # MB
                
                gpu_info.append({
                    'device_id': idx,
                    'device_name': name,
                    'total_memory_MB': total_mem,
                    'used_memory_MB': used_mem,
                    'free_memory_MB': free_mem,
                })
        return gpu_info
    except Exception as e:
        print(f"调用 nvidia-smi 失败: {e}")
        return []