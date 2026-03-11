"""
SAM3 模型测试脚本
测试 SAM3 模型在 MaaS 平台中的集成
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image
from xt_maas.models.cv.object_detection.sam3 import SAM3Model

def test_sam3_model():
    """测试 SAM3 模型"""
    
    # 模型路径
    model_path = "/data_ssd/maas_models/weight/Sam3/sam3.pt"
    image_path = "/data_ssd/workspace_qzh/sam3_code/media/images/f22_2.jpg"
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在：{model_path}")
        return False
    
    if not os.path.exists(image_path):
        print(f"错误：测试图像不存在：{image_path}")
        return False
    
    try:
        # 1. 初始化模型
        print("=" * 60)
        print("步骤 1: 初始化 SAM3 模型")
        print("=" * 60)
        model = SAM3Model(
            model_path=model_path,
            device="cuda:2",
            enable_cuda=True,
            box_threshold=0.5
        )
        print("✓ 模型初始化成功\n")
        
        # 2. 加载测试图像
        print("=" * 60)
        print("步骤 2: 加载测试图像")
        print("=" * 60)
        image = Image.open(image_path).convert("RGB")
        print(f"图像尺寸：{image.size}, 模式：{image.mode}\n")
        
        # 3. 执行推理
        print("=" * 60)
        print("步骤 3: 执行推理")
        print("=" * 60)
        categories = ["airplane"]
        print(f"检测类别：{categories}")
        
        result = model.predict(image, categories=categories)
        
        # 4. 打印结果
        print("\n" + "=" * 60)
        print("步骤 4: 推理结果")
        print("=" * 60)
        print(f"检测到目标数量：{result['count']}")
        print(f"推理时间：{result['processing_time']:.3f} 秒")
        print(f"成功：{result['success']}")
        
        if result['count'] > 0:
            print("\n检测结果:")
            for i, detection in enumerate(result['detections']):
                print(f"  目标 {i+1}:")
                print(f"    类别：{detection['class']}")
                print(f"    置信度：{detection['conf']:.4f}")
                print(f"    边界框：{detection['bbox']}")
                if detection.get('mask'):
                    print(f"    掩码：有")
        else:
            print("未检测到目标")
        
        print("\n" + "=" * 60)
        print("测试完成!")
        print("=" * 60)
        
        return result['success']
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_sam3_model()
    if success:
        print("\n✓ SAM3 模型测试通过!")
        sys.exit(0)
    else:
        print("\n✗ SAM3 模型测试失败!")
        sys.exit(1)
