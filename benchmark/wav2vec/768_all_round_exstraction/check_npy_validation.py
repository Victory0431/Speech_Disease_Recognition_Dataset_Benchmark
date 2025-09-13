import os
import numpy as np
import argparse

def validate_feature_file(file_path):
    """
    验证特征文件是否符合要求
    返回值：(是否通过验证, 错误信息列表)
    """
    errors = []
    
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        errors.append(f"文件不存在: {file_path}")
        return False, errors
    
    # 2. 检查文件格式是否为.npy
    if not file_path.endswith(".npy"):
        errors.append(f"文件格式错误，必须是.npy格式: {file_path}")
        return False, errors
    
    try:
        # 3. 加载文件并检查基本属性
        feats = np.load(file_path, allow_pickle=False)
        
        # 4. 检查是否为二维数组 (样本数 x 特征维度)
        if feats.ndim != 2:
            errors.append(f"特征数组维度错误，应为2维(样本数x特征维度)，实际为{feats.ndim}维")
        
        # 5. 检查特征维度是否为768 (Wav2Vec2的输出维度)
        if feats.ndim == 2 and feats.shape[1] != 768:
            errors.append(f"特征维度错误，应为768，实际为{feats.shape[1]}")
        
        # 6. 检查是否包含有效数据（非全零）
        if np.allclose(feats, 0):
            errors.append("特征文件全为零值，可能提取过程出错")
        
        # 7. 检查数据类型是否为float32
        if feats.dtype != np.float32:
            errors.append(f"数据类型错误，应为float32，实际为{feats.dtype}")
        
        # 8. 检查是否有NaN或无穷大值
        if np.isnan(feats).any():
            errors.append("特征文件包含NaN值")
        if np.isinf(feats).any():
            errors.append("特征文件包含无穷大值")
        
        # 9. 打印基本信息（供参考）
        print("="*50)
        print(f"特征文件信息: {os.path.basename(file_path)}")
        print(f"样本数量: {feats.shape[0]}")
        print(f"特征维度: {feats.shape[1] if feats.ndim == 2 else 'N/A'}")
        print(f"数据类型: {feats.dtype}")
        print(f"数值范围: [{feats.min():.4f}, {feats.max():.4f}]")
        print("="*50)
        
    except Exception as e:
        errors.append(f"加载文件时出错: {str(e)}")
        return False, errors
    
    # 判断是否通过验证
    if len(errors) == 0:
        return True, ["文件验证通过，符合预期格式和内容要求"]
    else:
        return False, errors

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="验证音频特征文件是否符合MLP分类要求")
    parser.add_argument("--file", required=True, 
                        help="特征文件路径，例如: /path/to/Asthma_Detection_Tawfik__and__asthma.npy")
    args = parser.parse_args()
    
    # 执行验证
    is_valid, messages = validate_feature_file(args.file)
    
    # 输出结果
    print("\n验证结果:")
    if is_valid:
        print("✅ " + messages[0])
        print("可以投入后续MLP分类流程")
    else:
        print("❌ 发现以下问题:")
        for msg in messages:
            print(f"- {msg}")
        print("请检查特征提取流程并重新生成文件")
