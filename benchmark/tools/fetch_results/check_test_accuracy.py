import pandas as pd
import os
import csv

def convert_to_numeric(value):
    """将值转换为数值类型，处理可能的异常"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def find_max_test_accuracy(file_path):
    """从单个结果文件中提取最高测试准确率及对应的epoch，增加类型检查"""
    try:
        # 读取文本文件，使用制表符作为分隔符
        df = pd.read_csv(file_path, sep='\t', dtype=str)  # 先都按字符串读取，避免自动类型识别错误
        
        # 检查是否包含所需列
        required_columns = ['Test Accuracy(%)', 'Epoch']
        for col in required_columns:
            if col not in df.columns:
                return None, f"文件缺少必要的列：{col}"
        
        # 转换为数值类型
        df['Test Accuracy(%)'] = df['Test Accuracy(%)'].apply(convert_to_numeric)
        df['Epoch'] = df['Epoch'].apply(convert_to_numeric)
        
        # 检查转换后的数据
        if df['Test Accuracy(%)'].isna().all():
            return None, "测试准确率列无法转换为数值"
        if df['Epoch'].isna().all():
            return None, "Epoch列无法转换为数值"
        
        # 移除无效行
        valid_rows = df.dropna(subset=['Test Accuracy(%)', 'Epoch'])
        if valid_rows.empty:
            return None, "没有有效的测试准确率数据行"
        
        # 找到最大值及其对应的epoch
        max_accuracy = valid_rows['Test Accuracy(%)'].max()
        max_epoch = valid_rows[valid_rows['Test Accuracy(%)'] == max_accuracy]['Epoch'].values[0]
        
        return (max_epoch, max_accuracy), None
        
    except Exception as e:
        return None, f"处理文件时出错: {str(e)}"

def process_cnn_directories(root_dir):
    """处理根目录下的所有子文件夹，提取测试结果"""
    results = []
    
    # 检查根目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误：根目录不存在 - {root_dir}")
        return results
    
    # 遍历所有子文件夹
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        
        # 只处理目录
        if not os.path.isdir(subdir_path):
            continue
        
        print(f"正在处理数据集: {subdir}")
        
        # 在子目录中查找txt文件
        txt_files = [f for f in os.listdir(subdir_path) if f.endswith('.txt')]
        
        if not txt_files:
            print(f"  警告：{subdir} 文件夹中没有找到文本文件")
            results.append({
                'dataset_name': subdir,
                'max_epoch': None,
                'max_test_accuracy(%)': None,
                'status': '没有找到文本文件'
            })
            continue
        
        # 假设每个子目录只有一个结果文件
        if len(txt_files) > 1:
            print(f"  警告：{subdir} 文件夹中有多个文本文件，将使用第一个文件: {txt_files[0]}")
        
        file_path = os.path.join(subdir_path, txt_files[0])
        result, error = find_max_test_accuracy(file_path)
        
        if error:
            print(f"  错误：{error}")
            results.append({
                'dataset_name': subdir,
                'max_epoch': None,
                'max_test_accuracy(%)': None,
                'status': error
            })
        else:
            max_epoch, max_accuracy = result
            print(f"  最高测试准确率: {max_accuracy}% 在第 {max_epoch} 轮")
            results.append({
                'dataset_name': subdir,
                'max_epoch': max_epoch,
                'max_test_accuracy(%)': max_accuracy,
                'status': '处理成功'
            })
    
    return results

def save_results_to_csv(results, output_file):
    """将结果保存为CSV文件"""
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\n结果已成功保存到: {output_file}")
    except Exception as e:
        print(f"\n保存结果时出错: {str(e)}")

if __name__ == "__main__":
    # CNN数据集根目录
    root_directory = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mlp"
    
    # 处理所有子目录
    print(f"开始处理根目录: {root_directory}")
    analysis_results = process_cnn_directories(root_directory)
    
    # 输出结果到根目录下的CSV文件
    out_dir = '/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/tools/fetch_results'
    output_csv = os.path.join(out_dir, "mlp_test_accuracy_summary.csv")
    save_results_to_csv(analysis_results, output_csv)
    