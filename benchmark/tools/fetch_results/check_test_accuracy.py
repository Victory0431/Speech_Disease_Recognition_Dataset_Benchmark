import pandas as pd
import os
import csv

def find_max_test_accuracy(file_path):
    """从单个结果文件中提取最高测试准确率及对应的epoch"""
    try:
        # 读取文本文件，使用制表符作为分隔符
        df = pd.read_csv(file_path, sep='\t')
        
        # 检查是否包含所需列
        if 'Test Accuracy(%)' not in df.columns or 'Epoch' not in df.columns:
            return None, "文件缺少必要的列（Test Accuracy(%) 或 Epoch）"
        
        # 提取测试集准确率列
        test_accuracies = df['Test Accuracy(%)']
        
        # 找到最大值及其对应的epoch
        max_accuracy = test_accuracies.max()
        max_epoch = df[test_accuracies == max_accuracy]['Epoch'].values[0]
        
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
    result_dir = '/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/tools/fetch_results'
    output_csv = os.path.join(result_dir, "cnn_test_accuracy_summary.csv")
    save_results_to_csv(analysis_results, output_csv)
    