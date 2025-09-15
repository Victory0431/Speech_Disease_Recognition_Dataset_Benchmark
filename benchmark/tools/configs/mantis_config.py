# 配置参数
class Config:
    # 数据路径
    DATASET_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/Parkinson_KCL_2017"
    RESULTS_DIR = ""
    RESULTS_ROOT = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mantis_new/all_run_datasets"
    # DATASET_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/COVID_19_CNN"
    # DATASET_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets/ICBHI"
    LOCAL_MODEL_PATH = "/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/mantis/model/"
    
    # 音频参数
    SAMPLE_RATE = 8000  # 降采样率
    MODEL_INPUT_LENGTH = 512  # 模型要求的输入长度
    
    # 训练参数
    BATCH_SIZE = 32
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    PATIENCE = 10  # 早停策略的耐心值
    
    
    
    # MLP参数
    HIDDEN_DIM = 128
    DROPOUT_RATE = 0.3