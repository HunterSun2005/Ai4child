import os
import yaml
import argparse
import pandas as pd
from tqdm import tqdm

# 导入你写的特征工程代码
from evgs_feature_engineering_v2 import process_video_directory

def load_config(yaml_path):
    """读取 YAML 配置文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="从 Track12 数据集提取 EVGS 步态特征")
    parser.add_argument("--config", "-c", type=str, default="configs/track12_multitask_b0.yaml", help="YAML 配置文件路径")
    parser.add_argument("--side", "-s", type=str, default="left", choices=["left", "right"], help="指定患病/目标侧 (默认: left)")
    args = parser.parse_args()

    # 1. 从 YAML 解析数据集路径
    try:
        config = load_config(args.config)
        dataset_root = config.get("paths", {}).get("dataset_root", "./dataset")
    except Exception as e:
        print(f"读取 YAML 配置失败: {e}")
        return

    if not os.path.exists(dataset_root):
        print(f"错误：找不到数据集目录 '{dataset_root}'。请确保路径正确且数据已解压。")
        return

    print(f"配置加载成功！数据集根目录为: {dataset_root}")
    
    all_features = []
    video_folders = []
    
    # 2. 遍历数据集目录
    # 逻辑：深度遍历 dataset_root，只要某个文件夹里面有 .json 文件，
    # 就认为这个文件夹是一个视频的所有帧序列。
    print("正在扫描包含 JSON 文件的视频目录...")
    for root, dirs, files in os.walk(dataset_root):
        json_files = [f for f in files if f.endswith('.json')]
        if len(json_files) > 0:
            video_folders.append(root)

    print(f"扫描完毕，共发现 {len(video_folders)} 个视频序列。")

    # 3. 逐个视频调用 EVGS 特征提取
    for folder in tqdm(video_folders, desc="提取 EVGS 特征"):
        # 获取文件夹名称（通常是 subject_id 或 video_name）
        video_id = os.path.basename(folder)
        # 获取父文件夹名 (比如 train 或 test) 以防命名冲突
        parent_id = os.path.basename(os.path.dirname(folder))
        full_id = f"{parent_id}/{video_id}"
        
        try:
            # 调用你的 EVGS 处理核心函数
            features = process_video_directory(
                video_dir=folder, 
                side=args.side, 
                fps=30  # 这里设为默认，你的代码内部 load_frame_json 会尝试读真实 fps
            )
            
            if features:
                features['video_id'] = full_id
                all_features.append(features)
        except Exception as e:
            print(f"\n处理 {full_id} 时出错: {e}")

    # 4. 汇总保存
    if all_features:
        df = pd.DataFrame(all_features)
        # 将 video_id 移动到第一列
        cols = ['video_id'] + [c for c in df.columns if c != 'video_id']
        df = df[cols]
        
        output_csv = "evgs_features_dataset.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n特征提取完成！已成功提取 {len(all_features)} 个视频的特征，保存至 {output_csv}")
    else:
        print("\n未提取到任何特征。请检查数据集下的 json 文件格式是否与 COCO-WholeBody 格式匹配。")

if __name__ == "__main__":
    # 确保在脚本所在目录执行
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()