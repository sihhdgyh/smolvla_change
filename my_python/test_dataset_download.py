import pandas as pd
import pathlib
from tqdm import tqdm


def check_parquet_files(directory):
    path = pathlib.Path(directory)
    # 递归查找目录下所有的 parquet 文件
    files = list(path.rglob("*.parquet"))

    if not files:
        print(f"❌ 在 {directory} 下没找到任何 .parquet 文件")
        return

    print(f"🔍 找到 {len(files)} 个文件，开始检查...")
    corrupt_files = []

    for f in tqdm(files):
        try:
            # 尝试只读取一行数据来验证解压是否正常
            pd.read_parquet(f, engine='pyarrow').head(1)
        except Exception as e:
            print(f"\n❌ 发现损坏文件: {f}")
            print(f"错误信息: {e}")
            corrupt_files.append(f)

    print("\n" + "=" * 50)
    if corrupt_files:
        print(f"总结：检测完毕，共发现 {len(corrupt_files)} 个损坏文件。")
        for cf in corrupt_files:
            print(f"- {cf}")
        print("\n建议：删除上述文件并重新下载。")
    else:
        print("✅ 所有文件读取正常，没有发现 Snappy 压缩损坏问题。")


if __name__ == "__main__":
    # 指向你报错信息里的数据集路径
    dataset_path = "/root/autodl-tmp/lerobot/datasets/metaworld_mt50"
    check_parquet_files(dataset_path)