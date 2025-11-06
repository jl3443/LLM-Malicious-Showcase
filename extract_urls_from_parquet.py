import pandas as pd
import numpy as np
from pathlib import Path

def extract_urls_from_parquet(parquet_path, output_csv_path, num_urls=2000):
    """
    从parquet文件中提取URL数据并保存为CSV格式
    
    Args:
        parquet_path: parquet文件路径
        output_csv_path: 输出CSV文件路径
        num_urls: 要提取的URL数量
    """
    
    print(f"正在读取parquet文件: {parquet_path}")
    
    try:
        # 读取parquet文件
        df = pd.read_parquet(parquet_path)
        print(f"成功读取parquet文件，总行数: {len(df)}")
        print(f"列名: {list(df.columns)}")
        
        # 显示前几行数据
        print("\n前5行数据:")
        print(df.head())
        
        # 检查是否有URL列
        url_columns = [col for col in df.columns if 'url' in col.lower() or 'link' in col.lower()]
        if not url_columns:
            print("\n警告: 未找到URL相关的列")
            print("可用的列:")
            for i, col in enumerate(df.columns):
                print(f"  {i}: {col}")
            
            # 让用户选择列
            try:
                url_col_idx = int(input("\n请选择包含URL的列索引: "))
                url_column = df.columns[url_col_idx]
            except (ValueError, IndexError):
                print("无效选择，使用第一列")
                url_column = df.columns[0]
        else:
            url_column = url_columns[0]
            print(f"\n使用列 '{url_column}' 作为URL列")
        
        # 提取URL数据
        urls = df[url_column].dropna().astype(str)
        print(f"\n提取到 {len(urls)} 个非空URL")
        
        # 随机采样
        if len(urls) >= num_urls:
            sampled_urls = urls.sample(n=num_urls, random_state=42)
            print(f"随机采样 {num_urls} 个URL")
        else:
            sampled_urls = urls
            print(f"URL数量不足 {num_urls}，使用全部 {len(sampled_urls)} 个URL")
        
        # 创建输出数据框
        output_df = pd.DataFrame({
            'url': sampled_urls,
            'type': 'unknown'  # 默认标签，你可以手动标注或使用其他方法
        })
        
        # 保存为CSV
        output_df.to_csv(output_csv_path, index=False)
        print(f"\n成功保存 {len(output_df)} 个URL到: {output_csv_path}")
        
        # 显示统计信息
        print(f"\n统计信息:")
        print(f"总URL数: {len(output_df)}")
        print(f"平均URL长度: {output_df['url'].str.len().mean():.1f}")
        print(f"最短URL: {output_df['url'].str.len().min()} 字符")
        print(f"最长URL: {output_df['url'].str.len().max()} 字符")
        
        # 显示前10个URL示例
        print(f"\n前10个URL示例:")
        for i, url in enumerate(output_df['url'].head(10)):
            print(f"  {i+1}: {url}")
            
        return output_df
        
    except Exception as e:
        print(f"处理parquet文件时出错: {e}")
        return None

def main():
    # 配置路径
    parquet_path = r"c:\Users\lujun\AppData\Local\Temp\a1a758b0-c444-4cb2-9655-f2c6177e4b78_DeepURLBench-main.zip.b78\DeepURLBench-main\urls_with_dns\part-00003.parquet"
    output_csv_path = "extracted_urls_2000.csv"
    
    print("=" * 60)
    print("从Parquet文件提取URL数据")
    print("=" * 60)
    
    # 检查parquet文件是否存在
    if not Path(parquet_path).exists():
        print(f"错误: Parquet文件不存在: {parquet_path}")
        print("\n请提供正确的parquet文件路径:")
        parquet_path = input("Parquet文件路径: ").strip()
        
        if not parquet_path or not Path(parquet_path).exists():
            print("无效路径，退出程序")
            return
    
    # 提取URL
    result_df = extract_urls_from_parquet(parquet_path, output_csv_path)
    
    if result_df is not None:
        print("\n" + "=" * 60)
        print("提取完成！")
        print(f"输出文件: {output_csv_path}")
        print("=" * 60)
        
        # 询问是否要查看文件内容
        try:
            view_file = input("\n是否查看输出文件的前几行? (y/n): ").lower().strip()
            if view_file == 'y':
                print(f"\n{output_csv_path} 的前10行:")
                print(result_df.head(10).to_string(index=False))
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main() 