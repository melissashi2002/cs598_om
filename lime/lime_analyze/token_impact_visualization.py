#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token Impact可视化脚本
基于token_impact_analysis.csv数据生成水平条形图
显示正面和负面impact的token分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TokenImpactVisualizer:
    def __init__(self, csv_file):
        """初始化可视化器"""
        self.csv_file = csv_file
        self.df = None
        self.token_impacts = None
        
    def load_data(self):
        """加载CSV数据"""
        print("正在加载token impact数据...")
        try:
            # 尝试不同的编码格式
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
            self.df = None
            
            for encoding in encodings:
                try:
                    print(f"尝试使用 {encoding} 编码...")
                    self.df = pd.read_csv(self.csv_file, encoding=encoding)
                    print(f"成功使用 {encoding} 编码加载数据")
                    break
                except UnicodeDecodeError:
                    print(f"{encoding} 编码失败，尝试下一个...")
                    continue
            
            if self.df is None:
                raise Exception("所有编码格式都无法读取文件")
                
            print(f"数据加载成功: {len(self.df)} 行记录")
            print(f"数据列: {self.df.columns.tolist()}")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def aggregate_token_impacts(self, min_occurrences=3):
        """聚合每个token的impact值"""
        print("正在聚合token impact数据...")
        
        # 清理token数据
        self.df['token_clean'] = self.df['token'].str.lower().str.strip('.,!?;:"()[]{}')
        
        # 检查是否有toxic_prob列，如果没有则使用toxic列
        if 'toxic_prob' in self.df.columns:
            prob_col = 'toxic_prob'
        elif 'toxic' in self.df.columns:
            prob_col = 'toxic'
        else:
            prob_col = None
        
        # 按token聚合impact值
        agg_dict = {
            'impact': ['mean', 'count', 'std']
        }
        if prob_col:
            agg_dict[prob_col] = 'mean'
        
        token_stats = self.df.groupby('token_clean').agg(agg_dict).round(4)
        
        # 扁平化列名
        if prob_col:
            token_stats.columns = ['mean_impact', 'count', 'std_impact', 'avg_toxic_prob']
        else:
            token_stats.columns = ['mean_impact', 'count', 'std_impact']
            token_stats['avg_toxic_prob'] = 0.5  # 默认值
        token_stats = token_stats.reset_index()
        
        # 过滤出现次数太少的token
        token_stats = token_stats[token_stats['count'] >= min_occurrences]
        
        # 按平均impact排序
        token_stats = token_stats.sort_values('mean_impact', ascending=True)
        
        print(f"聚合完成: {len(token_stats)} 个有效token (出现次数 >= {min_occurrences})")
        
        # 打印一些统计信息用于调试
        print(f"Impact值范围: {token_stats['mean_impact'].min():.4f} 到 {token_stats['mean_impact'].max():.4f}")
        print(f"正面impact token数量: {len(token_stats[token_stats['mean_impact'] > 0])}")
        print(f"负面impact token数量: {len(token_stats[token_stats['mean_impact'] < 0])}")
        
        return token_stats
    
    def create_impact_chart(self, top_n=20, figsize=(12, 10), save_path='token_impact_chart.png'):
        """创建token impact水平条形图"""
        print(f"正在创建token impact图表 (显示前{top_n}个token)...")
        
        # 聚合数据
        token_stats = self.aggregate_token_impacts()
        
        # 选择top N个token（正负impact各取一半）
        positive_tokens = token_stats[token_stats['mean_impact'] > 0].nlargest(top_n//2, 'mean_impact')
        negative_tokens = token_stats[token_stats['mean_impact'] < 0].nsmallest(top_n//2, 'mean_impact')
        
        # 合并数据
        selected_tokens = pd.concat([negative_tokens, positive_tokens]).sort_values('mean_impact')
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 设置颜色
        colors = ['#1f77b4' if x < 0 else '#d62728' for x in selected_tokens['mean_impact']]
        
        # 创建水平条形图
        bars = ax.barh(range(len(selected_tokens)), 
                      selected_tokens['mean_impact'],
                      color=colors,
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=0.5)
        
        # 设置y轴标签（token名称）
        ax.set_yticks(range(len(selected_tokens)))
        ax.set_yticklabels(selected_tokens['token_clean'], fontsize=10)
        
        # 设置x轴
        ax.set_xlabel('Mean LIME Impact on Toxic Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Token', fontsize=12, fontweight='bold')
        
        # 添加标题
        ax.set_title('Directional Token LIME Impact (Red = increases toxicity, Blue = decreases)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 添加网格线
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 添加零线
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        
        # 显示图表
        plt.show()
        
        # 打印统计信息
        self.print_token_statistics(selected_tokens)
        
        return selected_tokens
    
    def print_token_statistics(self, token_stats):
        """打印token统计信息"""
        print("\n=== Token Impact 统计信息 ===")
        
        positive_tokens = token_stats[token_stats['mean_impact'] > 0]
        negative_tokens = token_stats[token_stats['mean_impact'] < 0]
        
        print(f"\n增加毒性的token (正面impact):")
        for _, row in positive_tokens.iterrows():
            print(f"  {row['token_clean']:15} | Impact: {row['mean_impact']:7.4f} | 出现次数: {row['count']:3.0f}")
        
        print(f"\n减少毒性的token (负面impact):")
        for _, row in negative_tokens.iterrows():
            print(f"  {row['token_clean']:15} | Impact: {row['mean_impact']:7.4f} | 出现次数: {row['count']:3.0f}")
        
        print(f"\n总览:")
        print(f"  - 最大正面impact: {token_stats['mean_impact'].max():.4f}")
        print(f"  - 最大负面impact: {token_stats['mean_impact'].min():.4f}")
        print(f"  - 平均impact: {token_stats['mean_impact'].mean():.4f}")
    
    def create_detailed_analysis(self, save_path='detailed_token_analysis.png'):
        """创建详细的分析图表"""
        print("正在创建详细分析图表...")
        
        # 聚合数据
        token_stats = self.aggregate_token_impacts(min_occurrences=2)
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 主要impact分布
        top_positive = token_stats[token_stats['mean_impact'] > 0].nlargest(10, 'mean_impact')
        top_negative = token_stats[token_stats['mean_impact'] < 0].nsmallest(10, 'mean_impact')
        selected_tokens = pd.concat([top_negative, top_positive]).sort_values('mean_impact')
        
        colors1 = ['#1f77b4' if x < 0 else '#d62728' for x in selected_tokens['mean_impact']]
        ax1.barh(range(len(selected_tokens)), selected_tokens['mean_impact'], color=colors1, alpha=0.8)
        ax1.set_yticks(range(len(selected_tokens)))
        ax1.set_yticklabels(selected_tokens['token_clean'], fontsize=9)
        ax1.set_title('Top Token Impacts (Red=Positive, Blue=Negative)', fontweight='bold')
        ax1.set_xlabel('Mean Impact')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Impact分布直方图
        ax2.hist(token_stats['mean_impact'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Distribution of Token Impacts', fontweight='bold')
        ax2.set_xlabel('Mean Impact')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        ax2.grid(alpha=0.3)
        
        # 3. 出现次数 vs Impact
        scatter = ax3.scatter(token_stats['count'], token_stats['mean_impact'], 
                            c=token_stats['mean_impact'], cmap='RdBu_r', alpha=0.6, s=50)
        ax3.set_xlabel('Token Frequency')
        ax3.set_ylabel('Mean Impact')
        ax3.set_title('Token Frequency vs Impact', fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Impact Value')
        
        # 4. 最高正面impact的token
        positive_impacts = token_stats[token_stats['mean_impact'] > 0].nlargest(15, 'mean_impact')
        colors4 = ['#d62728' if x > 0 else '#1f77b4' for x in positive_impacts['mean_impact']]
        ax4.barh(range(len(positive_impacts)), positive_impacts['mean_impact'], 
                color=colors4, alpha=0.7)
        ax4.set_yticks(range(len(positive_impacts)))
        ax4.set_yticklabels(positive_impacts['token_clean'], fontsize=9)
        ax4.set_title('Tokens with Highest Positive Impact', fontweight='bold')
        ax4.set_xlabel('Mean Impact')
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"详细分析图表已保存: {save_path}")
        plt.show()

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Token Impact可视化分析脚本')
    parser.add_argument('--input', '-i', type=str, default='lime_impact_final_version.csv',
                        help='输入CSV文件路径 (默认: lime_impact_final_version.csv)')
    parser.add_argument('--output-chart', '-oc', type=str, default='token_impact_chart.png',
                        help='主要图表输出路径 (默认: token_impact_chart.png)')
    parser.add_argument('--output-detailed', '-od', type=str, default='detailed_token_analysis.png',
                        help='详细分析图表输出路径 (默认: detailed_token_analysis.png)')
    parser.add_argument('--top-n', '-n', type=int, default=20,
                        help='显示前N个token (默认: 20)')
    parser.add_argument('--min-occurrences', '-m', type=int, default=3,
                        help='token最小出现次数 (默认: 3)')
    parser.add_argument('--figsize', '-f', type=int, nargs=2, default=[12, 10],
                        help='图表尺寸 (默认: 12 10)')
    
    return parser.parse_args()

def main():
    """主函数"""
    print("=== Token Impact 可视化分析 ===\n")
    
    # 解析命令行参数
    args = parse_arguments()
    
    print(f"输入文件: {args.input}")
    print(f"主要图表输出: {args.output_chart}")
    print(f"详细分析输出: {args.output_detailed}")
    print(f"显示前N个token: {args.top_n}")
    print(f"最小出现次数: {args.min_occurrences}")
    print(f"图表尺寸: {args.figsize}")
    print()
    
    # 创建可视化器
    visualizer = TokenImpactVisualizer(args.input)
    
    # 加载数据
    if not visualizer.load_data():
        return
    
    try:
        # 创建主要图表
        print("\n" + "="*50)
        print("生成主要Token Impact图表...")
        token_stats = visualizer.create_impact_chart(
            top_n=args.top_n, 
            figsize=tuple(args.figsize),
            save_path=args.output_chart
        )
        
        # 创建详细分析
        print("\n" + "="*50)
        print("生成详细分析图表...")
        visualizer.create_detailed_analysis(save_path=args.output_detailed)
        
        print("\n" + "="*50)
        print("分析完成! 生成的文件:")
        print(f"- {args.output_chart} (主要Token Impact图表)")
        print(f"- {args.output_detailed} (详细分析图表)")
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
