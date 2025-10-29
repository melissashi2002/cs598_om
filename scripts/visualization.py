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
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TokenImpactVisualizer:
    def __init__(self, csv_file='toxic_shap_table_final_version (1).csv'):
        """初始化可视化器"""
        self.csv_file = csv_file
        self.df = None
        self.token_impacts = None
        
    def load_data(self):
        """加载CSV数据"""
        print("正在加载token impact数据...")
        try:
            # 如果给出的文件路径不存在，尝试一些常见的位置（脚本目录、shap/子目录）
            import os
            candidates = [self.csv_file]
            # 脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidates.append(os.path.join(script_dir, self.csv_file))
            # 仓库根下的 shap 子目录
            repo_shap = os.path.join(script_dir, self.csv_file)
            candidates.append(repo_shap)
            # 兼容性: 如果用户运行脚本时位于仓库根目录
            candidates.append(os.path.join(os.getcwd(), 'shap', self.csv_file))

            # 统一去重并保持顺序
            seen = set()
            candidates = [p for p in candidates if not (p in seen or seen.add(p))]

            # 尝试不同的编码格式
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
            self.df = None
            
            # 尝试在候选路径中读取文件，并尝试多种编码
            for path in candidates:
                if not path:
                    continue
                try:
                    if not os.path.exists(path):
                        print(f"文件未找到: {path}")
                        continue
                    for encoding in encodings:
                        try:
                            print(f"尝试加载文件 {path}，使用编码 {encoding} ...")
                            self.df = pd.read_csv(path, encoding=encoding)
                            print(f"成功加载数据: {path} (编码: {encoding})")
                            break
                        except UnicodeDecodeError:
                            # 尝试下一个编码
                            continue
                    if self.df is not None:
                        break
                except Exception as inner_e:
                    print(f"尝试从路径 {path} 读取时发生错误: {inner_e}")
                    continue
            
            if self.df is None:
                raise FileNotFoundError(f"无法找到或读取CSV文件。尝试的路径: {candidates}")
                
            print(f"数据加载成功: {len(self.df)} 行记录")
            print(f"数据列: {self.df.columns.tolist()}")

            # 规范化列名（兼容不同导出格式）
            self._normalize_columns()
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def aggregate_token_impacts(self, min_occurrences=3):
        """聚合每个token的impact值"""
        print("正在聚合token impact数据...")
        
        # 清理token数据
        if 'token' not in self.df.columns:
            raise KeyError("数据中缺少 'token' 列。请检查CSV列名。可用列: {}".format(self.df.columns.tolist()))

        self.df['token_clean'] = self.df['token'].astype(str).str.lower().str.strip('.,!?;:"()[]{}')
        
        # 按token聚合impact值
        # 如果没有 toxic 列，仍然聚合 impact 但 avg_toxic 会为 NaN
        agg_dict = {
            'impact': ['mean', 'count', 'std']
        }
        if 'toxic' in self.df.columns:
            agg_dict['toxic'] = 'mean'

        token_stats = self.df.groupby('token_clean').agg(agg_dict).round(4)
        
        # 扁平化列名
        # 扁平化列名（根据是否存在 toxic 列）
        if 'toxic' in self.df.columns:
            token_stats.columns = ['mean_impact', 'count', 'std_impact', 'avg_toxic']
        else:
            token_stats.columns = ['mean_impact', 'count', 'std_impact']
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
        ax.set_xlabel('Mean SHAP Impact on Toxic Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Token', fontsize=12, fontweight='bold')
        
        # 添加标题
        ax.set_title('Directional Token SHAP Impact (Red = increases toxicity, Blue = decreases)', 
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

    def _normalize_columns(self):
        """规范化DataFrame的列名，映射常见变体到统一列名: token, impact, toxic"""
        cols = {c: c for c in self.df.columns}
        lowered = {c.lower(): c for c in self.df.columns}

        # 可能的 token 列名
        for k in ['token', 'tokens', 'word']:
            if k in lowered:
                cols[lowered[k]] = 'token'
                break

        # 可能的 subreddit 列名
        for k in ['subreddit', 'subreddit_name', 'subreddit_id']:
            if k in lowered:
                cols[lowered[k]] = 'subreddit'
                break

        # 可能的 impact 列名
        for k in ['impact', 'shap_impact_on_toxic', 'shap_impact', 'shap_impact_on_toxicity', 'shapimpact']:
            if k in lowered:
                cols[lowered[k]] = 'impact'
                break

        # 可能的 toxic 列名
        for k in ['toxic', 'label', 'is_toxic', 'toxicity']:
            if k in lowered:
                cols[lowered[k]] = 'toxic'
                break

        # 也支持不同大小写的 'text' 或 'sentence_id' 保留原名
        # 应用重命名
        rename_map = {orig: new for orig, new in cols.items() if orig != new}
        if rename_map:
            try:
                self.df = self.df.rename(columns=rename_map)
                print(f"列名已规范化: {rename_map}")
            except Exception as e:
                print(f"列名规范化失败: {e}")

    def create_subreddit_plots(self, top_k=10, min_occurrences=3, out_dir='subreddit_plots'):
        """为前 top_k 个子版块生成token impact图表并保存

        Each subreddit will get a PNG named <subreddit>_token_impact.png
        """
        import os
        print(f"正在为前 {top_k} 个 subreddit 生成图表 (出现次数阈值={min_occurrences})...")

        if 'subreddit' not in self.df.columns:
            print("数据中没有 'subreddit' 列，跳过子版块图表生成。")
            return []

        # 选出出现次数最多的 top_k subreddit
        top_subs = self.df['subreddit'].value_counts().nlargest(top_k).index.tolist()
        saved_files = []

        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_dir)
        os.makedirs(out_path, exist_ok=True)

        for sub in top_subs:
            try:
                sub_df = self.df[self.df['subreddit'] == sub].copy()
                if sub_df.empty:
                    continue
                # 清理 token 并选择 impact 列名（兼容 'impact' 或原始 'SHAP_Impact_on_Toxic'）
                sub_df['token_clean'] = sub_df['token'].astype(str).str.lower().str.strip('.,!?;:"()[]{}')
                impact_col = 'impact' if 'impact' in sub_df.columns else ('SHAP_Impact_on_Toxic' if 'SHAP_Impact_on_Toxic' in sub_df.columns else None)
                if impact_col is None:
                    print(f"子版块 {sub} 数据中找不到 impact 列，跳过")
                    continue

                agg = sub_df.groupby('token_clean').agg({impact_col: ['mean', 'count']}).round(4)
                agg.columns = ['mean_impact', 'count']
                agg = agg.reset_index()
                agg = agg[agg['count'] >= min_occurrences]
                if agg.empty:
                    print(f"子版块 {sub} 没有足够的token满足出现次数阈值，跳过")
                    continue

                # 选择 top tokens (mix of pos/neg)
                positive = agg[agg['mean_impact'] > 0].nlargest(10, 'mean_impact')
                negative = agg[agg['mean_impact'] < 0].nsmallest(10, 'mean_impact')
                selected = pd.concat([negative, positive]).sort_values('mean_impact')

                # 绘图
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = ['#1f77b4' if x < 0 else '#d62728' for x in selected['mean_impact']]
                ax.barh(range(len(selected)), selected['mean_impact'], color=colors, alpha=0.8)
                ax.set_yticks(range(len(selected)))
                # token 列名可能为 'token_clean' 或 'token'
                if 'token_clean' in selected.columns:
                    labels = selected['token_clean']
                else:
                    labels = selected.iloc[:, 0]
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_title(f"{sub} - Token SHAP Impact", fontweight='bold')
                ax.axvline(x=0, color='black', linewidth=0.8)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()

                fname = f"{sub}_token_impact.png".replace(os.sep, '_')
                fpath = os.path.join(out_path, fname)
                plt.savefig(fpath, dpi=200, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(fpath)
                print(f"已为 {sub} 保存图表: {fpath}")
            except Exception as e:
                print(f"生成子版块 {sub} 图表时出错: {e}")
                continue

        return saved_files

def main():
    """主函数"""
    print("=== Token Impact 可视化分析 ===\n")
    import argparse
    parser = argparse.ArgumentParser(description='Token impact visualization')
    parser.add_argument('--csv', help='Path to CSV file to load', default='toxic_shap_table_from_data.csv')
    args = parser.parse_args()

    # 创建可视化器
    visualizer = TokenImpactVisualizer(csv_file=args.csv)
    
    # 加载数据
    if not visualizer.load_data():
        return
    
    try:
        # 创建主要图表
        print("\n" + "="*50)
        print("生成主要Token Impact图表...")
        token_stats = visualizer.create_impact_chart(
            top_n=20, 
            figsize=(12, 10),
            save_path='token_impact_chart.png'
        )
        
        # 创建详细分析
        print("\n" + "="*50)
        print("生成详细分析图表...")
        visualizer.create_detailed_analysis(save_path='detailed_token_analysis.png')
        
        # 为前10个subreddit生成子图
        print("\n" + "="*50)
        print("为每个子版块生成图表...")
        saved = visualizer.create_subreddit_plots(top_k=10, min_occurrences=3, out_dir='subreddit_plots')
        if saved:
            print(f"为 {len(saved)} 个子版块生成了图表，保存在 shap/subreddit_plots/ 下")
        else:
            print("没有为任何子版块生成图表（可能是缺少'subreddit'列或数据不足）")

        print("\n" + "="*50)
        print("分析完成! 生成的文件:")
        print("- token_impact_chart.png (主要Token Impact图表)")
        print("- detailed_token_analysis.png (详细分析图表)")
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
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
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TokenImpactVisualizer:
    def __init__(self, csv_file='toxic_shap_table_final_version (1).csv'):
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
                    # 尝试规范化列名到预期的 'token' / 'impact' / 'toxic' 等
                    try:
                        self._normalize_columns()
                    except Exception:
                        # 如果规范化失败，不要阻止加载，后续会有更友好的提示
                        pass
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
        
        # 确保列名被规范化
        if 'token' not in self.df.columns:
            # 尝试一些常见的替代列名
            for alt in ['tokens', 'word', 'token_clean', 'Token', 'Words']:
                if alt in self.df.columns:
                    self.df = self.df.rename(columns={alt: 'token'})
                    print(f"已将列 '{alt}' 映射为 'token'")
                    break

        if 'token' not in self.df.columns:
            raise KeyError(f"数据中缺少 'token' 列。可用列: {self.df.columns.tolist()}")

        # 清理token数据（强制转为字符串以避免属性错误）
        self.df['token_clean'] = self.df['token'].astype(str).str.lower().str.strip('.,!?;:"()[]{}')
        
        # 按token聚合impact值
        # 确保 impact 列存在或尝试替代列名
        if 'impact' not in self.df.columns:
            for alt in ['shap_impact_on_toxic', 'shap_impact', 'SHAP_Impact_on_Toxic', 'shapimpact']:
                if alt in self.df.columns:
                    self.df = self.df.rename(columns={alt: 'impact'})
                    print(f"已将列 '{alt}' 映射为 'impact'")
                    break

        agg_cols = {'impact': ['mean', 'count', 'std']}
        if 'toxic' in self.df.columns:
            agg_cols['toxic'] = 'mean'

        token_stats = self.df.groupby('token_clean').agg(agg_cols).round(4)

        # 扁平化列名：处理 MultiIndex 或者 单层列名的情况，并根据是否包含 'toxic' 自动命名
        try:
            if isinstance(token_stats.columns, pd.MultiIndex):
                new_cols = []
                for col in token_stats.columns:
                    # col is a tuple like ('impact','mean') or ('toxic','mean')
                    if col[0] == 'impact' and col[1] == 'mean':
                        new_cols.append('mean_impact')
                    elif col[0] == 'impact' and col[1] == 'count':
                        new_cols.append('count')
                    elif col[0] == 'impact' and col[1] == 'std':
                        new_cols.append('std_impact')
                    elif col[0] == 'toxic' and col[1] == 'mean':
                        new_cols.append('avg_toxic')
                    else:
                        new_cols.append(f"{col[0]}_{col[1]}")
                token_stats.columns = new_cols
            else:
                cols = token_stats.columns.tolist()
                new_cols = []
                for c in cols:
                    lc = str(c).lower()
                    if 'impact' in lc and 'mean' in lc:
                        new_cols.append('mean_impact')
                    elif 'impact' in lc and 'count' in lc:
                        new_cols.append('count')
                    elif 'impact' in lc and ('std' in lc or 'stddev' in lc):
                        new_cols.append('std_impact')
                    elif 'toxic' in lc and 'mean' in lc:
                        new_cols.append('avg_toxic')
                    else:
                        new_cols.append(c)
                token_stats.columns = new_cols
        except Exception:
            # 如果任何情况下命名失败，保留原列名并继续（后续可能抛出更具体的错误）
            pass

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
        ax.set_xlabel('Mean SHAP Impact on Toxic Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Token', fontsize=12, fontweight='bold')
        
        # 添加标题
        ax.set_title('Directional Token SHAP Impact (Red = increases toxicity, Blue = decreases)', 
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

def main():
    """主函数"""
    print("=== Token Impact 可视化分析 ===\n")
    
    # 创建可视化器
    visualizer = TokenImpactVisualizer()
    
    # 加载数据
    if not visualizer.load_data():
        return
    
    try:
        # 创建主要图表
        print("\n" + "="*50)
        print("生成主要Token Impact图表...")
        token_stats = visualizer.create_impact_chart(
            top_n=20, 
            figsize=(12, 10),
            save_path='token_impact_chart.png'
        )
        
        # 创建详细分析
        print("\n" + "="*50)
        print("生成详细分析图表...")
        visualizer.create_detailed_analysis(save_path='detailed_token_analysis.png')
        
        print("\n" + "="*50)
        print("分析完成! 生成的文件:")
        print("- token_impact_chart.png (主要Token Impact图表)")
        print("- detailed_token_analysis.png (详细分析图表)")
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
