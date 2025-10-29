#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
毒性检测模型的LIME分析脚本
用于识别哪些词汇对模型预测贡献最大
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# 导入torch（用于GPU检测）
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 导入LIME相关模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lime-experiments-master'))
from explainers import GeneralizedLocalExplainer, data_labels_distances_mapping_text


class TextLimeExplainer:
    """真正的文本级LIME解释器"""
    
    def __init__(self, kernel_fn, num_samples=100, return_mean=True, verbose=True):
        self.kernel_fn = kernel_fn
        self.num_samples = num_samples
        self.return_mean = return_mean
        self.verbose = verbose
    
    def explain_instance(self, text, label, classifier_fn, num_features):
        """对文本进行LIME解释"""
        try:
            # 分词
            tokens = text.split()
            if len(tokens) < 3:
                if self.verbose:
                    print(f"  文本太短，跳过: {len(tokens)} 个token")
                return [], 0.5
            
            # 生成扰动样本
            perturbed_texts, weights = self._generate_perturbations(text, tokens)
            
            # 获取扰动样本的预测
            predictions = []
            for perturbed_text in perturbed_texts:
                try:
                    pred = classifier_fn([perturbed_text])
                    if isinstance(pred, np.ndarray) and pred.ndim > 0:
                        predictions.append(pred[0][label])
                    else:
                        predictions.append(pred[label])
                except:
                    predictions.append(0.5)  # 默认值
            
            predictions = np.array(predictions)
            
            # 计算重要性
            importance_scores = self._calculate_importance(tokens, perturbed_texts, predictions, weights)
            
            # 排序并返回top-k特征
            sorted_features = sorted(importance_scores.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = sorted_features[:num_features]
            
            mean_pred = np.mean(predictions)
            return top_features, mean_pred
            
        except Exception as e:
            if self.verbose:
                print(f"  LIME解释失败: {e}")
            return [], 0.5
    
    def _generate_perturbations(self, original_text, tokens):
        """生成扰动文本样本"""
        perturbed_texts = [original_text]  # 第一个是原文
        weights = [1.0]  # 原文权重最高
        
        for _ in range(self.num_samples - 1):
            # 随机选择要保留的token
            num_keep = np.random.randint(1, len(tokens))
            keep_indices = np.random.choice(len(tokens), num_keep, replace=False)
            
            # 构建扰动文本
            perturbed_tokens = [tokens[i] for i in sorted(keep_indices)]
            perturbed_text = ' '.join(perturbed_tokens)
            
            # 计算权重（保留的token越多，权重越高）
            weight = self.kernel_fn(len(tokens) - num_keep)
            
            perturbed_texts.append(perturbed_text)
            weights.append(weight)
        
        return perturbed_texts, np.array(weights)
    
    def _calculate_importance(self, tokens, perturbed_texts, predictions, weights):
        """计算每个token的重要性"""
        importance = {}
        
        for i, token in enumerate(tokens):
            # 计算包含/不包含该token的预测差异
            with_token = []
            without_token = []
            
            for j, text in enumerate(perturbed_texts):
                if token in text.split():
                    with_token.append(predictions[j])
                else:
                    without_token.append(predictions[j])
            
            if with_token and without_token:
                # 使用加权平均
                importance[i] = np.mean(with_token) - np.mean(without_token)
            else:
                importance[i] = 0.0
        
        return importance

class ToxicityAnalyzer:
    def __init__(self, csv_file):
        """初始化毒性分析器"""
        self.csv_file = csv_file
        self.data = None
        self.vectorizer = None
        self.model = None
        self.lime_explainer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("正在加载数据...")
        self.data = pd.read_csv(self.csv_file)

        # 只取前300行数据
        self.data = self.data.head(300)
        
        # 检查数据基本信息
        print(f"数据形状: {self.data.shape}")
        print(f"列名: {self.data.columns.tolist()}")
        
        # 检查removed字段的分布
        if 'removed' in self.data.columns:
            print(f"毒性标签分布:")
            print(self.data['removed'].value_counts())
            print(f"毒性比例: {self.data['removed'].mean():.3f}")
        
        # 清理文本数据
        print("正在清理文本数据...")
        self.data['body_clean'] = self.data['body'].astype(str).str.replace(r'[^\w\s]', ' ', regex=True)
        self.data['body_clean'] = self.data['body_clean'].str.lower()
        self.data['body_clean'] = self.data['body_clean'].str.strip()
        
        # 移除空文本
        self.data = self.data[self.data['body_clean'].str.len() > 10]
        
        print(f"清理后数据形状: {self.data.shape}")
        
    def vectorize_text(self, method='count', max_features=500):
        """文本向量化"""
        print(f"正在使用{method}方法向量化文本...")
        
        if method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                binary=True
            )
        elif method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.95,
                min_df=2
            )
        
        # 准备特征和标签
        X = self.vectorizer.fit_transform(self.data['body_clean'])
        y = self.data['removed'].astype(int)
        
        print(f"特征矩阵形状: {X.shape}")
        print(f"标签分布: {np.bincount(y)}")
        
        return X, y
    
    def load_pretrained_model(self, model_type='detoxify'):
        """加载预训练的毒性检测模型"""
        print(f"正在加载预训练模型: {model_type}...")
        
        if model_type == 'detoxify':
            try:
                import torch
                from detoxify import Detoxify
                # 加载unbiased模型（强制使用GPU）
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    device = 'cuda'
                    print(f"GPU可用，使用设备: {device}")
                    print(f"GPU名称: {torch.cuda.get_device_name()}")
                    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                else:
                    device = 'cpu'
                    print(f"GPU不可用，使用设备: {device}")
                
                self.detoxify_model = Detoxify('unbiased', device=device)
                print("Detoxify模型加载成功")
                
                # 创建适配器函数
                self.model = self._detoxify_predict_proba
                
            except ImportError:
                print("Detoxify库未安装，请运行: pip install detoxify torch")
                raise
                
        elif model_type == 'transformers':
            try:
                from transformers import pipeline
                # 使用transformers pipeline（GPU加速）
                device_id = 0 if TORCH_AVAILABLE and torch.cuda.is_available() else -1
                self.transformers_model = pipeline(
                    "text-classification",
                    model="unitary/unbiased-toxic-roberta",
                    return_all_scores=True,
                    device=device_id  # 使用GPU
                )
                print("Transformers模型加载成功")
                
                # 创建适配器函数
                self.model = self._transformers_predict_proba
                
            except ImportError:
                print("Transformers库未安装，请运行: pip install transformers torch")
                raise
        
        # 为了兼容LIME，需要创建虚拟的训练/测试集
        # 因为预训练模型不需要训练，但仍需要数据索引
        self.train_indices = list(range(len(self.data)))
        self.test_indices = list(range(len(self.data)))
        
        # 初始化向量化器（LIME需要）
        print("初始化向量化器...")
        X, y = self.vectorize_text()
        
        # GPU内存管理
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        print("预训练模型加载完成")
        return True
    
    @torch.inference_mode()
    def _detoxify_predict_proba(self, texts):
        """Detoxify模型的预测函数适配器（推理模式优化）"""
        # 确保输入是文本列表
        if isinstance(texts, str):
            processed_texts = [texts]
        elif isinstance(texts, (list, tuple)):
            processed_texts = list(texts)
        else:
            # 如果是其他格式，转换为文本
            processed_texts = [str(texts)]
        
        # 使用Detoxify进行批量预测（一次性处理所有文本）
        results = self.detoxify_model.predict(processed_texts)
        
        # 转换输出格式为sklearn格式 [non-toxic_prob, toxic_prob]
        proba_results = []
        for i in range(len(processed_texts)):
            toxic_prob = results['toxicity'][i]
            non_toxic_prob = 1 - toxic_prob
            proba_results.append([non_toxic_prob, toxic_prob])
        
        return np.array(proba_results)
    
    @torch.inference_mode()
    def _transformers_predict_proba(self, texts):
        """Transformers模型的预测函数适配器（推理模式优化）"""
        # 处理输入格式
        if hasattr(texts, 'toarray'):
            processed_texts = []
            for i in range(texts.shape[0]):
                text = self.data.iloc[i % len(self.data)]['body']
                processed_texts.append(text)
        elif isinstance(texts, (list, tuple)):
            processed_texts = list(texts)
        else:
            processed_texts = [str(texts)]
        
        # 使用transformers进行批量预测
        results = []
        for text in processed_texts:
            pred = self.transformers_model(text)
            # 找到toxic和non-toxic的分数
            toxic_score = 0
            non_toxic_score = 0
            for item in pred[0]:
                if item['label'] == 'toxic':
                    toxic_score = item['score']
                elif item['label'] == 'non-toxic':
                    non_toxic_score = item['score']
            results.append([non_toxic_score, toxic_score])
        
        return np.array(results)
    
    def setup_lime_explainer(self):
        """设置LIME解释器"""
        print("正在设置文本级LIME解释器...")
        
        # 定义核函数 (与原始代码保持一致)
        rho = 25
        kernel = lambda d: np.sqrt(np.exp(-(d**2) / rho ** 2))
        
        # 创建文本级LIME解释器
        self.lime_explainer = TextLimeExplainer(
            kernel_fn=kernel,
            num_samples=100,
            return_mean=True,
            verbose=True
        )
        
        print("文本级LIME解释器设置完成")
    
    def _quick_filter_samples(self, num_examples, target_samples=50):
        """快速筛选高毒性/边界样本"""
        # 随机选择候选样本
        all_indices = np.random.choice(len(self.data), min(num_examples, len(self.data)), replace=False)
        
        # 快速预测所有候选样本
        predictions = []
        valid_indices = []
        
        print("正在快速预测候选样本...")
        for i, idx in enumerate(all_indices):
            if i % 50 == 0:
                print(f"  已处理 {i}/{len(all_indices)} 个样本")
            
            text = self.data.iloc[idx]['body']
            
            # 跳过过短的文本
            if len(text.split()) < 3:
                continue
            
            try:
                pred = self.model([text])
                if isinstance(pred, np.ndarray) and pred.ndim > 0:
                    toxic_prob = pred[0][1]  # 毒性概率
                    predictions.append(toxic_prob)
                    valid_indices.append(idx)
            except:
                continue
        
        if not predictions:
            return all_indices[:target_samples]
        
        # 选择策略：高毒性 + 边界样本
        predictions = np.array(predictions)
        
        # 高毒性样本 (top 30%)
        high_toxic_threshold = np.percentile(predictions, 70)
        high_toxic_mask = predictions >= high_toxic_threshold
        
        # 边界样本 (0.3-0.7之间)
        boundary_mask = (predictions >= 0.3) & (predictions <= 0.7)
        
        # 合并选择
        selected_mask = high_toxic_mask | boundary_mask
        
        if np.sum(selected_mask) < target_samples:
            # 如果不够，按毒性概率排序选择
            sorted_indices = np.argsort(predictions)[::-1]
            selected_indices = sorted_indices[:target_samples]
        else:
            selected_indices = np.where(selected_mask)[0]
            # 随机选择目标数量
            if len(selected_indices) > target_samples:
                selected_indices = np.random.choice(selected_indices, target_samples, replace=False)
        
        return [valid_indices[i] for i in selected_indices]
    
    def analyze_token_impact(self, num_examples=None):
        """分析每个token的impact，生成详细的CSV输出"""
        print("\n正在进行token级别的impact分析...")
        
        # 如果未指定样本数，分析所有数据
        if num_examples is None:
            num_examples = len(self.data)
        
        # 限制分析前300条数据
        num_examples = min(num_examples, 300)
        
        # 先筛后解：快速筛选高毒性/边界样本
        print("正在快速筛选样本...")
        candidate_indices = self._quick_filter_samples(num_examples)
        print(f"筛选出 {len(candidate_indices)} 个候选样本")
        
        sample_indices = candidate_indices
        
        csv_data = []
        
        # 在分析开始前清理GPU内存
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU内存已清理")
        
        # 设置批量大小
        batch_size = 10 if TORCH_AVAILABLE and torch.cuda.is_available() else 5
        
        for i, data_idx in enumerate(sample_indices):
            text = self.data.iloc[data_idx]['body']
            toxic_label = int(self.data.iloc[data_idx]['removed'])
            
            print(f"分析样本 {i+1}/{num_examples}: {text[:50]}...")
            
            # GPU内存监控（减少频率）
            if TORCH_AVAILABLE and torch.cuda.is_available() and (i+1) % 25 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU内存使用: {gpu_memory:.2f} GB")
            
            try:
                # 直接对文本进行LIME解释
                explanation, mean = self.lime_explainer.explain_instance(
                    text,  # 直接传入文本
                    1,  # 解释毒性类别
                    self.model,  # 直接使用预训练模型
                    num_features=min(5, len(text.split()))  # 进一步限制特征数量
                )
                
                # 获取模型预测
                pred_result = self.model([text])
                if isinstance(pred_result, np.ndarray) and pred_result.ndim > 0:
                    prediction = pred_result[0]
                else:
                    prediction = pred_result
                
                # 将解释转换为字典，方便查找
                impact_dict = {token_idx: impact for token_idx, impact in explanation}
                
                # 分词并分析每个token
                tokens = text.split()
                for token_idx, token in enumerate(tokens):
                    # 直接从字典获取impact
                    token_impact = impact_dict.get(token_idx, 0.0)
                    
                    # 添加到CSV数据
                    csv_data.append({
                        'sample_id': f"{i+1:04d}",
                        'sentence': f'"{text}"',
                        'token_index': token_idx,
                        'token': token,
                        'impact': round(token_impact, 4),
                        'toxic': toxic_label
                    })
                    
            except Exception as e:
                print(f"分析样本 {i+1} 失败: {e}")
                continue
        
        # 分析完成后清理GPU内存
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("分析完成，GPU内存已清理")
        
        return csv_data
    
    def save_token_analysis_csv(self, csv_data, filename='token_impact_analysis.csv'):
        """保存token分析结果到CSV文件"""
        print(f"\n正在保存token分析结果到 {filename}...")
        
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['sample_id', 'sentence', 'token_index', 'token', 'impact', 'toxic']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 写入数据
            for row in csv_data:
                writer.writerow(row)
        
        print(f"CSV文件已保存: {filename}")
        print(f"总共分析了 {len(csv_data)} 个tokens")
        
        # 显示前几行作为示例
        if csv_data:
            print("\nCSV文件前5行示例:")
            print("sample_id,sentence,token_index,token,impact,toxic")
            for i, row in enumerate(csv_data[:5]):
                print(f"{row['sample_id']},{row['sentence']},{row['token_index']},{row['token']},{row['impact']},{row['toxic']}")
    

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='毒性检测模型的LIME分析脚本')
    parser.add_argument('--input', '-i', type=str, default='mp1-data-train.csv',
                        help='输入CSV文件路径 (默认: mp1-data-train.csv)')
    parser.add_argument('--output', '-o', type=str, default='token_impact_analysis.csv',
                        help='输出CSV文件路径 (默认: token_impact_analysis.csv)')
    parser.add_argument('--num-examples', '-n', type=int, default=300,
                        help='分析的样本数量 (默认: 300)')
    parser.add_argument('--model', '-m', type=str, default='detoxify',
                        choices=['detoxify', 'transformers'],
                        help='使用的预训练模型 (默认: detoxify)')
    
    return parser.parse_args()

def main():
    """主函数"""
    print("=== 毒性检测模型LIME分析 ===\n")
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 找不到输入文件 {args.input}")
        return
    
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"分析样本数: {args.num_examples}")
    print(f"使用模型: {args.model}")
    print()
    
    # 创建分析器
    analyzer = ToxicityAnalyzer(args.input)
    
    try:
        # 1. 加载和预处理数据
        analyzer.load_and_preprocess_data()
        
        # 2. 加载预训练模型
        print("\n" + "="*50)
        print("加载预训练模型...")
        analyzer.load_pretrained_model(args.model)
        
        # 3. 设置LIME解释器
        print("\n" + "="*50)
        analyzer.setup_lime_explainer()
        
        # 4. Token级别分析
        print("\n" + "="*50)
        print("进行Token级别分析...")
        csv_data = analyzer.analyze_token_impact(num_examples=args.num_examples)
        analyzer.save_token_analysis_csv(csv_data, args.output)
        
        print(f"\n分析完成! 请查看生成的文件:")
        print(f"- {args.output} (Token级别分析结果)")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
