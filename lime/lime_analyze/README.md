# LIME 毒性检测分析工具

这个工具用于分析毒性检测模型，使用 LIME (Local Interpretable Model-agnostic Explanations) 方法识别哪些词汇对模型预测贡献最大。

## 环境要求

### Python 版本
- Python 3.7+

### 依赖库
```bash
pip install pandas numpy scikit-learn detoxify torch matplotlib seaborn
```

### GPU支持（可选）
- 如果有 NVIDIA GPU，建议安装 CUDA 版本的 PyTorch 以加速分析
- 工具会自动检测并使用 GPU

## 工具说明

### 1. toxicity_lime_analysis.py - LIME 分析脚本

分析文本数据，生成每个 token 的 impact 分数。

#### 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--input` | `-i` | 输入CSV文件路径 | `mp1-data-train.csv` |
| `--output` | `-o` | 输出CSV文件路径 | `token_impact_analysis.csv` |
| `--num-examples` | `-n` | 分析的样本数量 | `300` |
| `--model` | `-m` | 使用的预训练模型 (`detoxify` 或 `transformers`) | `detoxify` |

#### 使用示例

**基本用法（使用默认参数）：**
```bash
python toxicity_lime_analysis.py
```

**指定输入和输出文件：**
```bash
python toxicity_lime_analysis.py -i data/my_data.csv -o results/my_results.csv
```

**分析更多样本：**
```bash
python toxicity_lime_analysis.py -i final_version.csv -o analysis_results.csv -n 500
```

**使用 Transformers 模型：**
```bash
python toxicity_lime_analysis.py -m transformers -i data.csv -o results.csv
```

**查看帮助信息：**
```bash
python toxicity_lime_analysis.py --help
```

#### 输入文件格式

CSV 文件需要包含以下列：
- `body`: 文本内容
- `removed`: 毒性标签 (0=非毒性, 1=毒性)

示例：
```csv
body,removed
"This is a nice comment",0
"This is a toxic comment",1
```

#### 输出文件格式

生成的 CSV 文件包含以下列：
- `sample_id`: 样本ID
- `sentence`: 原始句子
- `token_index`: token在句子中的位置
- `token`: 具体的token
- `impact`: LIME impact分数（正值=增加毒性，负值=减少毒性）
- `toxic`: 该样本的毒性标签

示例：
```csv
sample_id,sentence,token_index,token,impact,toxic
0001,"This is toxic",0,This,-0.0234,1
0001,"This is toxic",1,is,-0.0156,1
0001,"This is toxic",2,toxic,0.3456,1
```

### 2. token_impact_visualization.py - 可视化脚本

根据 LIME 分析结果生成可视化图表。

#### 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--input` | `-i` | 输入CSV文件路径 | `lime_impact_final_version.csv` |
| `--output-chart` | `-oc` | 主要图表输出路径 | `token_impact_chart.png` |
| `--output-detailed` | `-od` | 详细分析图表输出路径 | `detailed_token_analysis.png` |
| `--top-n` | `-n` | 显示前N个token | `20` |
| `--min-occurrences` | `-m` | token最小出现次数 | `3` |
| `--figsize` | `-f` | 图表尺寸 (宽 高) | `12 10` |

#### 使用示例

**基本用法：**
```bash
python token_impact_visualization.py
```

**指定输入和输出：**
```bash
python token_impact_visualization.py -i token_impact_analysis.csv -oc chart.png
```

**显示更多 tokens：**
```bash
python token_impact_visualization.py -i analysis.csv -n 30 -m 5
```

**自定义图表尺寸：**
```bash
python token_impact_visualization.py -f 15 12
```

**查看帮助信息：**
```bash
python token_impact_visualization.py --help
```

## 完整工作流程

### 步骤 1: 准备数据
确保您的 CSV 文件包含 `body` 和 `removed` 列。

### 步骤 2: 运行 LIME 分析
```bash
python toxicity_lime_analysis.py -i final_version.csv -o analysis_results.csv -n 300
```

这将分析 300 个样本，生成 `analysis_results.csv` 文件。

### 步骤 3: 生成可视化
```bash
python token_impact_visualization.py -i analysis_results.csv -oc impact_chart.png -n 20
```

这将生成两个图表：
- `impact_chart.png`: 主要的 token impact 条形图
- `detailed_token_analysis.png`: 详细的四宫格分析图

## 性能优化建议

1. **使用 GPU**: 如果有 GPU，分析速度会显著提升
2. **调整样本数**: 从少量样本开始测试，确认无误后再增加
3. **批量处理**: 脚本会自动使用批量处理来提高效率

## 常见问题

### 1. GPU 内存不足
如果遇到 GPU 内存不足，可以：
- 减少 `--num-examples` 参数
- 脚本会自动清理 GPU 内存

### 2. 分析速度慢
- 确认是否正在使用 GPU（脚本启动时会显示）
- 减少分析的样本数量
- 使用 `detoxify` 模型（比 `transformers` 更快）

### 3. 输出文件已存在
脚本会直接覆盖已存在的文件，请注意备份重要数据。

## 模型说明

### Detoxify (默认)
- 更快的推理速度
- 基于 RoBERTa 的毒性检测模型
- 推荐用于大规模分析

### Transformers
- 使用 unitary/unbiased-toxic-roberta
- 更详细的分类结果
- 适合精细化分析

## 输出解读

### Impact 分数含义
- **正值 (> 0)**: 该 token 增加毒性概率
- **负值 (< 0)**: 该 token 减少毒性概率
- **绝对值大小**: 表示影响程度的强弱

### 示例解读
如果 "hate" 的 impact 为 0.45，表示这个词强烈增加了模型预测为毒性的概率。
如果 "kind" 的 impact 为 -0.23，表示这个词降低了模型预测为毒性的概率。

## 技术支持

如遇到问题，请检查：
1. Python 版本是否符合要求
2. 所有依赖库是否正确安装
3. 输入文件格式是否正确
4. GPU 驱动和 CUDA 是否正确安装（如使用 GPU）

## 许可证

本工具基于 LIME 框架开发，用于学术研究目的。

