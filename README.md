# AutoHelpdesk

智能故障诊断与解决方案系统，基于机器学习和自然语言处理技术，提供自动化的故障分类和解决方案推荐。

## 项目概述

AutoHelpdesk是一个集成了文本分类、相似度计算和Web服务的智能故障处理系统。该系统能够分析用户提交的故障描述，自动分类故障类型，并返回最相关的解决方案。

主要功能：
- 中文故障描述文本分类
- 基于TF-IDF和余弦相似度的解决方案匹配
- 简洁的Web用户界面
- MySQL数据库集成

## 项目结构

```
AutoHelpdesk/
├── fault_classifier.pkl       # 故障分类模型
├── label_encoder.pkl          # 标签编码器
├── qatosql.ipynb              # 数据处理和模型训练脚本
├── question_solution.csv      # 问题解决方案数据集
├── templates/
│   └── index.html             # Web界面模板
├── tfidf_vectorizer.pkl       # TF-IDF向量器
├── type_question_vectors.pkl  # 问题类型向量
└── webapp.py                  # Flask Web应用
```

## 技术栈

- **后端框架**: Flask
- **机器学习**: scikit-learn, jieba
- **数据库**: MySQL
- **数据处理**: pandas, NumPy
- **Web界面**: HTML, JavaScript

## 安装与设置

### 前提条件
- Python 3.7+
- MySQL
- pip (Python包管理器)

### 安装步骤

1. 克隆仓库
```bash
git clone <repository-url>
cd AutoHelpdesk
```

2. 安装依赖
```bash
pip install flask scikit-learn pandas numpy jieba pymysql joblib
```

3. 数据库设置
   - 创建MySQL数据库（默认名称: qapairs）
   - 导入数据库表结构和数据

4. 配置数据库连接
   修改`webapp.py`中的数据库连接参数：
```python
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='your_password',
    database='qapairs',
    charset='utf8mb4'
)
```

## 使用方法

### 启动Web服务
```bash
python webapp.py
```

### 访问系统
打开浏览器访问: http://127.0.0.1:5000/

### 使用流程
1. 在文本框中输入故障描述
2. 系统自动分类故障类型并查找最佳解决方案
3. 查看返回的解决方案

## 模型训练

如果需要重新训练模型，可以运行Jupyter Notebook：
```bash
jupyter notebook qatosql.ipynb
```
该脚本包含：
- 数据清洗和预处理
- 中文分词和TF-IDF向量化
- SVM模型训练
- 模型保存

## 故障类型
系统目前支持以下故障类型分类：
- 硬件类
- 软件类
- 网络类
- 系统类

## 许可证
[MIT](LICENSE)