# 基于朴素贝叶斯分类器的邮件分类系统代码核心功能说明

## 1. 文本预处理
### （1）读取文本并分词
函数 `get_words(filename)` 主要完成文本读取、无效字符过滤、分词以及过滤单字词的操作。

```python
import re
import jieba

def get_words(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
        # 使用正则表达式过滤掉无效字符（如标点符号、数字等）
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        # 使用 jieba.cut 对文本进行中文分词
        words = jieba.cut(text)
        # 过滤掉长度为 1 的词
        filtered_words = [word for word in words if len(word) > 1]
        return filtered_words
```

### （2）建立词库
函数 `get_top_words(top_num)` 遍历邮件文件，统计词频并提取出现次数最多的前 `top_num` 个词作为特征词库。

```python
from collections import Counter

def get_top_words(top_num):
    all_words = []
    for i in range(151):
        filename = f'C:\\Max\\邮件_files\\{i}.txt'
        words = get_words(filename)
        all_words.extend(words)
    # 使用 Counter 统计所有邮件中每个词的出现次数
    word_counter = Counter(all_words)
    # 提取出现次数最多的前 top_num 个词
    top_words = [word for word, _ in word_counter.most_common(top_num)]
    return top_words
```

## 2. 特征提取
对每封邮件，统计其在特征词库中每个词的出现次数，形成词频向量。设特征词库为 \(T = \{t_1, t_2, \cdots, t_n\}\)，对于某封邮件的词汇集合 \(W\)，其词频向量 \(\mathbf{v}=(v_1, v_2, \cdots, v_n)\)，其中 \(v_i\) 表示特征词 \(t_i\) 在 \(W\) 中出现的次数，计算公式为 \(v_i=\text{count}(t_i, W)\)。

```python
import numpy as np

top_words = get_top_words(100)  # 假设提取前 100 个词作为特征词库
all_words = []
for i in range(151):
    filename = f'C:\\Max\\邮件_files\\{i}.txt'
    words = get_words(filename)
    all_words.append(words)

vector = []
for words in all_words:
    word_map = list(map(lambda word: words.count(word), top_words))
    vector.append(word_map)
vector = np.array(vector)
```

## 3. 模型训练
使用 `sklearn` 中的 `MultinomialNB` 训练朴素贝叶斯分类器。朴素贝叶斯分类器基于贝叶斯定理，对于分类问题，假设类别集合为 \(C=\{c_1, c_2\}\)（这里 \(c_1\) 表示垃圾邮件，\(c_2\) 表示普通邮件），对于一个词频向量 \(\mathbf{v}\)，其分类结果由后验概率 \(P(c_j|\mathbf{v})\) 决定，根据贝叶斯定理有：

\[P(c_j|\mathbf{v})=\frac{P(\mathbf{v}|c_j)P(c_j)}{P(\mathbf{v})}\]

在实际应用中，由于 \(P(\mathbf{v})\) 对于所有类别都是相同的，所以只需要比较 \(P(\mathbf{v}|c_j)P(c_j)\) 的大小。

```python
from sklearn.naive_bayes import MultinomialNB

labels = np.array([1]*127 + [0]*24)
model = MultinomialNB()
model.fit(vector, labels)
```

## 4. 预测新邮件
对未知邮件进行预处理，提取其词汇并构建词频向量，然后使用训练好的模型进行分类。

```python
def predict(filename):
    words = get_words(filename)
    current_vector = np.array(tuple(map(lambda word: words.count(word), top_words)))
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'
```

## 5. 测试
对指定的未知邮件文件调用 `predict` 函数，输出其分类结果。

```python
print('151.txt分类情况:{}'.format(predict('C:\\Max\\邮件_files\\151.txt')))
print('152.txt分类情况:{}'.format(predict('C:\\Max\\邮件_files\\152.txt')))
```

# 核心功能总结
- **文本预处理**：读取邮件文本，过滤无效字符，分词，过滤无意义的词。
- **特征提取**：统计每封邮件在特征词库中的词频，形成词频向量。
- **模型训练**：使用朴素贝叶斯分类器训练模型，将邮件分为垃圾邮件和普通邮件。
- **预测新邮件**：对未知邮件进行预处理和分类，输出分类结果。 
