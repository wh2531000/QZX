import re
import numpy as np
from jieba import cut
from itertools import chain
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE  # 新增SMOTE
from sklearn.metrics import classification_report  # 新增评估报告


def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            line = cut(line)
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words


all_words = []


def get_top_words(top_num):
    """获取高频词"""
    filename_list = [f'邮件_files/{i}.txt' for i in range(151)]
    for filename in filename_list:
        all_words.append(get_words(filename))
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]


def main():
    # 1. 数据准备
    train_files = [f'邮件_files/{i}.txt' for i in range(151)]
    test_files = [f'邮件_files/{i}.txt' for i in range(151, 156)]

    # 2. 特征工程
    top_words = get_top_words(100)

    # 构建训练特征
    vector = []
    for words in all_words:
        word_map = list(map(lambda word: words.count(word), top_words))
        vector.append(word_map)
    X_train = np.array(vector)
    y_train = np.array([1] * 127 + [0] * 24)  # 原始标签

    # 3. 样本平衡处理（新增核心代码）
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # 4. 模型训练
    model = MultinomialNB()
    model.fit(X_res, y_res)

    # 5. 构建测试特征
    X_test = []
    for file in test_files:
        words = get_words(file)
        word_map = list(map(lambda word: words.count(word), top_words))
        X_test.append(word_map)
    X_test = np.array(X_test)

    # 假设测试集的真实标签（需要根据实际情况修改）
    y_test = np.array([1, 1, 0, 0, 1])  # 示例标签

    # 6. 预测与评估（新增核心代码）
    y_pred = model.predict(X_test)
    print("\n========== 分类评估报告 ==========")
    print(classification_report(y_test, y_pred,
                                target_names=["普通邮件", "垃圾邮件"]))

    # 7. 输出预测结果
    print("\n========== 邮件分类结果 ==========")
    for i, file in enumerate(test_files):
        result = "垃圾邮件" if y_pred[i] == 1 else "普通邮件"
        print(f"{file} 分类情况: {result}")


if __name__ == "__main__":
    main()