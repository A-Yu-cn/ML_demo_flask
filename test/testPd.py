import pandas as pd
from src.Model import Model

"""
测试模型是否可用，由于与服务器相对路径不同，现已无法运行，
原因：导入数据时候相对路径，此处允许需要的是'../datasets/cs-training.csv'，而由于服务器根目录下，只需要'datasets/cs-training.csv'
"""


def test1():
    df = pd.DataFrame([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    print(df)
    df = df.append(pd.Series([3, 2, 1]), ignore_index=True)
    print(df)


model = Model()
print(f"第1次评分预测：%s", model.predict([0, 0.766127, 2, 0, 45]))
print(f"第2次评分预测：%s", model.predict([0, 0.957151, 0, 0, 40]))
print(f"第3次评分预测：%s", model.predict([0, 0.658180, 1, 0, 38]))
