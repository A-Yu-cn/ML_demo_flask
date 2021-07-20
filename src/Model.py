import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor  # 随机森林算法将缺失值补充
from sklearn.model_selection import train_test_split  # 数据集划分模块
from scipy.stats import stats  # scipy.stats是一个很好的统计推断包
from sklearn.linear_model import LogisticRegression
import math


def optimal_bins(Y, X, n):
    """
    :param Y: 目标变量
    :param X: 待分箱特征
    :param n: 分箱数初始值
    :return: 统计值、分箱边界值列表、woe值、iv值
    """
    r = 0  # 初始值
    total_bad = Y.sum()  # 总的坏样本数
    total_good = Y.count() - total_bad  # 总的好样本数
    # 分箱过程
    while np.abs(r) < 1:
        df1 = pd.DataFrame({'X': X, 'Y': Y, 'bin': pd.qcut(X, n, duplicates='drop')})  # qcut():基于量化的离散化函数
        df2 = df1.groupby('bin')
        r, p = stats.spearmanr(df2.mean().X, df2.mean().Y)
        n = n - 1
    # 计算woe值和iv值
    df3 = pd.DataFrame()
    df3['min_' + X.name] = df2.min().X
    df3['max_' + X.name] = df2.max().X
    df3['sum'] = df2.sum().Y
    df3['total'] = df2.count().Y
    df3['rate'] = df2.mean().Y
    df3['badattr'] = df3['sum'] / total_bad
    df3['goodattr'] = (df3['total'] - df3['sum']) / total_good
    df3['woe'] = np.log(df3['badattr'] / df3['goodattr'])
    iv = ((df3['badattr'] - df3['goodattr']) * df3['woe']).sum()
    df3 = df3.sort_values(by='min_' + X.name).reset_index(drop=True)
    # 分箱边界值列表
    cut = []
    cut.append(float('-inf'))
    for i in range(1, n + 1):
        qua = X.quantile(i / (n + 1))
        cut.append(round(qua, 6))
    cut.append(float('inf'))
    # woe值列表
    woe = list(df3['woe'])
    return df3, cut, woe, iv


def custom_bins(Y, X, binList):
    """
    :param Y: 目标变量
    :param X: 待分箱特征
    :param binList: 分箱边界值列表
    :return: 统计值、woe值、iv值
    """
    r = 0
    total_bad = Y.sum()
    total_good = Y.count() - total_bad
    # 等距分箱
    df1 = pd.DataFrame({'X': X, 'Y': Y, 'bin': pd.cut(X, binList)})
    df2 = df1.groupby('bin', as_index=True)
    r, p = stats.spearmanr(df2.mean().X, df2.mean().Y)
    # 计算woe值和iv值
    df3 = pd.DataFrame()
    df3['min_' + X.name] = df2.min().X
    df3['max_' + X.name] = df2.max().X
    df3['sum'] = df2.sum().Y
    df3['total'] = df2.count().Y
    df3['rate'] = df2.mean().Y
    df3['badattr'] = df3['sum'] / total_bad
    df3['goodattr'] = (df3['total'] - df3['sum']) / total_good
    df3['woe'] = np.log(df3['badattr'] / df3['goodattr'])
    iv = ((df3['badattr'] - df3['goodattr']) * df3['woe']).sum()
    df3 = df3.sort_values(by='min_' + X.name).reset_index(drop=True)
    woe = list(df3['woe'])

    return df3, woe, iv


# 90D、RevolvingRatio、30-59D、60-89D、Age
class Model:
    data = None
    clf1 = None  # 保存训练模型的情况

    def __init__(self):  # 初始化时候要导入数据进行训练，保存导入的数据以及模型参数以供使用
        self.get_data()
        self.train()

    def predict(self, info_list: list) -> int:
        if self.data is None:  # 如果未导入数据则首先导入数据
            self.get_data()
        # 根据传入的数据信息构造一个字典
        dic = {"Label": 1, "90D": info_list[0], "RevolvingRatio": info_list[1], '30-59D': info_list[2],
               '60-89D': info_list[3], 'Age': info_list[4]}
        # print(self.data)
        # 插入要预测的信息
        self.data = self.data[['Label', '90D', 'RevolvingRatio', '30-59D', '60-89D', 'Age']]
        self.data = self.data.append(dic, ignore_index=True)
        # print(self.data)
        return self.get_score()

    def get_score(self) -> int:
        if self.data is None:  # 如果未导入数据则首先导入数据
            return 0  # 表示错误
        ninf = float('-inf')
        pinf = float('inf')
        cut_thirty = [ninf, 0, 1, 3, 5, pinf]  # 30-59D特征
        cut_open = [ninf, 1, 2, 3, 5, pinf]  # OpenL特征
        cut_ninety = [ninf, 0, 1, 3, 5, pinf]  # 90D特征
        cut_re = [ninf, 0, 1, 2, 3, pinf]  # RealEstate特征
        cut_sixty = [ninf, 0, 1, 3, pinf]  # 60-89D特征
        cut_dpt = [ninf, 0, 1, 2, 3, 5, pinf]  # Dependents特征
        cut_new2 = [ninf, 414, 1209, 2518, pinf]
        # 计算统计值、woe 和iv
        thirtyDf, woe_thirty, iv_thirty = custom_bins(self.data.Label, self.data['30-59D'], cut_thirty)  # 30-59D特征
        ninetyDf, woe_ninety, iv_ninety = custom_bins(self.data.Label, self.data['90D'], cut_ninety)  # 90D特征
        sixtyDf, woe_sixty, iv_sixty = custom_bins(self.data.Label, self.data['60-89D'], cut_sixty)  # 60-89D特征

        ageDf, cut_age, woe_age, iv_age = optimal_bins(self.data.Label, self.data.Age, n=10)
        rrDf, cut_rr, woe_rr, iv_rr = optimal_bins(self.data.Label, self.data.RevolvingRatio, n=10)
        n_data = pd.DataFrame()
        n_data['90D'] = pd.cut(self.data['90D'], bins=cut_ninety, labels=woe_ninety)  # 90D特征
        n_data['RevolvingRatio'] = pd.cut(self.data['RevolvingRatio'], bins=cut_rr,
                                          labels=woe_rr)  # RevolvingRatio特征
        n_data['30-59D'] = pd.cut(self.data['30-59D'], bins=cut_thirty, labels=woe_thirty)  # 30-59D特征
        n_data['60-89D'] = pd.cut(self.data['60-89D'], bins=cut_sixty, labels=woe_sixty)  # 60-89D特征
        n_data['Age'] = pd.cut(self.data['Age'], bins=cut_age, labels=woe_age)  # Age特征
        n_data['Label'] = self.data[['Label']]  # 将标签传递
        # 特征选择
        print(n_data.tail(10))

        X = n_data.iloc[:, 1:]  # 特征
        y = n_data.iloc[:, 0]  # 目标变量
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # 训练集：测试集 = 7:3
        # 计算分值

        # 计算基础分
        B = 20 / math.log(2)
        A = 600 + B * math.log(1 / 20)
        BaseScore = round(A - B * self.clf1.intercept_[0], 0)

        print("评分卡的基础分为：", BaseScore)

        # 每个特征列分值计算函数
        def score(coef, woe):
            """
            :param coef: 特征在逻辑回归模型中对应的参数
            :param woe: 特征的WOE编码取值列表
            :return: 分值
            """
            scores = []
            for x in woe:
                score = round(-B * coef * x, 0)
                scores.append(score)
            return scores

        # 不同特征各个区间对应的分值
        score_ninety = score(self.clf1.coef_[0][0], woe_ninety)  # 90D特征
        print("90D特征各个区间对应的分值为：", score_ninety)
        score_rr = score(self.clf1.coef_[0][1], woe_rr)  # RevolvingRatio特征
        print("RevolvingRatio特征各个区间对应的分值为：", score_rr)
        score_thirty = score(self.clf1.coef_[0][2], woe_thirty)  # 30-59D特征
        print("30-59D特征各个区间对应的分值为：", score_thirty)
        score_sixty = score(self.clf1.coef_[0][3], woe_sixty)  # 60-89D特征
        print("60-89D特征各个区间对应的分值为：", score_sixty)
        score_age = score(self.clf1.coef_[0][4], woe_age)  # Age特征
        print("Age特征各个区间对应的分值为：", score_age)

        # 测试集样本转化为分值形式
        cardDf = X_test.copy()  # 不改变原测试集，在副本上操作
        # 将特征值转化为分值
        n_data['90D'] = n_data['90D'].replace(woe_ninety, score_ninety)
        n_data['RevolvingRatio'] = n_data['RevolvingRatio'].replace(woe_rr, score_rr)
        n_data['30-59D'] = n_data['30-59D'].replace(woe_thirty, score_thirty)
        n_data['60-89D'] = n_data['60-89D'].replace(woe_sixty, score_sixty)
        n_data['Age'] = n_data['Age'].replace(woe_age, score_age)

        print(n_data.head(10))  # 观察此时的测试集副本

        # 计算每个样本的分值
        n_data['Score'] = BaseScore + n_data['90D'] + n_data['RevolvingRatio'] + \
                          n_data['30-59D'] + n_data['60-89D'] + n_data['Age']
        print(n_data.head(10))
        return int(n_data.tail(1).Score)

    def get_data(self):
        """
        导入数据操作，首先需要进行数据导入以及预处理
        :return: None
        """
        self.data = pd.read_csv('datasets/cs-training.csv')
        self.data = self.data.iloc[:, 1:]  # 舍弃Unnamed: 0列
        self.data.columns = ['Label', 'RevolvingRatio', 'Age', '30-59D', 'DebtRatio', 'MonthlyIncome',
                             'OpenL', '90D', 'RealEstate', '60-89D', 'Dependents']  # 列重命名
        # 用MonthlyIncome特征值非空的样本构建训练集，MonthlyIncome特征值缺失的样本构建测试集
        rfDf = self.data.iloc[:, [5, 1, 2, 3, 4, 6, 7, 8, 9]]  # 原始数据集中的无缺失数值特征
        rfDf_train = rfDf.loc[rfDf['MonthlyIncome'].notnull()]
        rfDf_test = rfDf.loc[rfDf['MonthlyIncome'].isnull()]

        # 划分训练数据和标签（label）
        X = rfDf_train.iloc[:, 1:]
        y = rfDf_train.iloc[:, 0]
        # 训练过程
        rf = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)  # 这里重在理解过程，因此仅简单选取部分参数
        rf.fit(X, y)
        # 预测过程
        pred = rf.predict(rfDf_test.iloc[:, 1:]).round(0)  # 预测值四舍五入并保留一位小数点
        self.data.loc[(self.data['MonthlyIncome'].isnull()), 'MonthlyIncome'] = pred  # 填补缺失值
        # Dependents特征处理
        self.data['Dependents'].fillna(self.data['Dependents'].mode()[0], inplace=True)  # 这里采用众数填充
        # 处理百分比类异常值
        # RevolvingRatio特征
        ruulDf = self.data[self.data['RevolvingRatio'] <= 1]  # 去掉高于1的部分
        ruul_mean = ruulDf['RevolvingRatio'].mean()  # 计算均值
        self.data.loc[self.data['RevolvingRatio'] > 1, 'RevolvingRatio'] = ruul_mean  # 均值替代

        # DebtRatio特征
        ruulDf = self.data[self.data['DebtRatio'] <= 1]  # 去掉高于1的部分
        ruul_mean = ruulDf['DebtRatio'].mean()  # 计算均值
        self.data.loc[self.data['DebtRatio'] > 1, 'DebtRatio'] = ruul_mean  # 均值替代

        # 处理逾期特征异常值
        self.data.drop(self.data[self.data['30-59D'] > 80].index, inplace=True)  # 根据索引删除样本

        # 处理年龄特征异常值
        self.data.drop(self.data[self.data['Age'] == 0].index, inplace=True)  # 根据索引删除样本
        self.data.drop(self.data[self.data['Age'] > 96].index, inplace=True)

    def train(self):
        warnings.filterwarnings('ignore')  # 忽略弹出的warnings

        data = pd.read_csv('datasets/cs-training.csv')
        data = data.iloc[:, 1:]  # 舍弃Unnamed: 0列
        data.columns = ['Label', 'RevolvingRatio', 'Age', '30-59D', 'DebtRatio', 'MonthlyIncome',
                        'OpenL', '90D', 'RealEstate', '60-89D', 'Dependents']  # 列重命名
        # print(data.head(10))    # 观察整理后数据集
        # MonthlyIncome特征处理

        # 用MonthlyIncome特征值非空的样本构建训练集，MonthlyIncome特征值缺失的样本构建测试集
        rfDf = data.iloc[:, [5, 1, 2, 3, 4, 6, 7, 8, 9]]  # 原始数据集中的无缺失数值特征
        rfDf_train = rfDf.loc[rfDf['MonthlyIncome'].notnull()]
        rfDf_test = rfDf.loc[rfDf['MonthlyIncome'].isnull()]

        # 划分训练数据和标签（label）
        X = rfDf_train.iloc[:, 1:]
        y = rfDf_train.iloc[:, 0]
        # 训练过程
        rf = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)  # 这里重在理解过程，因此仅简单选取部分参数
        rf.fit(X, y)
        # 预测过程
        pred = rf.predict(rfDf_test.iloc[:, 1:]).round(0)  # 预测值四舍五入并保留一位小数点
        data.loc[(data['MonthlyIncome'].isnull()), 'MonthlyIncome'] = pred  # 填补缺失值

        print("此时的MonthlyIncome特征统计指标:\n")
        print(rfDf['MonthlyIncome'].describe())
        # Dependents特征处理
        data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)  # 这里采用众数填充
        print("此时Dependents特征统计指标:\n")
        print(data['Dependents'].describe())

        # 处理百分比类异常值
        # RevolvingRatio特征
        ruulDf = data[data['RevolvingRatio'] <= 1]  # 去掉高于1的部分
        ruul_mean = ruulDf['RevolvingRatio'].mean()  # 计算均值
        data.loc[data['RevolvingRatio'] > 1, 'RevolvingRatio'] = ruul_mean  # 均值替代

        # DebtRatio特征
        ruulDf = data[data['DebtRatio'] <= 1]  # 去掉高于1的部分
        ruul_mean = ruulDf['DebtRatio'].mean()  # 计算均值
        data.loc[data['DebtRatio'] > 1, 'DebtRatio'] = ruul_mean  # 均值替代

        # 处理逾期特征异常值
        data.drop(data[data['30-59D'] > 80].index, inplace=True)  # 根据索引删除样本
        print("剩下的样本数为：", data.shape[0])

        # 处理年龄特征异常值
        data.drop(data[data['Age'] == 0].index, inplace=True)  # 根据索引删除样本
        data.drop(data[data['Age'] > 96].index, inplace=True)
        print("剩下的样本数为：", data.shape[0])

        # 构建新特征
        # IncAvg:家庭中每个人分摊的平均月收入
        data['IncAvg'] = data['MonthlyIncome'] / (data['Dependents'] + 1)
        # MonthlyDept:每月的债务
        data['MonthlyDept'] = data['MonthlyIncome'] * data['DebtRatio']
        # DeptAvg:家庭中平均每个人分摊每月应还债务
        data['DeptAvg'] = data['MonthlyDept'] / (data['Dependents'] + 1)

        data[['IncAvg', 'MonthlyDept', 'DeptAvg']].head(10)  # 查看新特征

        rrDf, cut_rr, woe_rr, iv_rr = optimal_bins(data.Label, data.RevolvingRatio, n=10)
        print(rrDf)
        print(cut_rr)

        # MonthlyIncome特征
        miDf, cut_mi, woe_mi, iv_mi = optimal_bins(data.Label, data.MonthlyIncome, n=10)
        print("MonthlyIncome特征分箱情况：", cut_mi)
        # Age特征
        ageDf, cut_age, woe_age, iv_age = optimal_bins(data.Label, data.Age, n=10)
        print("Age特征分箱情况：", cut_age)
        # DebtRatio特征
        drDf, cut_dr, woe_dr, iv_dr = optimal_bins(data.Label, data.DebtRatio, 10)
        print("DebtRatio特征分箱情况：", cut_dr)

        # 自定义分箱区间如下
        # 原始特征
        ninf = float('-inf')
        pinf = float('inf')
        cut_thirty = [ninf, 0, 1, 3, 5, pinf]  # 30-59D特征
        cut_open = [ninf, 1, 2, 3, 5, pinf]  # OpenL特征
        cut_ninety = [ninf, 0, 1, 3, 5, pinf]  # 90D特征
        cut_re = [ninf, 0, 1, 2, 3, pinf]  # RealEstate特征
        cut_sixty = [ninf, 0, 1, 3, pinf]  # 60-89D特征
        cut_dpt = [ninf, 0, 1, 2, 3, 5, pinf]  # Dependents特征
        # 新特征
        cut_new2 = [ninf, 414, 1209, 2518, pinf]  # 新特征MonthlyDept自定义分箱

        # 计算统计值、woe和iv
        thirtyDf, woe_thirty, iv_thirty = custom_bins(data.Label, data['30-59D'], cut_thirty)  # 30-59D特征
        openDf, woe_open, iv_open = custom_bins(data.Label, data.OpenL, cut_open)  # OpenL特征
        ninetyDf, woe_ninety, iv_ninety = custom_bins(data.Label, data['90D'], cut_ninety)  # 90D特征
        reDf, woe_re, iv_re = custom_bins(data.Label, data.RealEstate, cut_re)  # RealEstate特征
        sixtyDf, woe_sixty, iv_sixty = custom_bins(data.Label, data['60-89D'], cut_sixty)  # 60-89D特征
        dptDf, woe_dpt, iv_dpt = custom_bins(data.Label, data.Dependents, cut_dpt)  # Dependents特征
        newDf2, woe_new2, iv_new2 = custom_bins(data.Label, data.MonthlyDept, cut_new2)  # 新特征MonthlyDept

        # WOE编码
        data['90D'] = pd.cut(data['90D'], bins=cut_ninety, labels=woe_ninety)  # 90D特征
        data['RevolvingRatio'] = pd.cut(data['RevolvingRatio'], bins=cut_rr, labels=woe_rr)  # RevolvingRatio特征
        data['30-59D'] = pd.cut(data['30-59D'], bins=cut_thirty, labels=woe_thirty)  # 30-59D特征
        data['60-89D'] = pd.cut(data['60-89D'], bins=cut_sixty, labels=woe_sixty)  # 60-89D特征
        data['Age'] = pd.cut(data['Age'], bins=cut_age, labels=woe_age)  # Age特征

        # 特征选择
        data = data[['Label', '90D', 'RevolvingRatio', '30-59D', '60-89D', 'Age']]
        print(data.head(10))  # 此时的数据集

        X = data.iloc[:, 1:]  # 特征
        y = data.iloc[:, 0]  # 目标变量

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # 训练集：测试集 = 7:3

        '''
        LogisticRegression一些重要参数的默认值：

        penalty：正则化类型，默认值'l2'，当solver='liblinear'时还可以选择'l1'
        solver：最优化方法，默认值'liblinear'，还可以选择'newton-cg', 'lbfgs', 'sag', 'saga'
        tol：迭代终止的阈值，默认值为1e-4
        max_iter：最大迭代次数，默认值100    
        （...等其他参数）
        '''

        model1 = LogisticRegression()  # 首先全部采用默认值进行训练
        clf1 = model1.fit(X_train, y_train)  # 模型训练

        # 记录训练后的模型
        self.clf1 = clf1
