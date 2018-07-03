# 对数据进行整理清洗整合成训练样本
import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


class initialize(object):
    def __init__(self):
        '''
        user:用户名和Label
        file:全部文件List集合
        user_motion:用户和其相对应的全部动作
        '''
        self.users = dict()
        self.file = list()
        self.motion = set()
        self.user_motion = dict()

    # 用key-value形式保存用户ID和标签
    def getUser(self, filename):
        with open(filename, 'r', encoding='utf-8', errors='replace') as user_read:
            for lines in user_read:
                data_line = lines.strip().split(',')
                self.users[data_line[0]] = data_line[1]
        # print(self.users)
        print("getUser: %s" % len(self.users))

    # 提取用户特征
    def getMostion(self, file1, file2, file3, file4):
        temp_dict = dict()
        self.file.append(file1)
        self.file.append(file2)
        self.file.append(file3)
        self.file.append(file4)
        print("fileNum: %s" % len(self.file))

        for file in self.file:
            print(file)
            with open(file, 'r', encoding='utf-8', errors='replace') as motion_read:
                for lines in motion_read:
                    temp = list()
                    data_line = lines.strip().split(',')
                    motion = data_line[1] + data_line[3]
                    self.motion.add(motion)
                    if data_line[0] in temp_dict:
                        temp_dict[data_line[0]].append(motion)
                    else:
                        temp_dict[data_line[0]] = temp
        self.user_motion = temp_dict
        print("dataNum: %s" % len(self.user_motion))
        # print(len(self.user_motion))
        # print(self.user_motion)
        # print(len(self.motion))

    def createData(self):
        # user_count 记录没有活动用户数
        user_count = 0
        temp_dict = dict()

        for key, value in self.user_motion.items():
            if len(value) != 0:
                temp = ','.join(value)
                temp_dict[key] = temp
            else:
                user_count += 1

        print("有%s用户没有用户行为!" % user_count)
        # print(temp_user)
        print("temp_dict:", len(temp_dict))
        csv_index = temp_dict.keys()
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(list(temp_dict.values()))
        words = vectorizer.get_feature_names()
        # TD-IDF会将全部字母转成小写,构建List转换成大写
        upper = [i.capitalize() for i in words]
        # print(upper)
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(X)
        data = tfidf.toarray()
        data_new = data.tolist()
        df = pd.DataFrame(data=data_new, columns=upper, index=csv_index)
        label = np.zeros(df.shape[0])
        label_count = 0

        for key, value in temp_dict.items():
            if key in self.users and self.users[key] == '1':
                label[label_count] = 1
            else:
                label[label_count] = 0
            label_count += 1
        df.insert(len(upper), 'Label', label)
        # print(df.head(3))
        df.to_csv('../feature_data/initData_TFIDF.csv', index=csv_index, index_label='role_id')

    def getMergeData(self):
        # acquire_feature = pd.read_csv(r'E:/Coding/PredictionOfRetain/2DRetain/Train/feature_data/feature_acquire.csv')
        # getitem_feature = pd.read_csv(r'E:/Coding/PredictionOfRetain/2DRetain/Train/feature_data/feature_getitem.csv')
        moneycost_feature = pd.read_csv(r'E:/Coding/PredictionOfRetain/2DRetain/Train/feature_data/feature_moneycost.csv')
        print(moneycost_feature.shape)
        removeitem_feature = pd.read_csv(r'E:/Coding/PredictionOfRetain/2DRetain/Train/feature_data/feature_removeitem.csv')
        print(removeitem_feature.shape)
        initData_feature = pd.read_csv(r'E:/Coding/PredictionOfRetain/2DRetain/Train/feature_data/initData_TFIDF.csv')
        print(initData_feature.shape)

        dataset = pd.merge(initData_feature, moneycost_feature, on=['role_id'], how='left')
        dataset = pd.merge(dataset, removeitem_feature, on=['role_id'], how='left')
        dataset.total_cost_num = dataset.total_cost_num.replace(np.nan, 0.0)
        dataset.max_cost = dataset.max_cost.replace(np.nan, 0.0)
        dataset.min_cost = dataset.min_cost.replace(np.nan, 0.0)
        dataset.mean_cost = dataset.mean_cost.replace(np.nan, 0.0)
        dataset.total_remove_num = dataset.total_remove_num.replace(np.nan, 0.0)
        dataset.once_max_remove = dataset.once_max_remove.replace(np.nan, 0.0)
        dataset.once_min_remove = dataset.once_min_remove.replace(np.nan, 0.0)
        dataset.once_mean_remove = dataset.once_mean_remove.replace(np.nan, 0.0)
        # print(dataset.head(5))
        dataset.to_csv('../feature_data/MergeData_TFIDF.csv', index=None)

    def trainData_XGBoost(self, filename):
        trainDatas = pd.read_csv(filename)
        print(trainDatas.shape)
        # print(trainDatas[['Label']])
        target = trainDatas[['Label']]
        print(target.shape)
        # del trainDatas['Label']
        # train = trainDatas
        train = trainDatas.drop(['role_id', 'Label'], axis=1)
        print(train.shape)
        train_x, test_x, train_y, test_y = train_test_split(train, target, test_size=0.2, random_state=0)
        print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x)

        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 5,
            'lambda': 70,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'eta': 0.001,
            'seed': 1024,
            'nthread': 8,
            'silent': 1
        }
        watchlist = [(dtrain, 'train')]
        model = xgb.train(params, dtrain, num_boost_round=200, evals=watchlist)
        ypred = model.predict(dtest)
        print(test_y)
        print(ypred)

        y_pred = (ypred >= 0.5) * 1
        print('AUC: %.4f' % metrics.roc_auc_score(test_y, ypred))
        print('ACC: %.4f' % metrics.accuracy_score(test_y, y_pred))
        print('Recall: %.4f' % metrics.recall_score(test_y, y_pred))
        print('F1-scall: %.4f' % metrics.f1_score(test_y, y_pred))
        print('Precesion: %.4f' % metrics.precision_score(test_y, y_pred))
        metrics.confusion_matrix(test_y, y_pred)

    def trainData_LGR(self, filename):
        train_data = pd.read_csv(filename)
        print(train_data.shape)
        target = train_data[['Label']]
        del train_data['Label']
        train = train_data
        train_x, test_x, train_y, test_y = train_test_split(train, target, test_size=0.2, random_state=0)
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        model = LogisticRegression()
        model.fit(train_x, train_y)
        predicted = model.predict(test_x)
        print(metrics.classification_report(test_y, predicted))
        print(metrics.confusion_matrix(test_y, predicted))
        print("AUC:%s" % roc_auc_score(test_y, predicted))

    def trainData_SVM(self, filename):
        train_data = pd.read_csv(filename)
        print(train_data.shape)
        data_column = [x for x in train_data.columns if x not in ['Label']]
        train = train_data[data_column]
        target = train_data['Label']
        train_x, test_x, train_y, test_y = train_test_split(train, target, test_size=0.2, random_state=0)
        train_x, test_x, train_y, test_y = np.array(train_x), np.array(test_x), np.array(train_y), np.array(test_y)
        clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
        clf.fit(train_x, train_y)
        predicted = clf.predict(test_x)
        # y_predprob = clf.predict_proba(test_x)
        print(metrics.classification_report(test_y, predicted))
        print(metrics.confusion_matrix(test_y, predicted))
        print("ACC:%s" % metrics.accuracy_score(test_y, predicted))
        # print("AUC:%s" % roc_auc_score(test_y, y_predprob))

    def trainData_RF(self, filename):
        trainData = pd.read_csv(filename)
        data_column = [x for x in trainData.columns if x not in ['Label']]
        train = trainData[data_column]
        target = trainData['Label']
        train_x, test_x, train_y, test_y = train_test_split(train, target, test_size=0.2, random_state=0)
        clf = RandomForestClassifier(oob_score=True, random_state=10)
        clf.fit(train_x, train_y)
        predicted = clf.predict(test_x)
        y_predprob = clf.predict_proba(test_x)[:, 1]
        print(metrics.classification_report(test_y, predicted))
        print(metrics.confusion_matrix(test_y, predicted))
        print("ACC:%s" % metrics.accuracy_score(test_y, predicted))
        print("AUC:%s" % roc_auc_score(test_y, y_predprob))

    def trainData_GBDT(self, filename):
        trainData = pd.read_csv(filename)
        data_column = [x for x in trainData.columns if x not in ['Label']]
        train = trainData[data_column]
        target = trainData['Label']
        train_x, test_x, train_y, test_y = train_test_split(train, target, test_size=0.2, random_state=0)
        clf = GradientBoostingClassifier(random_state=10)
        clf.fit(train_x, train_y)
        predicted = clf.predict(test_x)
        y_predprob = clf.predict_proba(test_x)[:, 1]
        print(metrics.classification_report(test_y, predicted))
        print(metrics.confusion_matrix(test_y, predicted))
        print("ACC:%s" % metrics.accuracy_score(test_y, predicted))
        print("AUC:%s" % roc_auc_score(test_y, y_predprob))


if __name__ == '__main__':
    start = time.clock()
    files = r'../../Datas/login_flag.txt'
    huobiUse = r'E:/数据/天龙3D/货币消耗日志/moneycost_2015_03_01.txt'
    huobiGet = r'E:/数据/天龙3D/经验或货币获得日志/acquire_2015_03_01.txt'
    wupingUse = r'E:/数据/天龙3D/物品消耗日志/removeitem_2015_03_01.txt'
    wupingGet = r'E:/数据/天龙3D/物品获得日志/getitem_2015_03_01.txt'
    initData = r'../feature_data/MergeData_TFIDF.csv'
    demo = initialize()
    # demo.getUser(files)
    # demo.getMostion(huobiUse, huobiGet, wupingUse, wupingGet)
    # demo.createData()
    demo.getMergeData()
    # demo.trainData_XGBoost(initData)
    # demo.trainData_LGR(initData)
    # demo.trainData_SVM(initData)
    # demo.trainData_RF(initData)
    # demo.trainData_GBDT(initData)
    end = time.clock()
    print("消耗时间：%f s" % (end - start))
    # print(demo.users)
