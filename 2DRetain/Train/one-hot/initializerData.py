# 对数据进行整理清洗整合成训练样本
import csv
import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA


class initialize(object):
    def __init__(self):
        self.users = dict()
        self.motion = set()
        self.file = list()
        self.user_motion = dict()

    def getUser(self, filename):
        with open(filename, 'r', encoding='utf-8', errors='replace') as user_read:
            for lines in user_read:
                data_line = lines.strip().split(',')
                self.users[data_line[0]] = data_line[1]
                # print(self.users)

    def getMostion(self, file1, file2, file3, file4):
        temp_dict = dict()
        self.file.append(file1)
        self.file.append(file2)
        self.file.append(file3)
        self.file.append(file4)

        for file in self.file:
            print(file)
            with open(file, 'r', encoding='utf-8', errors='replace') as motion_read:
                for lines in motion_read:
                    temp = set()
                    data_line = lines.strip().split(',')
                    motion = data_line[1] + data_line[3]
                    self.motion.add(motion)
                    if data_line[0] in temp_dict:
                        temp_dict[data_line[0]].add(motion)
                    else:
                        temp_dict[data_line[0]] = temp
        self.user_motion = temp_dict
        # print(len(self.user_motion))
        # print(self.user_motion)
        # print(len(self.motion))

    def createData(self):
        # 标记动作为0和1的用户
        temp_action_0 = list()
        temp_action_1 = list()
        temp_index = self.users.keys()
        print(len(temp_index))
        temp_columns = list(self.motion)
        temp_columns.append('Label')
        print(len(temp_columns))
        df = pd.DataFrame(np.zeros((len(temp_index), len(temp_columns))), index=temp_index, columns=temp_columns)
        for key, values in self.user_motion.items():
            if len(values) == 0 and key in temp_index:
                temp_action_0.append(key)
                # df.drop(key, inplace=True)
            if len(values) == 1 and key in temp_index:
                temp_action_1.append(key)
                # df.drop(key, inplace=True)
            if key in temp_index:
                print(key)
                for motion in values:
                    df[motion][key] = 1.0

        for key, value in self.users.items():
            if key in temp_index:
                df['Label'][key] = value
        # print(df['Label'])
        print(len(temp_action_0), len(temp_action_1))
        # df.index = df.index.droplevel()
        df.to_csv('../feature_data/initData.csv', index=temp_index, index_label='role_id')

    def getMergeFeature(self):
        '''
        混合全部已有特征
        :return: 返回混合全部特征之后的训练数据
        '''
        acquire_feature = pd.read_csv(r'E:/Coding/PredictionOfRetain/2DRetain/Train/feature_data/feature_acquire.csv')
        getitem_feature = pd.read_csv(r'E:/Coding/PredictionOfRetain/2DRetain/Train/feature_data/feature_getitem.csv')
        moneycost_feature = pd.read_csv(r'E:/Coding/PredictionOfRetain/2DRetain/Train/feature_data/feature_moneycost.csv')
        print(moneycost_feature.shape)
        removeitem_feature = pd.read_csv(r'E:/Coding/PredictionOfRetain/2DRetain/Train/feature_data/feature_removeitem.csv')
        print(removeitem_feature.shape)
        initData_feature = pd.read_csv(r'E:/Coding/PredictionOfRetain/2DRetain/Train/feature_data/initData.csv')
        print(initData_feature.shape)

        # dataset = initData_feature
        dataset = pd.merge(initData_feature, moneycost_feature, on=['role_id'], how='left')
        dataset = pd.merge(dataset, acquire_feature, on=['role_id'], how='left')
        dataset = pd.merge(dataset, getitem_feature, on=['role_id'], how='left')
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
        dataset.to_csv('../feature_data/MergeData.csv', index=None)

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
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'eta': 0.001,
            'seed': 1024,
            'nthread': 4,
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
        prediction = model.predict(test_x)
        print(metrics.classification_report(test_y, prediction))
        print(metrics.confusion_matrix(test_y, prediction))
        # print(metrics.auc(test_y, prediction))
        print(type(prediction))
        print(roc_auc_score(test_y, prediction))

    def trainData_SVM(self, filename):
        train_data = pd.read_csv(filename)
        print(train_data.shape)
        target = train_data['Label']
        # target = np.array(train_data['Label'])
        del train_data['Label']
        train = train_data
        train.ix[:, ~(train == 0).all()]
        # train = np.array(train_data)
        train_x, test_x, train_y, test_y = train_test_split(train, target, test_size=0.2, random_state=0)
        # print(test_y, len(test_y))
        train_x, test_x, train_y, test_y = np.array(train_x), np.array(test_x), np.array(train_y), np.array(test_y)
        print(type(test_x), test_x)
        # model = svm.LinearSVC()
        clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
        clf.fit(train_x, train_y)
        predition = clf.predict(test_x)
        # y_predprob = clf.predict_proba(test_x)
        print(predition)
        print(metrics.classification_report(test_y, predition))
        print(metrics.confusion_matrix(test_y, predition))
        print("ACC:%s" % metrics.accuracy_score(test_x, predition))
        # print("AUC:%s" % roc_auc_score(test_y, y_predprob))

    def trainData_RF(self, filename):
        train_data = pd.read_csv(filename)
        print(train_data.shape)
        data_columns = [x for x in train_data.columns if x not in ['Label', 'role_id']]
        train = train_data[data_columns]
        target = train_data['Label']
        train_x, text_x, train_y, test_y = train_test_split(train, target, test_size=0.2, random_state=0)
        clf = RandomForestClassifier(oob_score=True, random_state=10)
        clf.fit(train_x, train_y)
        predicted = clf.predict(text_x)
        y_predprob = clf.predict_proba(text_x)[:, 1]
        print(metrics.classification_report(test_y, predicted))
        print(metrics.confusion_matrix(test_y, predicted))
        print("ACC:%s" % metrics.accuracy_score(test_y, predicted))
        print("AUC:%s" % roc_auc_score(test_y, y_predprob))

    def trainData_GBDT(self, filename):
        train_data = pd.read_csv(filename)
        print(train_data.shape)
        data_column = [x for x in train_data.columns if x not in ['Label']]
        train = train_data[data_column]
        target = train_data['Label']
        train_x, test_x, train_y, test_y = train_test_split(train, target, test_size=0.2, random_state=0)
        clf = GradientBoostingClassifier(random_state=10)
        clf.fit(train_x, train_y)
        predicted = clf.predict(test_x)
        y_predprob = clf.predict_proba(test_x)[:, 1]
        print(metrics.classification_report(test_y, predicted))
        print(metrics.confusion_matrix(test_y, predicted))
        print("ACC:%s" % metrics.accuracy_score(test_y, predicted))
        print("AUC:%s" % roc_auc_score(test_y, y_predprob))

    def data_PCA(self, filename):
        train_data = pd.read_csv(filename)
        print(train_data.shape)
        target = train_data['Label']
        data_columns = [x for x in train_data.columns if x not in ['Label']]
        train = train_data[data_columns]
        train_x, test_x, train_y, test_y = train_test_split(train, target, test_size=0.2, random_state=0)
        print(train_x.shape, test_x.shape)
        train_x, test_x, train_y, test_y = np.array(train_x), np.array(test_x), np.array(train_y), np.array(test_y)
        clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
        pca = PCA(n_components=190)
        train_pca = pca.fit_transform(train_x)
        test_pca = pca.fit_transform(test_x)
        # print(train_pca, test_pca)
        clf.fit(train_pca, train_y)
        predicted = clf.predict(test_pca)
        print(predicted)
        print(metrics.classification_report(test_y, predicted))
        print(metrics.confusion_matrix(test_y, predicted))

    def txtTcsv(self, inPutFile, outPutFile):
        with open(outPutFile, 'w') as savecsv:
            spamwriter = csv.writer(savecsv, dialect='excel')
            with open(inPutFile, 'r')as readTxt:
                for line in readTxt.readlines():
                    line_list = line.strip("").replace("\n", "").split(",")[:5]
                    print(line_list)
                    spamwriter.writerow(line_list)

    def getFeature_acquire(self):

        # 对于Acquire特征进行处理
        fileName = r'E:/Coding/PredictionOfRetain/2DRetain/Train/one-hot/acquire_0301.csv'
        original_data = pd.read_csv(fileName)
        original_data.columns = ['role_id', 'type_id', 'quantity_id', 'server_time']
        print(original_data.shape)
        feature = original_data

        # 对特征feature的处理
        # t = feature[['role_id']]
        # t['num_of_total_action'] = 1
        # t = t.groupby('role_id').agg('sum').reset_index()
        t1 = feature[['role_id', 'type_id']]
        t1['num_of_difType_action'] = 1
        t1 = t1.groupby(['role_id', 'type_id']).agg('sum').reset_index()
        print(t1.head(10))
        t2 = feature[['role_id', 'type_id', 'quantity_id']]
        t2 = t2.groupby(['role_id', 'type_id'])['quantity_id'].agg('sum').reset_index()
        t2.rename(columns={'quantity_id': 'sum_quantity_id'}, inplace=True)
        t3 = feature[['role_id', 'type_id', 'quantity_id']]
        t3 = t3.groupby(['role_id', 'type_id'])['quantity_id'].agg('max').reset_index()
        t3.rename(columns={'quantity_id': 'max_quantity_id'}, inplace=True)
        t3 = pd.merge(t3, t2, on=['role_id', 'type_id'], how='left')
        t4 = feature[['role_id', 'type_id', 'quantity_id']]
        t4 = t4.groupby(['role_id', 'type_id'])['quantity_id'].agg('min').reset_index()
        t4.rename(columns={'quantity_id': 'min_quantity_id'}, inplace=True)
        t4 = pd.merge(t4, t3, on=['role_id', 'type_id'], how='left')
        t5 = feature[['role_id', 'type_id', 'quantity_id']]
        t5 = t5.groupby(['role_id', 'type_id'])['quantity_id'].agg('mean').reset_index()
        t5.rename(columns={'quantity_id': 'mean_quantity_id'}, inplace=True)
        t5 = pd.merge(t5, t4, on=['role_id', 'type_id'], how='left')
        t5 = pd.merge(t5, t1, on=['role_id', 'type_id'], how='left')
        # one-hot 处理type_id 特征
        # type_mapping = {label: index for index, label in enumerate(set(feature['type_id']))}
        # t5['type_id'] = t5['type_id'].map(type_mapping)
        # t5 = pd.get_dummies(t5)
        print("Acquire feature is complete!")
        print(t5.head(5))
        t5.to_csv('../feature_data/feature_acquire.csv', index=None)

    def getFeature_getitem(self):

        # 对于 getitem 特征进行处理
        fileName = r'E:/Coding/PredictionOfRetain/2DRetain/Train/one-hot/getitem_0301.csv'
        # roleid-角色ID、itemtypeid-物品类型ID、itemid-物品ID、quantity-总量
        original_data = pd.read_csv(fileName)
        original_data.columns = ['role_id', 'itemtype_id', 'item_id', 'quantity']
        feature = original_data
        print(feature.shape)
        # 每个用户购买物品总量
        t1 = feature[['role_id']]
        t1['same_type_item_num'] = 1
        t1 = t1.groupby(['role_id']).agg('sum').reset_index()
        # print(t1.head(10))
        # 每个用户购买相同类型(itemtype_id)的物品总数量
        t2 = feature[['role_id', 'itemtype_id', 'quantity']]
        t2 = t2.groupby(['role_id', 'itemtype_id'])['quantity'].agg('sum').reset_index()
        t2.rename(columns={'quantity': 'sum_same_item_type_num'}, inplace=True)
        t3 = feature[['role_id', 'itemtype_id', 'quantity']]
        t3 = t3.groupby(['role_id', 'itemtype_id'])['quantity'].agg('max').reset_index()
        t3.rename(columns={'quantity': 'max_same_item_type_num'}, inplace=True)
        t4 = feature[['role_id', 'itemtype_id', 'quantity']]
        t4 = t4.groupby(['role_id', 'itemtype_id'])['quantity'].agg('mean').reset_index()
        t4.rename(columns={'quantity': 'mean_same_item_type_num'}, inplace=True)

        merge_data = pd.merge(t1, t2, on=['role_id'], how='left')
        merge_data = pd.merge(merge_data, t3, on=['role_id', 'itemtype_id'], how='left')
        merge_data = pd.merge(merge_data, t4, on=['role_id', 'itemtype_id'], how='left')
        # print(merge_data.head(10))
        merge_data.to_csv('../feature_data/feature_getitem.csv', index=None)

    def getFeature_moneycost(self):

        # 对于 moneycost 特征进行处理
        fileName = r'E:/Coding/PredictionOfRetain/2DRetain/Train/one-hot/moneycost_0301.csv'
        original_data = pd.read_csv(fileName)
        original_data.columns = ['role_id', 'type_id', 'quantity']
        feature = original_data
        # print(feature.head(10))
        t1 = feature[['role_id']]
        t1['total_cost_num'] = 1
        t1 = t1.groupby(['role_id']).agg('sum').reset_index()
        t2 = feature[['role_id', 'quantity']]
        t2 = t2.groupby(['role_id'])['quantity'].agg('max').reset_index()
        t2.rename(columns={'quantity': 'max_cost'}, inplace=True)
        t3 = feature[['role_id', 'quantity']]
        t3 = t3.groupby(['role_id'])['quantity'].agg('min').reset_index()
        t3.rename(columns={'quantity': 'min_cost'}, inplace=True)
        t4 = feature[['role_id', 'quantity']]
        t4 = t4.groupby(['role_id'])['quantity'].agg('mean').reset_index()
        t4.rename(columns={'quantity': 'mean_cost'}, inplace=True)

        merge_data = pd.merge(t1, t2, on=['role_id'], how='left')
        merge_data = pd.merge(merge_data, t3, on=['role_id'], how='left')
        merge_data = pd.merge(merge_data, t4, on=['role_id'], how='left')
        # print(merge_data.head(10))
        merge_data.to_csv('../feature_data/feature_moneycost.csv', index=None)

    def getFeature_removeitem(self):

        # 对于 removeitem 特征进行处理
        fileName = r'E:/Coding/PredictionOfRetain/2DRetain/Train/one-hot/removeitem_0301.csv'
        original_data = pd.read_csv(fileName)
        original_data.columns = ['role_id', 'item_type', 'item_id', 'quantity']
        feature = original_data
        t1 = feature[['role_id']]
        t1['total_remove_num'] = 1
        t1 = t1.groupby(['role_id']).agg('sum').reset_index()
        t2 = feature[['role_id', 'quantity']]
        t2 = t2.groupby(['role_id'])['quantity'].agg('max').reset_index()
        t2.rename(columns={'quantity': 'once_max_remove'}, inplace=True)
        t3 = feature[['role_id', 'quantity']]
        t3 = t3.groupby(['role_id'])['quantity'].agg('min').reset_index()
        t3.rename(columns={'quantity': 'once_min_remove'}, inplace=True)
        t4 = feature[['role_id', 'quantity']]
        t4 = t4.groupby(['role_id'])['quantity'].agg('mean').reset_index()
        t4.rename(columns={'quantity': 'once_mean_remove'}, inplace=True)

        merge_data = pd.merge(t1, t2, on=['role_id'], how='left')
        merge_data = pd.merge(merge_data, t3, on=['role_id'], how='left')
        merge_data = pd.merge(merge_data, t4, on=['role_id'], how='left')
        # print(merge_data .head(10))
        merge_data.to_csv('../feature_data/feature_removeitem.csv', index=None)


if __name__ == '__main__':
    start = time.clock()
    files = r'../../Datas/login_flag.txt'
    huobiUse = r'E:/数据/天龙3D/货币消耗日志/moneycost_2015_03_01.txt'
    huobiGet = r'E:/数据/天龙3D/经验或货币获得日志/acquire_2015_03_01.txt'
    wupingUse = r'E:/数据/天龙3D/物品消耗日志/removeitem_2015_03_01.txt'
    wupingGet = r'E:/数据/天龙3D/物品获得日志/getitem_2015_03_01.txt'
    # 经验获取
    experience = r'E:/Coding/PredictionOfRetain/2DRetain/Datas/removeitem_1.txt'

    initData = r'../feature_data/MergeData.csv'
    # initData = r'../feature_data/initData.csv'
    demo = initialize()

    # demo.txtTcsv(experience, 'removeitem_0301.csv')

    # 获取经验acquire特征，包括总量、最大最小值、平均值
    # demo.getFeature_acquire()
    # demo.getFeature_getitem()
    # demo.getFeature_moneycost()
    # demo.getFeature_removeitem()

    # demo.getUser(files)
    # demo.getMostion(huobiUse, huobiGet, wupingUse, wupingGet)
    # demo.createData()

    # 混合全部特征生成训练数据
    # demo.getMergeFeature()

    # demo.trainData_XGBoost(initData)
    demo.trainData_LGR(initData)
    # demo.trainData_SVM(initData)
    # demo.trainData_RF(initData)
    # demo.trainData_GBDT(initData)
    # demo.data_PCA(initData)
    end = time.clock()
    print("消耗时间：%f s" % (end - start))
    # print(demo.users)
