from sklearn import datasets
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn import metrics

if __name__ == '__main__':
    iris = datasets.load_iris()
    data = iris.data[:100]
    print(data.shape)
    label = iris.target[:100]
    print(label)
    train_x, test_x, train_y, test_y = train_test_split(data, label, random_state=0)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'lambda': 10,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 2,
        'eta': 0.025,
        'seed': 0,
        'nthread': 8,
        'silent': 1
    }
    wathlist = [(dtrain, 'train')]

    bst = xgb.train(params, dtrain, num_boost_round=10, evals=wathlist)
    ypred = bst.predict(dtest)

    y_pred = (ypred >= 0.5)*1
    print('AUC: %.4f' % metrics.roc_auc_score(test_y, ypred))
    print('ACC: %.4f' % metrics.accuracy_score(test_y, y_pred))
    print('Recall: %.4f' % metrics.recall_score(test_y, y_pred))
    print('F1-score: %.4f' % metrics.f1_score(test_y, y_pred))
    print('Precesion: %.4f' % metrics.precision_score(test_y, y_pred))
    metrics.confusion_matrix(test_y, y_pred)

    ypred = bst.predict(dtest)
    print(ypred)
    ypred_leaf = bst.predict(dtest, pred_leaf=True)
    print(ypred_leaf)
    xgb.to_graphviz(bst, num_trees=0)

