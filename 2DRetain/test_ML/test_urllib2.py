import pandas as pd
import numpy as np
from datetime import date

if __name__ == '__main__':
    # filePath = r'E:/Coding/机器学习/O2O-Coupon-Usage-Forecast-master/data/ccf_offline_stage1_train.csv'
    # filePath2 = r'E:/Coding/机器学习/O2O-Coupon-Usage-Forecast-master/data/ccf_offline_stage1_test_revised.csv'
    # off_train = pd.read_csv(filePath, header=None)
    # off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
    # off_test = pd.read_csv(filePath2, header=None)
    # off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']
    # # dataset3 = off_test
    # dataset2 = off_train[(off_train.date_received >= '20160515') & (off_train.date_received <= '20160615')]
    # feature2 = off_train[(off_train.date >= '20160201') & (off_train.date <= '20160514') | ((off_train.date == 'null') & (off_train.date_received >= '20160201') & (off_train.date_received <= '20160514'))]

    def get_label(s):
        s = s.split(':')
        if s[0] == 'null':
            return 0
        elif (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days <= 15:
            return 1
        else:
            return -1

    coupon2 = pd.read_csv('E:/data/coupon2_feature.csv')
    print(coupon2.shape)
    merchant2 = pd.read_csv('E:/data/merchant2_feature.csv')
    print(merchant2.shape)
    user2 = pd.read_csv('E:/data/user2_feature.csv')
    print(user2.shape)
    user_merchant2 = pd.read_csv(r'E:/data/user_merchant2.csv')
    print(user_merchant2.shape)
    other_feature2 = pd.read_csv(r'E:/data/other_feature2.csv')
    print(other_feature2.shape)
    dataset2 = pd.merge(coupon2, merchant2, on='merchant_id', how='left')
    dataset2 = pd.merge(dataset2, user2, on='user_id', how='left')
    dataset2 = pd.merge(dataset2, user_merchant2, on=['user_id', 'merchant_id'], how='left')
    dataset2 = pd.merge(dataset2, other_feature2, on=['user_id', 'coupon_id', 'date_received'], how='left')
    print(dataset2.shape)
    dataset2.drop_duplicates(inplace=True)

    dataset2.user_merchant_buy_total = dataset2.user_merchant_buy_total.replace(np.nan, 0)
    dataset2.user_merchant_any = dataset2.user_merchant_any.replace(np.nan, 0)
    dataset2.user_merchant_received = dataset2.user_merchant_received.replace(np.nan, 0)
    dataset2['is_weekend'] = dataset2.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
    weekday_dummies = pd.get_dummies(dataset2.day_of_week)
    weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
    dataset2 = pd.concat([dataset2, weekday_dummies], axis=1)
    dataset2['label'] = dataset2.date.astype('str') + ':' + dataset2.date_received.astype('str')
    dataset2.label = dataset2.label.apply(get_label)
    dataset2.drop(['merchant_id', 'day_of_week', 'date', 'date_received', 'coupon_id', 'coupon_count'], axis=1, inplace=True)
    dataset2 = dataset2.replace('null', np.nan)
    print(dataset2.head(10))
    # dataset2.to_csv('data/dataset2.csv', index=None)


    # # for dataset2
    # all_user_merchant = feature2[['user_id', 'merchant_id']]
    # all_user_merchant.drop_duplicates(inplace=True)
    #
    # t = feature2[['user_id', 'merchant_id', 'date']]
    # t = t[t.date != 'null'][['user_id', 'merchant_id']]
    # t['user_merchant_buy_total'] = 1
    # t = t.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    # t.drop_duplicates(inplace=True)
    #
    # t1 = feature2[['user_id', 'merchant_id', 'coupon_id']]
    # t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
    # t1['user_merchant_received'] = 1
    # t1 = t1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    # t1.drop_duplicates(inplace=True)
    #
    # t2 = feature2[['user_id', 'merchant_id', 'date', 'date_received']]
    # t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
    # t2['user_merchant_buy_use_coupon'] = 1
    # t2 = t2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    # t2.drop_duplicates(inplace=True)
    #
    # t3 = feature2[['user_id', 'merchant_id']]
    # t3['user_merchant_any'] = 1
    # t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    # t3.drop_duplicates(inplace=True)
    #
    # t4 = feature2[['user_id', 'merchant_id', 'date', 'coupon_id']]
    # t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][['user_id', 'merchant_id']]
    # t4['user_merchant_buy_common'] = 1
    # t4 = t4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    # t4.drop_duplicates(inplace=True)
    #
    # user_merchant2 = pd.merge(all_user_merchant, t, on=['user_id', 'merchant_id'], how='left')
    # user_merchant2 = pd.merge(user_merchant2, t1, on=['user_id', 'merchant_id'], how='left')
    # user_merchant2 = pd.merge(user_merchant2, t2, on=['user_id', 'merchant_id'], how='left')
    # user_merchant2 = pd.merge(user_merchant2, t3, on=['user_id', 'merchant_id'], how='left')
    # user_merchant2 = pd.merge(user_merchant2, t4, on=['user_id', 'merchant_id'], how='left')
    # user_merchant2.user_merchant_buy_use_coupon = user_merchant2.user_merchant_buy_use_coupon.replace(np.nan, 0)
    # user_merchant2.user_merchant_buy_common = user_merchant2.user_merchant_buy_common.replace(np.nan, 0)
    # user_merchant2['user_merchant_coupon_transfer_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype(
    #     'float') / user_merchant2.user_merchant_received.astype('float')
    # user_merchant2['user_merchant_coupon_buy_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype(
    #     'float') / user_merchant2.user_merchant_buy_total.astype('float')
    # user_merchant2['user_merchant_rate'] = user_merchant2.user_merchant_buy_total.astype(
    #     'float') / user_merchant2.user_merchant_any.astype('float')
    # user_merchant2['user_merchant_common_buy_rate'] = user_merchant2.user_merchant_buy_common.astype(
    #     'float') / user_merchant2.user_merchant_buy_total.astype('float')
    # user_merchant2.to_csv('E:/Coding/机器学习/O2O-Coupon-Usage-Forecast-master/code/wepon/season one/data/user_merchant2.csv', index=None)
    # print(user_merchant2.head(10))


    # # for dataset2
    # merchant2 = feature2[['merchant_id', 'coupon_id', 'distance', 'date_received', 'date']]
    #
    # t = merchant2[['merchant_id']]
    # t.drop_duplicates(inplace=True)
    #
    # t1 = merchant2[merchant2.date != 'null'][['merchant_id']]
    # t1['total_sales'] = 1
    # t1 = t1.groupby('merchant_id').agg('sum').reset_index()
    #
    # t2 = merchant2[(merchant2.date != 'null') & (merchant2.coupon_id != 'null')][['merchant_id']]
    # t2['sales_use_coupon'] = 1
    # t2 = t2.groupby('merchant_id').agg('sum').reset_index()
    #
    # t3 = merchant2[merchant2.coupon_id != 'null'][['merchant_id']]
    # t3['total_coupon'] = 1
    # t3 = t3.groupby('merchant_id').agg('sum').reset_index()
    #
    # t4 = merchant2[(merchant2.date != 'null') & (merchant2.coupon_id != 'null')][['merchant_id', 'distance']]
    # t4.replace('null', -1, inplace=True)
    # t4.distance = t4.distance.astype('int')
    # t4.replace(-1, np.nan, inplace=True)
    # t5 = t4.groupby('merchant_id').agg('min').reset_index()
    # t5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)
    #
    # t6 = t4.groupby('merchant_id').agg('max').reset_index()
    # t6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)
    #
    # t7 = t4.groupby('merchant_id').agg('mean').reset_index()
    # t7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)
    #
    # t8 = t4.groupby('merchant_id').agg('median').reset_index()
    # t8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)
    #
    # merchant2_feature = pd.merge(t, t1, on='merchant_id', how='left')
    # merchant2_feature = pd.merge(merchant2_feature, t2, on='merchant_id', how='left')
    # merchant2_feature = pd.merge(merchant2_feature, t3, on='merchant_id', how='left')
    # merchant2_feature = pd.merge(merchant2_feature, t5, on='merchant_id', how='left')
    # merchant2_feature = pd.merge(merchant2_feature, t6, on='merchant_id', how='left')
    # merchant2_feature = pd.merge(merchant2_feature, t7, on='merchant_id', how='left')
    # merchant2_feature = pd.merge(merchant2_feature, t8, on='merchant_id', how='left')
    # merchant2_feature.sales_use_coupon = merchant2_feature.sales_use_coupon.replace(np.nan, 0)  # fillna with 0
    # merchant2_feature['merchant_coupon_transfer_rate'] = merchant2_feature.sales_use_coupon.astype('float') / merchant2_feature.total_coupon
    # merchant2_feature['coupon_rate'] = merchant2_feature.sales_use_coupon.astype('float') / merchant2_feature.total_sales
    # merchant2_feature.total_coupon = merchant2_feature.total_coupon.replace(np.nan, 0)  # fillna with 0
    # merchant2_feature.to_csv('E:/Coding/机器学习/O2O-Coupon-Usage-Forecast-master/code/wepon/season one/data/merchant2_feature.csv', index=None)


    # # dataset2
    # # print(dataset2.head(5))
    # dataset2['day_of_week'] = dataset2.date_received.astype('str').apply(lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday() + 1)
    # dataset2['day_of_month'] = dataset2.date_received.astype('str').apply(lambda x: int(x[6:8]))
    # dataset2['days_distance'] = dataset2.date_received.astype('str').apply(lambda x: (date(int(x[0:4]), int(x[4:6]), int(x[6:8])) - date(2016, 5, 14)).days)
    # # print(dataset2.head(5))
    # # dataset2['discount_man'] = dataset2.discount_rate.apply(get_discount_man)
    # # dataset2['discount_jian'] = dataset2.discount_rate.apply(get_discount_jian)
    # # dataset2['is_man_jian'] = dataset2.discount_rate.apply(is_man_jian)
    # # dataset2['discount_rate'] = dataset2.discount_rate.apply(calc_discount_rate)
    # d = dataset2[['coupon_id']]
    # d['coupon_count'] = 1
    # d = d.groupby('coupon_id').agg('sum').reset_index()
    # dataset2 = pd.merge(dataset2, d, on='coupon_id', how='left')
    # # print(dataset2.head(10))
    # dataset2.to_csv('E:/Coding/机器学习/O2O-Coupon-Usage-Forecast-master/code/wepon/season one/data/coupon2_feature.csv', index=None)

    # # for dataset2
    # def get_user_date_datereceived_gap(s):
    #     s = s.split(':')
    #     return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days
    # user2 = feature2[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]
    #
    # t = user2[['user_id']]
    # t.drop_duplicates(inplace=True)
    #
    # t1 = user2[user2.date != 'null'][['user_id', 'merchant_id']]
    # t1.drop_duplicates(inplace=True)
    # t1.merchant_id = 1
    # t1 = t1.groupby('user_id').agg('sum').reset_index()
    # t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)
    #
    # t2 = user2[(user2.date != 'null') & (user2.coupon_id != 'null')][['user_id', 'distance']]
    # t2.replace('null', -1, inplace=True)
    # t2.distance = t2.distance.astype('int')
    # t2.replace(-1, np.nan, inplace=True)
    # # print(t2.head(10))
    # t3 = t2.groupby('user_id').agg('min').reset_index()
    # t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)
    #
    # t4 = t2.groupby('user_id').agg('max').reset_index()
    # t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)
    #
    # t5 = t2.groupby('user_id').agg('mean').reset_index()
    # t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)
    #
    # t6 = t2.groupby('user_id').agg('median').reset_index()
    # t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)
    #
    # t7 = user2[(user2.date != 'null') & (user2.coupon_id != 'null')][['user_id']]
    # t7['buy_use_coupon'] = 1
    # t7 = t7.groupby('user_id').agg('sum').reset_index()
    #
    # t8 = user2[user2.date != 'null'][['user_id']]
    # t8['buy_total'] = 1
    # t8 = t8.groupby('user_id').agg('sum').reset_index()
    #
    # t9 = user2[user2.coupon_id != 'null'][['user_id']]
    # t9['coupon_received'] = 1
    # t9 = t9.groupby('user_id').agg('sum').reset_index()
    #
    # t10 = user2[(user2.date_received != 'null') & (user2.date != 'null')][['user_id', 'date_received', 'date']]
    # t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
    # t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
    # t10 = t10[['user_id', 'user_date_datereceived_gap']]
    #
    # t11 = t10.groupby('user_id').agg('mean').reset_index()
    # t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
    # t12 = t10.groupby('user_id').agg('min').reset_index()
    # t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
    # t13 = t10.groupby('user_id').agg('max').reset_index()
    # t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)
    #
    # user2_feature = pd.merge(t, t1, on='user_id', how='left')
    # user2_feature = pd.merge(user2_feature, t3, on='user_id', how='left')
    # user2_feature = pd.merge(user2_feature, t4, on='user_id', how='left')
    # user2_feature = pd.merge(user2_feature, t5, on='user_id', how='left')
    # user2_feature = pd.merge(user2_feature, t6, on='user_id', how='left')
    # user2_feature = pd.merge(user2_feature, t7, on='user_id', how='left')
    # user2_feature = pd.merge(user2_feature, t8, on='user_id', how='left')
    # user2_feature = pd.merge(user2_feature, t9, on='user_id', how='left')
    # user2_feature = pd.merge(user2_feature, t11, on='user_id', how='left')
    # user2_feature = pd.merge(user2_feature, t12, on='user_id', how='left')
    # user2_feature = pd.merge(user2_feature, t13, on='user_id', how='left')
    # user2_feature.count_merchant = user2_feature.count_merchant.replace(np.nan, 0)
    # user2_feature.buy_use_coupon = user2_feature.buy_use_coupon.replace(np.nan, 0)
    # user2_feature['buy_use_coupon_rate'] = user2_feature.buy_use_coupon.astype('float') / user2_feature.buy_total.astype('float')
    # user2_feature['user_coupon_transfer_rate'] = user2_feature.buy_use_coupon.astype('float') / user2_feature.coupon_received.astype('float')
    # user2_feature.buy_total = user2_feature.buy_total.replace(np.nan, 0)
    # user2_feature.coupon_received = user2_feature.coupon_received.replace(np.nan, 0)
    # # print(user2_feature.head(20))
    # user2_feature.to_csv('E:/Coding/机器学习/O2O-Coupon-Usage-Forecast-master/code/wepon/season one/data/user2_feature.csv', index=None)

    # # for dataset3
    # # 每个用户每个月所收到的全部打折券,不会有重复的user_id
    # t = dataset2[['user_id']]
    # t['this_month_user_receive_all_coupon_count'] = 1
    # t = t.groupby('user_id').agg('sum').reset_index()
    # # 每个用户每个月所收到的相同的打折券,会有重复的user_id
    # t1 = dataset2[['user_id', 'coupon_id']]
    # t1['this_month_user_receive_same_coupon_count'] = 1
    # t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()
    # # 每个用户收到打折券的时间,会有重复的user_id
    # t2 = dataset2[['user_id', 'coupon_id', 'date_received']]
    # t2.date_received = t2.date_received.astype('str')
    # t2 = t2.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    # t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
    # t2 = t2[t2.receive_number > 1]
    # t2['max_date_received'] = t2.date_received.apply(lambda s: max([int(d) for d in s.split(':')]))
    # t2['min_date_received'] = t2.date_received.apply(lambda s: min([int(d) for d in s.split(':')]))
    # t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]
    # # 会有重复的user_id
    # t4 = dataset2[['user_id', 'date_received']]
    # t4['this_day_user_receive_all_coupon_count'] = 1
    # t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()
    # # 会有重复的user_id
    # t5 = dataset2[['user_id', 'coupon_id', 'date_received']]
    # t5['this_day_user_receive_same_coupon_count'] = 1
    # t5 = t5.groupby(['user_id', 'coupon_id', 'date_received']).agg('sum').reset_index()
    #
    # other_feature3 = pd.merge(t1, t, on='user_id')
    # other_feature3 = pd.merge(other_feature3, t4, on=['user_id'])
    # other_feature3 = pd.merge(other_feature3, t5, on=['user_id', 'coupon_id', 'date_received'])
    # print(other_feature3.head(20))
    # other_feature3.to_csv('E:/Coding/机器学习/O2O-Coupon-Usage-Forecast-master/code/wepon/season one/data/other_feature2.csv', index=None)