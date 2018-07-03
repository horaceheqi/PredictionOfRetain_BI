# 验证正负样本的比例
from __future__ import division


def checkData(fileName):
    with open(fileName, 'r', encoding='utf-8', errors='replace')as f:
        positive = 0
        negative = 0
        for line in f.readlines():
            datas = line.strip().split(',')
            if datas[1] == '1':
                positive += 1
            else:
                negative += 1
        print("正样本数量%s,负样本数量%s, 正负样本比%s" % (positive, negative, positive/negative))


if __name__ == '__main__':
    checkData(r'roleflag_2015_0301.txt')