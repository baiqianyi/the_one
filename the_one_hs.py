import datetime
import requests
import logging

def weight_mean1(list, mean_num=20, minus=0.7, i_minus=0):
    mean = 0.0
    t_list = []
    if len(list) > mean_num:
        t_list = list[len(list) - mean_num - 1:]
    else:
        t_list = list
    if len(t_list) > 0:
        i_sum = 0
        for i in range(len(t_list)):
            i_sum = i_sum + i + 1 - minus
        for i in range(len(t_list)):
            mean = mean + float(i + 1 - minus) / float(i_sum) * t_list[i]
    else:
        return 0
    return mean


import tushare
import pandas


def get_zz_codes(url='http://www.csindex.com.cn/uploads/file/autofile/closeweight/000905closeweight.xls?t=1612595513'):
    pd = pandas.read_excel(url, dtype=str)
    pd.sort_values(by='权重(%)Weight(%)', axis=0, ascending=False, inplace=True)
    pd = pd.iloc[:110, ]['成分券代码Constituent Code']
    return set(pd)


zz_codes = get_zz_codes()
start = (datetime.datetime.now() - datetime.timedelta(days=450)).date().strftime('%Y-%m-%d')
hs300 = tushare.get_hist_data("hs300", start=start)
hs_p = list(hs300['p_change'])
hs_p.reverse()
date = list(hs300.index)
stocks = tushare.get_hs300s()
codes = []
for i in stocks.index:
    codes.append(stocks.loc[i, "code"])
codes.extend(zz_codes)
# def get_hists(symbols, start=None, end=None,ktype='D', retry_count=3,pause=0.001):
#     import gevent
#     import pandas as pd
#     df = pd.DataFrame()
#     def append_data(symbol, start, end,ktype, retry_count,pause):
#         data = tushare.get_hist_data(symbol, start=start, end=end,
#                                      ktype=ktype, retry_count=retry_count,
#                                      pause=pause)
#         data['code'] = symbol
#         df.append(data, ignore_index=True)
#     if isinstance(symbols, list) or isinstance(symbols, set) or isinstance(symbols, tuple) or isinstance(symbols, pd.Series):
#         tasks = []
#         for symbol in symbols:
#             try:
#                 tasks.append(gevent.spawn(append_data(symbol, start, end, ktype, retry_count,pause)))
#             except Exception as e:
#         gevent.joinall(tasks)
#         return df
#     else:
#         return None

res = tushare.get_hists(codes, start=start)

def get_a50_codes():
    import requests
    import re
    from name_code import name_code
    a50_url = "https://cn.investing.com/indices/ftse-china-a50-components"
    headers = {
        'Host': 'cn.investing.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36'
    }
    data = requests.get(a50_url, headers=headers).text
    data = re.findall('data-name=".{3,5}?"', data)
    data = [x[x.index('="') + 2:x.rindex('"')] for x in data]

    def delete_(s):
        if s:
            s = s.replace(" ", "")
            s = s.replace("Ａ", "A")
            s = s.replace("Ｂ", "B")
            s = s.replace("Ｇ", "G")
        return s

    res = []
    for d in data:
        res.append(name_code.get(delete_(d)))
    logging.info("获取A50成分股")
    return res


a50_codes = get_a50_codes()


def get_datas(codes):
    datas = {}
    for code in codes:
        d1 = res[res["code"] == code]
        from pandas import Series

        if len(d1.index) == len(date):
            d1.index = Series(range(len(date)))
        else:
            continue
        for i in range(len(date)):
            if datas.get(date[i]) == None:
                datas[date[i]] = [(d1.loc[i]["p_change"], d1.loc[i]["volume"] * d1.loc[i]["close"], code)]
            else:
                datas[date[i]].append((d1.loc[i]["p_change"], d1.loc[i]["volume"] * d1.loc[i]["close"], code))
    logging.info("数据处理")
    return datas


datas = get_datas(codes=codes)
a50_datas = get_datas(codes=a50_codes)
res = []


def weight_point(up_list, weights, point=0.15, cut=0):
    up_list.sort(key=lambda x: x[1], reverse=True)
    up_list = up_list[cut:len(up_list) - cut]
    weight_list = []
    for i in range(len(up_list)):
        if weights.get(up_list[i][0]) != None:
            weight_list.append(weights.get(up_list[i][0]))
    w_sum = sum(weight_list)
    point = w_sum * point
    weight_sum = 0
    for i in range(len(weight_list)):
        weight_sum = weight_sum + weight_list[i]
        if weight_sum > point:
            # if i == 0:
            #     return up_list[0][1]
            return up_list[i][1]
            # return (up_list[i-1][1]*(weight_sum-point)/weight_list[i]+up_list[i][1]*(point-(weight_sum-weight_list[i]))/weight_list[i])
    return 0


def weight_mean(up_list, weights):
    tmp_sum = 0
    for i in range(len(up_list)):
        if weights.get(up_list[i][0]) != None:
            tmp_sum += up_list[i][1] * weights.get(up_list[i][0])
    w_sum = sum(weights.values())
    return tmp_sum / w_sum


def weight_mean3(up_list, date_list, cut=0):
    ups = []
    up_list = up_list.copy()
    date_list = date_list.copy()
    up_list.reverse()
    date_list.reverse()
    for i in range(len(up_list)):
        if len(ups) >= cut:
            break
        if (datetime.datetime.strptime(date_list[i], '%Y-%m-%d') - datetime.datetime.strptime(date_list[i + 1],
                                                                                              '%Y-%m-%d')).days > 2:
            dalta = (datetime.datetime.strptime(date_list[i], '%Y-%m-%d') - datetime.datetime.strptime(date_list[i + 1],
                                                                                                       '%Y-%m-%d')).days / 3
            ups.append(up_list[i])
            dalta = int(dalta)
            for j in range(dalta):
                ups.append(up_list[i])
        else:
            ups.append(up_list[i])
    ups = ups[0:cut]
    ups.reverse()
    mean = 0
    i_sum = 0
    for i in range(cut):
        i_sum = i_sum + i + 1
    for i in range(cut):
        mean = mean + float(i + 1) / float(i_sum) * ups[i]
    return mean


def sum_(lists, func=lambda x: abs(x)):
    res = 0
    for d in lists:
        res = res + func(d)
    return res


def getRealData(codes=codes):
    tmp = tushare.get_realtime_quotes(codes)
    up_list = {}
    weights = {}
    for i in tmp.index:
        up_list[tmp.loc[i, "code"]] = (float(tmp.loc[i, 'price']) / float(tmp.loc[i, 'pre_close']) - 1.0) * 100.0
        weights[tmp.loc[i, "code"]] = float(tmp.loc[i, 'amount'])
    return list(up_list.items()), weights


sqrt_ = 0.8


def add_today(lists, dates, diffs):
    import time
    today = int(time.strftime("%w"))
    if today == 6 or today == 7:
        return lists
    elif datetime.datetime.now().time() > datetime.time(hour=15) or datetime.datetime.now().time() < datetime.time(
            hour=11, minute=30):
        return lists
    else:
        up_list, weights = getRealData(codes)
        mean = weight_mean(up_list, weights)
        medium = weight_point(up_list, weights, point=0.5)
        upTenPercent = weight_point(up_list, weights=weights,
                                    point=0.15)  # + weight_point(up_list,weights=weights,point=0.14)) / 2.0
        downTenPercent = weight_point(up_list, weights=weights,
                                      point=0.85)  # + weight_point(up_list,weights=weights,point=0.86)) / 2.0
        r = (abs(-mean + medium) + 0.574) * (upTenPercent - downTenPercent)
        r = r ** sqrt_
        func = lambda x: 0 if x < 0 else x
        rate = 1 - func((datetime.datetime.now() - datetime.datetime.combine(datetime.datetime.now().date(),
                                                                             datetime.time(hour=11,
                                                                                           minute=30))).seconds - 1.5 * 60 * 60) / (
                       2 * 60 * 60)
        k, b = (1.2 - 1) * rate + 1, -0. * rate
        # [1.9, 0.83]
        real_r = k * r + b
        lists.append(real_r)
        dates.append(datetime.datetime.now().date().strftime("%Y-%m-%d"))
        tmp = tushare.get_realtime_quotes('hs300')
        hs300_up = (float(tmp.loc[tmp.index[0], 'price']) / float(tmp.loc[tmp.index[0], 'pre_close']) - 1.0) * 100.0
        # hs300_up = 0
        diffs.append(real_r - hs300_up ** sqrt_ / 2.0)


def get_ress(datas):
    res = []
    for d in datas.values():
        up_list = []
        weights = {}
        for md in d:
            if md[0] < 11:
                up_list.append((md[2], md[0]))
            else:
                up_list.append((md[2], 0))
            weights[md[2]] = md[1]

        def takeSecond(elem):
            return elem[1]

        up_list.sort(key=takeSecond, reverse=True)
        mean = weight_mean(up_list, weights)
        medium = weight_point(up_list, weights, point=0.5)
        upTenPercent = weight_point(up_list, weights=weights,
                                    point=0.15)  # + weight_point(up_list,weights=weights,point=0.14)) / 2.0
        downTenPercent = weight_point(up_list, weights=weights,
                                      point=0.85)  # + weight_point(up_list,weights=weights,point=0.86)) / 2.0
        r = (abs(-mean + medium) + 0.574) * (upTenPercent - downTenPercent)
        # r = (-mean + medium + 0.5) * (upTenPercent - downTenPercent)
        # if r > 8:
        #     r = 8
        r = r ** sqrt_
        res.append(r)
    logging.info("结果计算")
    return res


def get_a50_weight_list(datas=datas):
    weights = []
    for k, d in datas.items():
        a50_sum = 0
        del_a50_sum = 0
        for md in d:
            code = md[2]
            if code in a50_codes:
                a50_sum += md[1]
            else:
                del_a50_sum += md[1]
        weights.append(0.5)
    return weights


def get_a50_res(res, a50res, weights):
    true_res = []
    for i, r in enumerate(res):
        w = weights[i]
        true_res.append((r - w * a50res[i]) / (1 - w))
    return true_res


res = get_ress(datas=datas)
a50_res = get_ress(datas=a50_datas)
# a50_res = get_a50_res(res, a50_res, get_a50_weight_list(datas))
res.reverse()
date.reverse()
a50_res.reverse()
# 结果平均
means = []
mean_num = 8
diffs = []
diffs_mean = []
for i in range(1, len(res) + 1):
    diffs.append(res[i - 1] - hs_p[i - 1] ** sqrt_)

add_today(lists=res, dates=date, diffs=diffs)

for i in range(1, len(res) + 1):
    if i < mean_num:
        m = sum(res[0:i]) / float(len(res[0:i]))
        diff = sum(diffs[0:i]) / float(len(diffs[0:i]))
    else:
        # m = sum(res[i-10:i]) / float(len(res[i-10:i]))
        m = weight_mean1(res[i - mean_num:i], mean_num)
        diff = weight_mean1(diffs[i - mean_num:i], mean_num)
    diffs_mean.append(diff)
    means.append(m)
# a50结果平均
a50_means = []
# a50_diffs = []
# a50_diffs_mean = []
for i in range(1, len(a50_res) + 1):
    if i < mean_num:
        m = sum(a50_res[0:i]) / float(len(a50_res[0:i]))
    else:
        m = weight_mean1(a50_res[i - mean_num:i], mean_num)
    a50_means.append(m)
logging.info("求平均序列")
# 剔除法求A50res
# def
# for i in range(1,len(res)+1):
#     if i < mean_num:
#         m = sum(res[0:i])/float(len(res[0:i]))
#     else:
#         # m = sum(res[i-10:i]) / float(len(res[i-10:i]))
#         m = weight_mean3(res[0:i],date[0:i],mean_num)
#     means.append(m)
ts1 = []
tss = []
index = 0
for time in date:
    if index % 4 == 0:
        ts1.append(index)
        tss.append(str(time)[5:10])
    index += 1

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
means_means = []
means_means_up = []
means_means_down = []
mean_num = 26
import numpy as np

std = np.std(means, ddof=1)

zero_i_list = []
zero_i_len = 120

for i in range(1, len(means) + 1):
    if i < mean_num:
        m = sum(means[0:i]) / float(len(means[0:i]))
    else:
        # m = sum(res[i-10:i]) / float(len(res[i-10:i]))
        m = weight_mean1(means[i - mean_num:i], mean_num, minus=0)
    # 半年平均
    if i < zero_i_len:
        hs_ = hs_p[:i]
    else:
        hs_ = hs_p[i - zero_i_len:i]
    hs_ = sorted(hs_)
    zero_i = 0
    for i in range(len(hs_)):
        if hs_[i] > 0:
            zero_i = i
            break
    zero = 0.6 * sum_(hs_p[:zero_i]) / (sum_(hs_p[zero_i:]) + sum_(hs_p[:zero_i])) + 0.4 * (zero_i) / float(len(hs_))
    # zero = zero/2.0
    if zero == 0:
        zero = 1
    zero_i_list.append(zero)
    means_means.append(m)

# 排序，求 0 位指标的值
zero_means = []
for i in range(1, len(means) + 1):
    if i < zero_i_len:
        means_ = means[:i]
    else:
        means_ = means[i - zero_i_len:i]
    means_ = sorted(means_)
    index = int(zero_i_list[i - 1] * float(len(means_) - 1))
    if index == 0:
        zero_means.append(means_[index])
    else:
        zero_means.append((means_[index] + means_[index - 1]) / 2.0)
    means_means_up.append(means_means[i - 1] + 0.3)
    means_means_down.append(means_means[i - 1] - 0.3)

plt.scatter(date, means, s=15, color="deeppink", linewidths=0.5)
plt.plot(date, means_means, color="blue", linewidth=0.9, label=str(mean_num) + '移动平均线')
plt.plot(date, a50_means, color="grey", linewidth=0.8)
plt.plot(date, zero_means, color="red", linewidth=1.3)
plt.plot(date, means_means_up, color="blue", linewidth=0.5)
plt.plot(date, means_means_down, color="blue", linewidth=0.5)
plt.plot(date, diffs_mean, color="black", linewidth=0.6, label='diff线:极值确认')
# plt.yticks(np.arange(23, 35, step=1))
plt.xticks(ts1, tss, rotation=55)
plt.grid()
plt.legend()
plt.show()  # 显示

"""
class A:
    name_code = {
"""