# Pairing-transaction
基于协整配对和距离配对，构建了一个新的两阶段配对交易策略。在股票配对选择方面，首先通过协整分析选取具有类似股价趋势的候选股票配对;其次，利用欧式距离计算每个候选配对股票的距离，并根据最小距离选择最佳配对股票，以避免同时存在同一股票。买卖卖空的风险很小。在资金配置方面，考虑保证金交易制度的当前背景，在有限资本约束下解决最优资金配置方案，确保模型设计更接近实际交易情况。上证50指数的成分股作为实证对象。实证研究结果表明，构建的新两阶段方法可获得超额收益。
# 通过各个银行股票的实时价格与前一日收盘价格对比，取得最大最小的比率差，
#超过0.05那么就买入比率最小的那个（认为被低估），同时卖出最大的那个（认为被低估）
# 作者：何家雄

import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as scs
import scipy.optimize as sco
import talib as tl
from datetime import timedelta

#enable_profile()
bank_stocks=['601398.XSHG', '601288.XSHG','601939.XSHG','601988.XSHG']  # 设置银行股票 工行，建行
   
# 初始化参数
def initialize(context):
    # 初始化此策略
    # 设置要操作的股票池为空，每天需要不停变化股票池
    set_universe([])
    g.riskbench = '000300.XSHG'
    set_commission(PerTrade(buy_cost=0.0002, sell_cost=0.00122, min_cost=5))
    #设置滑点
    #这个价差可以是一个固定的值(比如0.02元, 交易时加减0.01元), 设定方式为：FixedSlippage(0.02)
    set_slippage(FixedSlippage(0)) 
    set_option('use_real_price', True)
    # 设置基准对比为沪深300指数
    
    g.inter = 0.005
    
# 每天交易前调用
def before_trading_start(context):
    #history读取的是前一天的数据
    g.df_last = history(1, unit='1d', field='close', security_list=bank_stocks, df=False, skip_paused=True, fq='pre')
  
# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def handle_data(context, data):
   
    raito = []
    
    for code in bank_stocks:
        #data[code].close存放前一个单位时间(按天回测, 是前一天, 按分钟回测, 则是前一分钟) 的数据
        #print(code,data[code].close,g.df_last[code][-1])
        raito.append( data[code].close / g.df_last[code][-1] )
        
    if not context.portfolio.positions.keys():
        #context.portfolio.positions.keys()当前持仓的股票代码
        if max(raito) - min(raito) > g.inter:
            min_index = raito.index(min(raito))
            order_value(bank_stocks[min_index], context.portfolio.total_value)
            g.is_stop = True
    else:
        code = context.portfolio.positions.keys()[0]
    
        index = bank_stocks.index(code)
        if raito[index] - min(raito) > g.inter:
            order_target(code, 0)
            min_index = raito.index(min(raito))
            order_value(bank_stocks[min_index], context.portfolio.total_value)
            g.is_stop = True
            
# 每天交易后调用
def after_trading_end(context):
    if bank_stocks[0] in context.portfolio.positions and context.portfolio.positions[bank_stocks[0]].total_amount > 0:
        record(code0=context.portfolio.positions[bank_stocks[0]].total_amount)
    if bank_stocks[1] in context.portfolio.positions and context.portfolio.positions[bank_stocks[1]].total_amount > 0:
        record(code1=context.portfolio.positions[bank_stocks[1]].total_amount)
    if bank_stocks[2] in context.portfolio.positions and context.portfolio.positions[bank_stocks[2]].total_amount > 0:
        record(code2=context.portfolio.positions[bank_stocks[2]].total_amount)
    if bank_stocks[3] in context.portfolio.positions and context.portfolio.positions[bank_stocks[3]].total_amount > 0:
        record(code3=context.portfolio.positions[bank_stocks[3]].total_amount)    
