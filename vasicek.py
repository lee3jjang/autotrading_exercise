import pyupbit as pub
from pyupbit import WebSocketManager
from scipy.optimize import minimize
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

import os
import yaml
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

from rich.console import Console
from rich.traceback import install
from rich.progress import track
install()

console = Console()

# 환경설정
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
ACCESS_KEY = config['ACCESS_KEY']
SECRET_KEY = config['SECRET_KEY']
PAST = config['PAST']
FUTURE = config['FUTURE']
NUM = config['NUM']
CAPACITY = config['CAPACITY']

## 계정 접속
account = pub.Upbit(ACCESS_KEY, SECRET_KEY)


if __name__ == '__main__':

    while True:

        # 종목 선정
        while True:
            # 티커 조회
            tickers = pub.get_tickers()
            tickers = np.array(list(filter(lambda x: x[:3] == 'KRW', tickers)))
            tickers = [
                'KRW-SAND', 'KRW-MANA', 'KRW-AAVE', 'KRW-STRK', 'KRW-SXP',
                'KRW-LOOM', 'KRW-BORA', 'KRW-SOL', 'KRW-ETH', 'KRW-FLOW',
                'KRW-NEAR', 'KRW-MATIC', 'KRW-AVAX', 'KRW-STRAX', 'KRW-XRP',
                'KRW-CELO', 'KRW-AXS', 'KRW-HUNT', 'KRW-WAVES'
            ]

            result = []
            for ticker in tickers:

                # 기초데이터 수집
                ohlcv = pub.get_ohlcv(ticker, interval='minute1', count=PAST, period=1)
                value = ohlcv['value'].mean()
                # if value <= MIN_VALUE:
                #     time.sleep(0.05)
                #     continue
                price = ohlcv['close']

                # 데이터 가공
                r_t = price.diff().shift(-1).iloc[:-1]/price[:-1]
                dr_t = r_t.diff().shift(-1).iloc[:-1]

                # 모수 추정
                def obj_func(p):
                    a, b, sigma = p
                    loglik = np.log(norm.pdf(x=dr_t-a*(b-r_t.iloc[:-1]), loc=0, scale=sigma)).sum()
                    return -loglik

                a, b, sigma = minimize(obj_func, (1.0, -0.0001, 0.002), method='Nelder-Mead').x

                # 손실 확률
                t = FUTURE
                r0 = r_t.iloc[-2]
                dt = 10/60
                rt = np.zeros([NUM, int(FUTURE/dt)+1])
                rt[:, 0] = r0
                dw = norm.rvs(size=(NUM, int(FUTURE/dt)), loc=0, scale=np.sqrt(dt))
                for i in range(dw.shape[1]):
                    rt[:, i+1] = rt[:, i] + a*(b-rt[:, i])*dt + sigma*dw[:, i]
                rt = rt[:, 1:]
                cum_rt = np.cumprod(1+rt, axis=1)

                # 전략 최적화
                def obj_func_2(p):
                    lb, ub = p
                    yld = np.array(list(map(lambda z: reduce(lambda x, y: x if (x >= ub or x <= lb) else y, z), cum_rt)))
                    yld -= 1 + 0.005
                    return -yld.mean()/yld.std()

                lb, ub = minimize(obj_func_2, (0.99, 1.01), method='Nelder-Mead').x
                yld = np.array(list(map(lambda z: reduce(lambda x, y: x if (x >= ub or x <= lb) else y, z), cum_rt)))
                yld -= 1 + 0.01
                
                # 결과 적재
                result.append([ticker, price[-1], a, b, sigma, lb, ub, yld.mean(), yld.std(), yld.mean()/yld.std(), value])

            # 타겟 선정
            target = (pd.DataFrame(result, columns=['ticker', 'price', 'a', 'b', 'sigma', 'lb', 'ub', 'yld_mean', 'yld_std', 'yld_rr', 'value'])
                .query('yld_mean > 0.002')
                .query('lb < 0.99')
                .sort_values(by='yld_rr', ascending=False)
            )
            if len(target) > 0:
                target = target.iloc[0]
                console.log(f"[Target] Ticker: {target['ticker']}, Mean: {target['yld_mean']:,.3f}, Std: {target['yld_std']:,.3f}, LB: {target['lb']}, UB: {target['ub']}")
                break
            else:
                console.log('Targeting...')
                continue
                
        # 매수 주문
        while True:
            if pub.get_current_price(target['ticker']) <= target['price']:
                account.buy_market_order(target['ticker'], CAPACITY)
                break
            else:
                console.log('Ordering...')
                time.sleep(0.01)
                continue

        # 주문 완료 대기
        while True:
            if len(account.get_order(target['ticker'])) == 0:
                balance = account.get_balance(target['ticker'])
                buying_price = float(list(filter(lambda x: x['currency'] == target['ticker'][4:], account.get_balances()))[0]['avg_buy_price'])
                console.log(f"[Buy] Ticker: {target['ticker']}, Buying Price: {buying_price}, Balance: {balance}")
                break
            time.sleep(0.01)

        # 매도 대기
        start_time = datetime.now()
        end_time = start_time + relativedelta(minutes=FUTURE)
        while datetime.now() <= end_time:
            curr_price = pub.get_current_price(target['ticker'])
            if curr_price <= buying_price*target['lb'] or curr_price >= buying_price*target['ub']:
                break
            else:
                console.log(f"[Waiting] Ticker: {target['ticker']}, Lower: {buying_price*target['lb']:,.1f} < Current: {curr_price:,.1f} (Profit: {curr_price-buying_price:,.1f}) < Upper: {buying_price*target['ub']:,.1f}, Remaining: {(end_time - datetime.now()).seconds}s")
                time.sleep(1)

        # 매도 주문
        account.sell_market_order(ticker=target['ticker'], volume=balance)
        while True:
            if len(account.get_order(target['ticker'], state='done')) >= 1:
                uuid = account.get_order(target['ticker'], state='done')[0]['uuid']
                selling_price = float(account.get_order(uuid)['trades'][0]['price'])
                console.log(f"[Sell] Ticker: {target['ticker']}, Selling Price: {selling_price:,.1f}, Profit: {buying_price-selling_price:+,.1f}, Balance: {balance}")
                break