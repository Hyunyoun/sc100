import os
import pickle
import numpy as np
import pymysql
from PIL import Image
from pandas_dbms import read_sql_query
from contextlib import closing
#from data_utils import *


def connect_mysql(db_name):
    return pymysql.connect(host='192.168.30.200', port=3306,
                           user='hyunyoun', passwd='tition2787',
                           db=db_name, charset='utf8', local_infile=True)


def getStockList():
    sql = """SELECT SHORT_STOCK_ID
               FROM KR_MASTER
              WHERE (SHORT_STOCK_ID LIKE 'A0%' OR SHORT_STOCK_ID LIKE 'A1%')
                AND SHORT_STOCK_ID LIKE '%0'
                AND SECTION = '1'
              ORDER BY SHORT_STOCK_ID"""
    with closing(connect_mysql('STOCK_DB')) as con:
        df_stocks = read_sql_query(sql, con)
        stocks = list(df_stocks['SHORT_STOCK_ID'])
    return stocks


def getCandleHeight(quote, bounds, dpi):
    lb, ub = bounds
    o, h, l, c = quote
    if o <= c:
        high = int(dpi * (ub - c) / (ub - lb))
        low = int(dpi * (ub - o) / (ub - lb))
    else:
        high = int(dpi * (ub - o) / (ub - lb))
        low = int(dpi * (ub - c) / (ub - lb))
    v_high = int(dpi * (lb - h) / (ub - lb))
    v_low = int(dpi * (lb- l) / (ub - lb))
    return high, low, v_high, v_low

def candlestickManual(quotes, width=3, v_width=1, space=2, **kwargs):
    """
    Plot the time, open, high, low, close as a vertical line ranging
    from low to high.  Use a rectangular bar to represent the
    open-close span.  If close >= open, use colorup to color the bar,
    otherwise use colordown
    Parameters
    ----------
    quotes : sequence of quote sequences
        data to plot.  time must be in float date format - see date2num
        (time, open, high, low, close, ...) vs
        (time, open, close, high, low, ...)
        set by `ochl`
    width : float
        fraction of a day for the rectangle width
    colorup : color
        the color of the rectangle where close >= open
    colordown : color
         the color of the rectangle where close <  open
    alpha : float
        the rectangle alpha level
    ochl: bool
        argument to select between ochl and ohlc ordering of quotes
    Returns
    -------
    ret : tuple
        returns (lines, patches) where lines is a list of lines
        added and patches is a list of the rectangle patches added
    """
    
    n_data = len(quotes)
    dpi = n_data * (width + space)
    image = np.ones((dpi, dpi, 3)).astype(np.uint8) * 255

    gap = np.max(quotes) - np.min(quotes)
    ub = np.max(quotes) + 0.02 * gap
    lb = np.min(quotes) - 0.02 * gap
    bounds = (lb, ub)

    for n, quote in enumerate(quotes):
        o, h, l, c = quote

        high, low, v_high, v_low = getCandleHeight(quote, bounds, dpi)
        x_pos = n * (width + space) + int(space / 2)
        x_vpos = x_pos + int((width - v_width) / 2)
        if o > c:
            image[high:low, x_pos:x_pos+width, 1:3] = 0
            image[v_high:v_low, x_vpos:x_vpos+v_width, 1:3] = 0
        else:
            image[high:low, x_pos:x_pos+width, 0:2] = 0
            image[v_high:v_low, x_vpos:x_vpos+v_width, 0:2] = 0
    
    if 'fname' in kwargs:
        fname = kwargs['fname']
        im = Image.fromarray(image)
        im.save(fname)
    return image

def makeLabelManual(quotes, n):
    """
    Make 6 labels
    1: 1-day return (0: ~-3%, 1:-3~3%, 2: 3%~)
    2: 1-day return (0: ~-5%, 1:-5~5%, 2: 5%~)
    3: 5-day return (0: ~-3%, 1:-3~3%, 2: 3%~)
    4: 5-day return (0: ~-5%, 1:-5~5%, 2: 5%~)
    5: 5-day return (0: ~-10%, 1:-10~-3%, 2:-3~3%, 3:3~10%, 4: 10%~)
    """
    ret1 = quotes[n][3] / quotes[n-1][3] - 1
    label1 = 0 if ret1 < -0.03 else 1 if ret1 < 0.03 else 2
    
    ret2 = ret1
    label2 = 0 if ret1 < -0.05 else 1 if ret1 < 0.05 else 2
    
    ret3 = quotes[n+4][3] / quotes[n-1][3] - 1
    label3 = 0 if ret1 < -0.03 else 1 if ret1 < 0.03 else 2
    
    ret4 = ret3
    label4 = 0 if ret4 < -0.05 else 1 if ret4 < 0.05 else 2
    
    ret5 = ret3
    label5 = 0 if ret5 < -0.1 \
        else 1 if ret5 < -0.03 \
        else 2 if ret5 < 0.03 \
        else 3 if ret5 < 0.1 \
        else 4
    
    label = np.array([label1, label2, label3, label4, label5])
    return label


def makeChartManual(stock, s_date, e_date):
    df_quote = getStockData(stock, s_date=s_date, e_date=e_date)
    trade_date = list(df_quote['TRADE_DATE'])
    del df_quote['TRADE_DATE']
    quotes = [tuple(x) for x in df_quote.values]
    del df_quote
    print('Quotes of %s are loaded' % stock)

    n_window = 20
    n_data = len(quotes)-n_window
    data_list = []
    label_list = []
    for n in range(n_window, n_window+n_data-5):
        dt = trade_date[n-1]
        _dir = 'dpi100/raw_image2/' + stock
        if not os.path.exists(_dir):
            os.makedirs(_dir)
            print('directory of %s is created' % stock)
        fname = stock + '_' + dt + '.png'
        fname = os.path.join(_dir, fname)
        
        data = candlestickManual(quotes[n-n_window:n], fname=fname)
        data_list.append([data])
        
        label = makeLabelManual(quotes, n)        
        label_list.append([label])
        #insert_db('STOCK_DB', 'KR_CHART_RGB', buf)
    if n_data > 5:
        datas = np.concatenate(data_list)
        labels = np.concatenate(label_list)
        fname = 'dpi100/new/' + stock
        save2binary(datas, labels, fname)
        print('Saved %d images!' % (n_data-5))

    
def getStockData(stock_id, s_date=20100101, e_date=20161231):
    sql = """SELECT DA.TRADE_DATE
                  , DA.OPEN_PRICE
                  , DA.HIGH_PRICE
                  , DA.LOW_PRICE
                  , DA.END_PRICE
               FROM KR_PRICE_DAILY_ADJUST DA
              WHERE DA.SHORT_STOCK_ID = '%s'
                AND DA.TRADE_DATE BETWEEN '%s' AND '%s'
              ORDER BY DA.SHORT_STOCK_ID, DA.TRADE_DATE""" % (stock_id, s_date, e_date)
    with closing(connect_mysql('STOCK_DB')) as con:
        df_quote = read_sql_query(sql, con)
    return df_quote


def save2binary(data, labels, fname):
    n_data = data.shape[0]
    data_reshaped = np.reshape(data, (n_data, -1))

    outdata = dict(data=data_reshaped.astype(np.uint8), labels=labels.astype(np.uint8))
    with open(fname, 'wb') as handle:
        pickle.dump(outdata, handle, protocol=pickle.HIGHEST_PROTOCOL)