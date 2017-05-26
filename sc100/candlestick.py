import os
import pickle
import time
import datetime
import pymysql
import numpy as np
import pandas as pd
from pandas.core.api import Series, DataFrame
from pandas_dbms import read_sql_query, write_frame
from contextlib import closing
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from data_utils import *

def connect_mysql(db_name):
    return pymysql.connect(host='192.168.30.200', port=3306,
                           user='hyunyoun', passwd='tition2787',
                           db=db_name, charset='utf8', local_infile=True)


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw( )
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h, 4 )
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll( buf, 3, axis = 2 )
    # pop ALPHA channel
    buf = buf[:, :, :3]
    return buf


def fig2dataframe(fig):
    """
    @brief Convert a Matplotlib figure to a data frame with RGB channels and return it
    @param fig a matplotlib figure
    @return a data frame of RGB values
    """
    # draw the renderer
    fig.canvas.draw( )
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = (w*h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode. Keep only RGB.
    buf = buf[:, 1:]

    col_rgb = ['SHORT_STOCK_ID','TRADE_DATE','X_POS','Y_POS','R','G','B']
    df = DataFrame(columns=col_rgb)
    df.loc[:, 'X_POS'] = np.tile(np.arange(w), h)
    df.loc[:, 'Y_POS'] = np.repeat(np.arange(h), w)
    df.loc[:, ['R', 'G','B']] = buf
    return df


def candlestick_sq(quotes, **kwargs):
    if 'dpi' in kwargs:
        dpi = kwargs['dpi']
    else:
        dpi = 100

    fig = plt.figure(dpi=dpi, figsize=(1, 1), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    _candlestick(ax, quotes)

    if 'fname' in kwargs:
        plt.savefig(kwargs['fname']) #, bbox_inches='tight')
    buf = fig2data(fig)
    plt.close()
    return buf


def _candlestick(ax, quotes, width=0.6, v_width=0.2, colorup='b', colordown='r',
                 alpha=1.0, ochl=False):

    """
    Plot the time, open, high, low, close as a vertical line ranging
    from low to high.  Use a rectangular bar to represent the
    open-close span.  If close >= open, use colorup to color the bar,
    otherwise use colordown
    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
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

    OFFSET = width / 2.0
    v_OFFSET = v_width / 2.0

    lines = []
    patches = []
    for q in quotes:
        if ochl:
            t, open, close, high, low = q[:5]
        else:
            t, open, high, low, close = q[:5]

        if close >= open:
            color = colorup
            lower = open
            height = close - open
            v_height = high - low
        else:
            color = colordown
            lower = close
            height = open - close
            v_height = high - low

        vline = Rectangle(
            xy=(t - v_OFFSET, low),
            width=v_width,
            height=v_height,
            facecolor=color,
            edgecolor=color,
        )
        vline.set_alpha(alpha)

        rect = Rectangle(
            xy=(t - OFFSET, lower),
            width=width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect.set_alpha(alpha)

        lines.append(vline)
        patches.append(rect)
        ax.add_patch(vline)
        ax.add_patch(rect)
    ax.autoscale_view()
    xmin = quotes[0][0]-0.5
    xmax = quotes[-1][0]+0.5
    ax.set_xlim(xmin, xmax)

    return lines, patches


def getStockList_old():
    sql = """SELECT DISTINCT PD.SHORT_STOCK_ID
               FROM KR_PRICE_DAILY_ADJUST PD
              WHERE PD.SHORT_STOCK_ID LIKE 'A%'
              ORDER BY PD.SHORT_STOCK_ID"""
    with closing(connect_mysql('STOCK_DB')) as con:
        df_stocks = read_sql_query(sql, con)
        stocks = list(df_stocks['SHORT_STOCK_ID'])
    return stocks

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


def getBusDay(s_date, e_date):
    sql = """SELECT *
               FROM KR_BUS_DAY BD
              WHERE BD.TRADE_DATE BETWEEN '%s' AND '%s'
              ORDER BY BD.TRADE_DATE""" % (s_date, e_date)
    with closing(connect_mysql('STOCK_DB')) as con:
        df_days = read_sql_query(sql, con)
    return df_days

def getStockData(stock_id, s_date=20100101, e_date=20161231):
    sql = """SELECT TRADE_DATE
                  , OPEN_PRICE
                  , HIGH_PRICE
                  , LOW_PRICE
                  , END_PRICE
               FROM KR_PRICE_DAILY_ADJUST
              WHERE SHORT_STOCK_ID = '%s'
                AND TRADE_DATE BETWEEN '%s' AND '%s'
              ORDER BY TRADE_DATE""" % (stock_id, s_date, e_date)
    with closing(connect_mysql('STOCK_DB')) as con:
        df_quote = read_sql_query(sql, con)
    return df_quote

def save2binary(data, labels, fname):
    n_data = data.shape[0]
    data_reshaped = np.reshape(data, (n_data, -1))

    outdata = dict(data=data_reshaped.astype(np.uint8), labels=labels.astype(np.uint8))
    with open(fname, 'wb') as handle:
        pickle.dump(outdata, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loadbinary2array(fname, w=100, h=100):
    indata = np.load(fname)
    y_data = indata[:, 0]
    X_data = indata[:, 1:].reshape((-1, w, h, 3))
    return X_data, y_data

def make_label(quotes, n):
    """
    Make 6 labels
    1: 1-day return (0: ~-3%, 1:-3~3%, 2: 3%~)
    2: 1-day return (0: ~-5%, 1:-5~5%, 2: 5%~)
    3: 5-day return (0: ~-3%, 1:-3~3%, 2: 3%~)
    4: 5-day return (0: ~-5%, 1:-5~5%, 2: 5%~)
    """
    ret1 = quotes[n][4] / quotes[n-1][4] - 1
    label1 = 0 if ret1 < -0.03 else 1 if ret1 < 0.03 else 2
    
    ret2 = quotes[n][4] / quotes[n-1][4] - 1
    label2 = 0 if ret1 < -0.05 else 1 if ret1 < 0.05 else 2
    
    ret3 = quotes[n+4][4] / quotes[n-1][4] - 1
    label3 = 0 if ret1 < -0.03 else 1 if ret1 < 0.03 else 2
    
    ret4 = quotes[n+4][4] / quotes[n-1][4] - 1
    label4 = 0 if ret4 < -0.05 else 1 if ret4 < 0.05 else 2
    label = np.array([label1, label2, label3, label4])
    return label

def makeChart(stock, s_date, e_date):
    df_quote = getStockData(stock, s_date=s_date, e_date=e_date)
    trade_date = list(df_quote['TRADE_DATE'])
    df_quote['TRADE_DATE'] = np.arange(df_quote.shape[0]) + 0.5
    quotes = [tuple(x) for x in df_quote.values]
    print('Quotes of %s are loaded' % stock)
    
    n_window = 20
    n_data = len(quotes)-n_window
    data_list = []
    label_list = []
    for n in range(n_window, n_window+n_data-5):
        dt = trade_date[n-1]
        _dir = 'dpi100/raw_image/' + stock
        if not os.path.exists(_dir):
            os.makedirs(_dir)
            print('directory of %s is created' % stock)
        fname = stock + '_' + dt + '.png'
        fname = os.path.join(_dir, fname)
        
        data = candlestick_sq(quotes[n-n_window:n], fname=fname)
        data_list.append([data])
        
        label = make_label(quotes, n)        
        label_list.append([label])
        #insert_db('STOCK_DB', 'KR_CHART_RGB', buf)
        if n % 100 == 0:
            print(n)
    datas = np.concatenate(data_list)
    labels = np.concatenate(label_list)
    
    fname = 'dpi100/' + stock
    #save2binary(datas, labels, fname)
    print('Saved!')
    return datas, labels
    

def insert_db(db_name, table_name, frame):
    with closing(connect_mysql(db_name)) as con:
        write_frame(frame, name=table_name, con=con, flavor='mysql', if_exists='append')