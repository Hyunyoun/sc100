# -*- coding: utf-8 -*-
"""
example use of pandas with oracle mysql postgresql sqlite
    - updated 9/18/2012 with better column name handling; couple of bug fixes.
    - used ~20 times for various ETL jobs.  Mostly MySQL, but some Oracle. 

    to do:  
            save/restore index (how to check table existence? just do select count(*)?), 
            finish odbc, 
            add booleans?, 
            sql_server?
"""
import numpy as np
# import cStringIO
from io import StringIO
import pymysql
from datetime import datetime, date, time

from pandas import read_sql_query
from pandas.core.api import DataFrame
from pandas import isnull
#from pandas import is_datetime64tz_dtype
from pandas.tseries.tools import to_datetime

from dateutil import parser
from contextlib import closing


dbtypes = {
    'mysql'     : {'DATE':'DATE', 'DATETIME':'DATETIME', 'INT':'BIGINT',
                   'FLOAT':'FLOAT', 'VARCHAR':'VARCHAR'},
    'oracle'    : {'DATE':'DATE', 'DATETIME':'DATE', 'INT':'NUMBER',
                   'FLOAT':'NUMBER', 'VARCHAR':'VARCHAR2'},
    'sqlite'    : {'DATE':'TIMESTAMP', 'DATETIME':'TIMESTAMP', 'INT':'NUMBER',
                   'FLOAT':'NUMBER', 'VARCHAR':'VARCHAR2'},
    'postgresql': {'DATE':'TIMESTAMP', 'DATETIME':'TIMESTAMP', 'INT':'BIGINT',
                   'FLOAT':'REAL', 'VARCHAR':'TEXT'}
}


def table_exists(name=None, con=None, flavor='mysql'):
    if flavor == 'mysql':
        sql = """SHOW TABLES
                 LIKE 'MYTABLE';""".replace('MYTABLE', name)
    elif flavor == 'sqlite':
        sql = """SELECT name
                   FROM sqlite_master
                  WHERE type = 'table' 
                    AND name = 'MYTABLE';""".replace('MYTABLE', name)
    elif flavor == 'oracle':
        sql = """SELECT table_name
                   FROM user_tables
                  WHERE table_name = 'MYTABLE'""".replace('MYTABLE', name.upper())
    elif flavor == 'postgresql':
        sql = """SELECT *
                   FROM pg_tables
                  WHERE tablename = 'MYTABLE';""".replace('MYTABLE', name)
    elif flavor == 'odbc':
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    with closing(con.cursor()) as cur:
        exists = cur.execute(sql) > 0
    return exists


def read_sql_query(sql, con, index_col=None, coerce_float=True, parse_dates=None): 
                  #, params=None, chunksize=None):
    """Read SQL query into a DataFrame.

    Returns a DataFrame corresponding to the result set of the query
    string. Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    Parameters
    ----------
    sql : string SQL query or SQLAlchemy Selectable (select or text object)
        to be executed.
    con : SQLAlchemy connectable(engine/connection) or database string URI
        or sqlite3 DBAPI2 connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    index_col : string or list of strings, optional, default: None
        Column(s) to set as index(MultiIndex)
    coerce_float : boolean, default True
        Attempt to convert values to non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}
    parse_dates : list or dict, default: None
        - List of column names to parse as dates
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the number of
        rows to include in each chunk.

    Returns
    -------
    DataFrame

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC

    """

    with closing(con.cursor()) as cur:
        cur.execute(sql)
        desc = cur.description
        cols = [desc[i][0] for i in range(len(desc))]
        data = list(cur.fetchall())
    
    frame = _wrap_result(data, cols, index_col=index_col,
                        coerce_float=coerce_float,
                        parse_dates=parse_dates)    
    return frame


def write_frame(frame, name=None, con=None, flavor='mysql', if_exists='fail'):
    """
    Write records stored in a DataFrame to specified dbms. 
    
    if_exists:
        'fail'    - create table will be attempted and fail
        'replace' - if table with 'name' exists, it will be deleted        
        'append'  - assume table with correct schema exists and add data.  if no table or bad data, then fail.
            ??? if table doesn't exist, make it.
        if table already exists.  Add: if_exists=('replace','append','fail')
    """

    if if_exists == 'replace' and table_exists(name, con, flavor):    
        with closing(con.cursor()) as cur:
            cur.execute("drop table " + name)

    # if table doesn't exist then create table
    if if_exists in ('fail', 'replace') or \
       (if_exists=='append' and table_exists(name, con, flavor)==False):
        schema = get_schema(frame, name, flavor)
        if flavor == 'oracle':
            schema = schema.replace(';','')

        with closing(con.cursor()) as cur:
            if flavor == 'mysql':
                cur.execute("SET sql_mode='ANSI_QUOTES';")
            # print 'schema\n', schema
            cur.execute(schema)
            # print 'created table'
        
    # bulk insert
    with closing(con.cursor()) as cur:
        if flavor == 'mysql':
            wildcards = ','.join(['%s'] * len(frame.columns))
            cols = [db_colname(k) for k in frame.dtypes.index]
            colnames = ','.join(cols)
            insert_sql = """INSERT INTO %s (%s)
                            VALUES (%s)""" % (name, colnames, wildcards)
            # print insert_sql
            # data = [tuple(x) for x in frame.values]
            data = [tuple([None if isnull(v) else v for v in rw]) for rw in frame.values] 
            # print data[0]
            cur.executemany(insert_sql, data)
          
        elif flavor == 'sqlite' or flavor == 'odbc':
            wildcards = ','.join(['?'] * len(frame.columns))
            insert_sql = 'INSERT INTO %s VALUES (%s)' % (name, wildcards)
            # print 'insert_sql', insert_sql
            data = [tuple(x) for x in frame.values]
            cur.executemany(insert_sql, data)

        elif flavor == 'oracle':
            cols = [db_colname(k) for k in frame.dtypes.index]
            colnames = ','.join(cols)
            colpos = ', '.join([':'+str(i+1) for i,f in enumerate(cols)])
            insert_sql = """INSERT INTO %s (%s)
                            VALUES (%s)""" % (name, colnames, colpos)

            data = [convertSequenceToDict(rec) for rec in frame.values] 
            cur.executemany(insert_sql, data)

        elif flavor == 'postgresql':
            postgresql_copy_from(frame, name, cur)

        else:
            raise NotImplementedError

        con.commit()
    return


def nan2none(df):
    dnp = df.values
    for rw in dnp:
        rw2 = tuple([None if v==np.Nan else v for v in rw])
        
    tpl_list = [tuple([None if v==np.Nan else v for v in rw]) for rw in dnp]
    return tpl_list

    
def db_colname(pandas_colname):
    '''convert pandas column name to a DBMS column name
        TODO: deal with name length restrictions, esp for Oracle
    '''
    colname =  pandas_colname.replace(' ','_').strip()
    return colname
    

def postgresql_copy_from(df, name, cur):
    # append data into existing postgresql table using COPY
    
    # 1. convert df to csv no header
    output = StringIO()
    
    # deal with datetime64 to_csv() bug
    have_datetime64 = False
    dtypes = df.dtypes
    for i, k in enumerate(dtypes.index):
        dt = dtypes[k]
        # print 'dtype', dt, dt.itemsize
        if str(dt.type) == "<type 'numpy.datetime64'>":
            have_datetime64 = True

    if have_datetime64:
        d2 = df.copy()    
        for i, k in enumerate(dtypes.index):
            dt = dtypes[k]
            if str(dt.type) == "<type 'numpy.datetime64'>":
                d2[k] = [v.to_pydatetime() for v in d2[k]]
        # convert datetime64 to datetime
        # ddt= [v.to_pydatetime() for v in dd] #convert datetime64 to datetime
        d2.to_csv(output, sep='\t', header=False, index=False)
    else:
        df.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    # print 'contents\n', contents
       
    # 2. copy from
    cur.copy_from(output, name)    
    return


def convertSequenceToDict(list_):
    # source: http://www.gingerandjohn.com/archives/2004/02/26/cx_oracle-executemany-example/
    """for  cx_Oracle:
        For each element in the sequence, creates a dictionary item equal
        to the element and keyed by the position of the item in the list.
        >>> convertListToDict(("Matt", 1))
        {'1': 'Matt', '2': 1}
    """
    dict_ = dict()
    arg_list = range(1, len(list_)+1)
    for k, v in zip(arg_list, list_):
        dict_[str(k)] = v
    return dict_

    
def get_schema(frame, name, flavor):
    types = dbtypes[flavor]  #deal with datatype differences
    column_types = []
    dtypes = frame.dtypes
    
    for i, k in enumerate(dtypes.index):
        dt = dtypes[k]
        # print 'dtype', dt, dt.itemsize
        if str(dt.type) == "<type 'numpy.datetime64'>":
            sqltype = types['DATETIME']
        elif issubclass(dt.type, np.datetime64):
            sqltype = types['DATETIME']
        elif issubclass(dt.type, (np.integer, np.bool_)):
            sqltype = types['INT']
        elif issubclass(dt.type, np.floating):
            sqltype = types['FLOAT']
        else:
            sampl = frame[frame.columns[i]][0]
            # print 'other', type(sampl) 
            if str(type(sampl)) == "<type 'datetime.datetime'>":
                sqltype = types['DATETIME']
            elif str(type(sampl)) == "<type 'datetime.date'>":
                sqltype = types['DATE']                   
            else:
                if flavor in ('mysql','oracle'):                
                    size = 2 + max(len(str(a)) for a in frame[k])
                    # print k, 'varchar sz', size
                    sqltype = types['VARCHAR'] + '(?)'.replace('?', str(size))
                else:
                    sqltype = types['VARCHAR']
        colname = db_colname(k)  #k.upper().replace(' ','_')                  
        column_types.append((colname, sqltype))
    
    columns = ',\n '.join('%s %s' % x for x in column_types)
    template_create = """CREATE TABLE %(name)s (
                      %(columns)s
                      );"""    
    # print 'COLUMNS:\n', columns
    create = template_create % {'name' : name, 'columns' : columns}
    return create


def _handle_date_column(col, format=None):
    if isinstance(format, dict):
        return to_datetime(col, errors='ignore', **format)
    else:
        if format in ['D', 's', 'ms', 'us', 'ns']:
            return to_datetime(col, errors='coerce', unit=format, utc=True)
        elif (issubclass(col.dtype.type, np.floating) or
              issubclass(col.dtype.type, np.integer)):
            # parse dates as timestamp
            format = 's' if format is None else format
            return to_datetime(col, errors='coerce', unit=format, utc=True)
        elif is_datetime64tz_dtype(col):
            # coerce to UTC timezone
            # GH11216
            return (to_datetime(col, errors='coerce')
                    .astype('datetime64[ns, UTC]'))
        else:
            return to_datetime(col, errors='coerce', format=format, utc=True)


def _parse_date_columns(data_frame, parse_dates):
    """
    Force non-datetime columns to be read as such.
    Supports both string formatted and integer timestamp columns
    """
    # handle non-list entries for parse_dates gracefully
    if parse_dates is True or parse_dates is None or parse_dates is False:
        parse_dates = []

    if not hasattr(parse_dates, '__iter__'):
        parse_dates = [parse_dates]

    for col_name in parse_dates:
        df_col = data_frame[col_name]
        try:
            fmt = parse_dates[col_name]
        except TypeError:
            fmt = None
        data_frame[col_name] = _handle_date_column(df_col, format=fmt)


def _wrap_result(data, columns, index_col=None, coerce_float=True,
                 parse_dates=None):
    """Wrap result set of query in a DataFrame """

    frame = DataFrame.from_records(data, columns=columns,
                                   coerce_float=coerce_float)

    _parse_date_columns(frame, parse_dates)

    if index_col is not None:
        frame.set_index(index_col, inplace=True)

    return frame


def connect_mysql():
    return pymysql.connect(host='192.168.60.100', port=3306,
                           user='hyunyoun', passwd='tition2787',
                           db='STOCK_DB', charset='utf8', local_infile=True)

###############################################################################

'''
def test_sqlite(name, testdf):
    print '\nsqlite, using detect_types=sqlite3.PARSE_DECLTYPES for datetimes'
    import sqlite3
    with sqlite3.connect('test.db', detect_types=sqlite3.PARSE_DECLTYPES) as con:
        #conn.row_factory = sqlite3.Row
        write_frame(testdf, name, con=con, flavor='sqlite', if_exists='replace')
        df_sqlite = read_db('select * from '+name, con=con)    
        print 'loaded dataframe from sqlite', len(df_sqlite)   
    print 'done with sqlite'


def test_oracle(name, testdf):
    print '\nOracle'
    import cx_Oracle
    with cx_Oracle.connect('YOURCONNECTION') as ora_con:
        testdf['d64'] = np.datetime64( testdf['hire_date'] )
        write_frame(testdf, name, con=ora_con, flavor='oracle', if_exists='replace')    
        df_ora2 = read_db('select * from '+name, con=ora_con)    

    print 'done with oracle'
    return df_ora2
   
    
def test_postgresql(name, testdf):
    #from pg8000 import DBAPI as pg
    import psycopg2 as pg
    print '\nPostgresQL, Greenplum'    
    pgcn = pg.connect(YOURCONNECTION)
    print 'df frame_query'
    try:
        write_frame(testdf, name, con=pgcn, flavor='postgresql', if_exists='replace')   
        print 'pg copy_from'    
        postgresql_copy_from(testdf, name, con=pgcn)    
        df_gp = read_db('select * from '+name, con=pgcn)    
        print 'loaded dataframe from greenplum', len(df_gp)
    finally:
        pgcn.commit()
        pgcn.close()
    print 'done with greenplum'
'''


def test_mysql(name, testdf):
    with closing(connect_mysql()) as con:
        write_frame(testdf, name='test_df', con=con, 
                    flavor='mysql', if_exists='replace')
        df_mysql = read_db('select * from '+name, con=con)    
        # print 'loaded dataframe from mysql', len(df_mysql)
    print('mysql done')

##############################################################################

if __name__=='__main__':

    from pandas import DataFrame
    from datetime import datetime
    
    print("""Aside from sqlite, you'll need to install the driver and set a valid
            connection string for each test routine.""")
    
    test_data = {
        "name": ['Joe', 'Bob', 'Jim', 'Suzy', 'Cathy', 'Sarah'],
        "hire_date": [datetime(2012,1,1),
                      datetime(2012,2,1),
                      datetime(2012,3,1),
                      datetime(2012,4,1),
                      datetime(2012,5,1),
                      datetime(2012,6,1)],
        "erank": [1,   2,   3,   4,   5,   6  ],
        "score": [1.1, 2.2, 3.1, 2.5, 3.6, 1.8]
    }
    df = DataFrame(test_data)

    name = 'test_df'
    # test_sqlite(name, df)
    # test_oracle(name, df)
    # test_postgresql(name, df)    
    test_mysql(name, df)        
    
    print('done')
