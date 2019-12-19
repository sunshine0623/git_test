import pandas as pd
from sqlalchemy import create_engine

db_info = {
      'user':'sa',
      'pwd':'6yhn6yhn',
      'host':'192.168.67.34',
      'port':'1433',
      'database':'SJMY'
}

engine = create_engine('mssql+pymssql://%(user)s:%(pwd)s@%(host)s:%(port)s/%(database)s?charset=utf8' % db_info,encoding='utf-8')

sql = "select * from dbo.LSYJ where _logdate_ in ('20190701','20190731')"
data = pd.read_sql_query(sql,con=engine,index_col=['_logdate_','serveraccountid'])

print(data.head())