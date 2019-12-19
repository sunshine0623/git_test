# 输出数据库中所有的表名

from sqlalchemy import create_engine
from sqlalchemy import Column,String,Integer,DateTime,Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

db_info = {
	'user':'sa',
	'pwd':'6yhn6yhn',
	'ip':'192.168.67.34',
	'port':'1433', # sql server的默认端口号，可以修改
	'db':'SJMY', 
}

engine = create_engine('mssql+pymssql://%(user)s:%(pwd)s@%(ip)s:%(port)s/%(db)s?charset=utf8' % db_info,encoding='utf-8')

# 返回值是一个列表
print(engine.table_names())


