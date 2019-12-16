# 更多资料详见：https://www.liaoxuefeng.com/wiki/1016959663602400/1017803857459008

from sqlalchemy import Column,String,create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd

# 创建对象的基类
Base = declarative_base()


# 定义表对象
class LSYJ(Base):
	# 表的名字
	__tablename__ = 'LSYJ'

	# 表的结构
	_logdate_ = Column(String(20), primary_key=True) # 在sqlalchemy的文档上有说明,一定要有主键,不然建表的时候会报错
	serveraccountid = Column(String(20), primary_key=True)
	level = Column(String(20))


# 初始化数据库连接
engine = create_engine('mssql+pymssql://sa:6yhn6yhn@192.168.67.34:1433/SJMY')

# 创建DBSession类型
DBSession = sessionmaker(bind=engine)

# 创建session
session = DBSession()

# 创建query查询，filter是where条件，最后调用one()返回唯一行，如果调用all()返回所有行
lsyj = session.query(LSYJ).filter(LSYJ._logdate_=='2019-07-01 00:00:00.000').all()

# all()→lsyj是一个列表；one()→是一个对象
print('type',type(lsyj))
print('_logdate_',lsyj[0]._logdate_)

data = [[lsyj[i]._logdate_,lsyj[i].serveraccountid,lsyj[i].level] for i in range(len(lsyj))]

print(pd.DataFrame(data,columns=['_logdate_','serveraccountid','level']))
print(pd.DataFrame(data,columns=['_logdate_','serveraccountid','level']).dtypes)

session.close()

====
