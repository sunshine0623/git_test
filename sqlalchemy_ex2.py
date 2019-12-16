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

Session = sessionmaker(bind=engine)

Base = declarative_base()

class News(Base):
	__tablename__ = 'News'
	serveraccountid = Column(String(50),primary_key=True)
	channel_id = Column(Integer)
	firstlogin = Column(DateTime)
	is_yc = Column(Boolean)


class orm_test(object):
	def __init__(self):
		self.session = Session()

	def add_one(self,serveraccountid,channel_id,firstlogin,is_yc):
		'''添加记录'''
		new_obj1 = News(serveraccountid=serveraccountid,channel_id=channel_id
			,firstlogin=firstlogin,is_yc=is_yc)
		# 添加一条记录
		self.session.add(new_obj1)
		# # 添加多条记录
		# self.session.add([new_obj1,new_obj2])
		self.session.commit()

	def get_all(self):
		# 返回的数据是一个列表，列表成员是表对象，可以通过(.字段名)读取字段值
		result = self.session.query(News).filter(News.is_yc==False).all()
		# result = self.session.query(News).all()		
		return result

	def update_data(self):
		new_obj = self.session.query(News).filter(News.is_yc==False).all()[0]
		if new_obj:
			new_obj.serveraccountid='update2'
			# 更新数据表的时候，原有的数据会被覆盖
			self.session.add(new_obj)
			# # 删除数据表的时候，原有的数据会被删除
			# self.session.delete(new_obj)
			self.session.commit()


if __name__ == '__main__':
	# # 创建一张表
	# News.metadata.create_all(engine)
	# # # 删除一张表
	# # News.metadata.drop_all(engine)
	# orm_test().add_one('lbs',1043,'20190918',True)
	orm_test().update_data()
	orm_test().session.close()



	
	
		

