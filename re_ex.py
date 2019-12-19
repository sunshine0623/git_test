import re

data = '[_logdate_]:2019-01-01 00:00:00 & [serveraccountid]:123456 & [name]:陆昔昔'

pattern = re.compile('\[_logdate_]:(.*?) & \[serveraccountid]:(.*?) & \[name]:(.*)')

# serach只能找到1个（match必须从头匹配，若开始没有则返回None）
# 返回值是一个对象
if pattern.search(data):
      print("OK")
      _logdate_ = pattern.search(data).group(1)
      serveraccountid = pattern.search(data).group(2)
      name = pattern.search(data).group(3)
      print(_logdate_,serveraccountid,name)      
else:
      print("Fail")

with open('test.log','r',encoding='utf8') as f:
      # 读取文件所有内容并返回一个str
      data = f.read()
      print(data)
      print(type(data))

# findall找到所有满足匹配的
if pattern.findall(data):
      print("OK")
      # 返回值是列表套元组
      data = pattern.findall(data)
      print(data)
      print(type(data))
else:
      print("Fail")


