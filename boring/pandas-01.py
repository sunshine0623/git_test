import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import matplotlib

np.random.seed(100)
# def randint(self, a, b):"Return random integer in range [a, b], including both end points."
# closed：string 或者 None，默认值为 None，表示 start 和 end 这个区间端点是否包含在区间内，可以有三个值，'left' 表示左闭右开区间，'right' 表示左开右闭区间，None 表示两边都是闭区间
# 如果我们传入的是一个带有时间戳的日期 但是希望产生得到的时间都被规范到午夜，可以传入 normalize 选项
# freq:D、B(每工作日)、2D、Y、WOM-2MON、WOM-3TUE、WOM-1WED(每月的第几周的第几天)、W-MON、W-TUE（每周第几天）
# freq:M(每月最后一个日历日)、BM（每月最后一个工作日）、MS（每月第一个日历日）、BMS（每月第一个工作日）
# freq参数表：https://blog.csdn.net/RHJlife/article/details/89597796
df = pd.DataFrame(np.random.randint(-10,10,(10,3)),
	index=pd.date_range(start='1/1/2019 12:56:31',periods=10,normalize=True,freq='MS'),
	columns=list('ABC'))

# axis=0:按行累加；axis=1:按列累加
print(df.cumsum(axis=0))
print(df.head())

# # 默认是折线图，kind='line'，默认index为X轴，列为Y轴
# df.plot()
df.plot(kind='bar',x='C',y=['A','B'])
df.plot(kind='bar',stacked=True)
df.plot(kind='barh')
df.plot(kind='box')
df.plot(kind='scatter',x='A',y='B')
df.A[:5].abs().plot(kind='pie',figsize=(10,10))
plt.show()

