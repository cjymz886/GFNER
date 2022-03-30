import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体和负号正常显示
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False

label_list = ['train', 'test']    # 横坐标刻度显示值
num_list1 = [163, 23]      # 纵坐标值1
num_list2 = [394, 48]      # 纵坐标值2
x = range(len(num_list1))
"""
绘制条形图
left:长条形中点横坐标
height:长条形高度
width:长条形宽度，默认值0.8
label:为后面设置legend准备
"""
rects1 = plt.bar(x=[i+0.1 for i in x], height=num_list2, width=0.2, alpha=0.8, color='royalblue', label="Locate and Label")
rects2 = plt.bar(x=[i + 0.3 for i in x], height=num_list1, width=0.2, color='orangered', label="TFNER")
plt.ylim(0, 450)     # y轴取值范围
plt.ylabel("Seconds")
"""
设置x轴刻度显示值
参数一：中点坐标
参数二：显示值
"""
plt.xticks([x[0]+0.2, x[1]+0.2], label_list)
plt.xlabel("Mode")
plt.legend()     # 设置题注
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
plt.tight_layout()

plt.savefig('compare.png')
plt.show()