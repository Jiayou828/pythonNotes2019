# 3. 给出一个年份，判断是否为闰年并打印结果
#   闰年规则: 每四年一闰，每百年不闰，四百年又是一个闰年
#   例:
#     2016年 闰年 
#     2100年 不是闰年
#     2400年 是闰年

y = int(input('请输入年份: '))
if y % 400 == 0:
    print(y, "是闰年")
elif y % 100 == 0:
    print(y, "不是闰年")
elif y % 4 == 0:
    print(y, '是闰年')
else:
    print(y, "不是闰年")