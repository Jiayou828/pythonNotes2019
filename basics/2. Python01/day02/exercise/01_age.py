# 01_age.py

# 练习：
#   1. 今天是小明20周岁的生日，假设每年365天，
#   计算他过了多少个星期，余多少天

age = 20
days = 20 * 365
weeks = days // 7  # 多少个星期
day = days % 7  # 余多少天
print("小时过了:", weeks,
      "个星期，余", day, "天")
