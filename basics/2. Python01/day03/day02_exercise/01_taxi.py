# 1. 北京出租车计费
#   收费标准:
#     3公里以内收费13元
#     超过3公里后基本单价为 2.3元/公里
#     空驶费: 超过15公里后，每公里加收基本单价
#         的50%作为返程的空驶费(3.45元/公里)
#   要求：
#     输入公里数，打印出费用的金额(以元为单位进行四舍五入)

km = int(input("请输入公里数: "))
# 方法1
# if 0 <= km <= 3:
#     print("收费13元")
# elif 3 < km <= 15:
#     fee = 13 + 2.3 * (km - 3)
#     print('收费', round(fee), '元')
# elif km > 15:
#     fee = 13 + 2.3 * (km - 3) + \
#           1.15 * (km - 15)
#     print('收费', round(fee), '元')

# 方法2
fee = 13
if km > 3:
    fee += 2.3 * (km - 3)
if km > 15:  # 超过15km加收的部分
    fee += 1.15 * (km - 15)
print('收费', round(fee), '元')
