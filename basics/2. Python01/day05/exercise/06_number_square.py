# 2 写程序．输入一个整数n 代表正方形的宽度和高度．
# 打印数字组成的正方形:
#  如
#    输入: 5
#  打印:
#   1 2 3 4 5
#   2 3 4 5 6
#   3 4 5 6 7
#   4 5 6 7　8
#   5 6 7　8 9
#  　　输入: 4 
#  打印:
#   1 2 3 4
#   2 3 4 5
#   3 4 5 6
#   4 5 6 7

w = int(input("请输入宽度: "))
for y in range(1, w + 1):  # y代表当前的行数
    for x in range(y, y + w):
        print("%2d" % x, end=' ')
    print()  # 换行

