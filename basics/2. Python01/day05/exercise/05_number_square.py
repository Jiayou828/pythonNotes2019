# 1 写程序．输入一个整数n 代表正方形的宽度和高度．
# 打印数字组成的正方形:
#  如
#    输入: 5
#  打印:
#   1 2 3 4 5
#   1 2 3 4 5
#   1 2 3 4 5
#   1 2 3 4 5
#   1 2 3 4 5
#  　　输入: 4 
#  打印:
#   1 2 3 4
#   1 2 3 4
#   1 2 3 4
#   1 2 3 4


w = int(input("请输入宽度: "))
for _ in range(w):
    # 此处将被执行四次
    for x in range(1, w + 1):
        print(x, end=' ')
    print()  # 换行
    