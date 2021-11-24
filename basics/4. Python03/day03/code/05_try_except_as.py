# 03_try_except.py


# 此示例示意try-except语句中  as 的用法

def div_apple(n):
    print('%d个苹果您想分给几个人?' % n)
    s = input('请输入人数: ')
    cnt = int(s)  # <<= 可能触发ValueError错误异常
    # 以下一行可能触发ZeroDivisionError错误异常
    result = n / cnt
    print("每人个分了", result, '个苹果')

# 以下是调用者
try:
    print("开始分苹果")
    div_apple(10)
    print("分苹果完成")
except ValueError as err:
    print('苹果退回来不分了')
    print('错误信息是:', err)

print("程序正常退出")