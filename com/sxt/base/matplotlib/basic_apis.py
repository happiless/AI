import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
# print(x)
# plt.plot(x)

plt.figure("sin wave")
plt.plot(x, np.sin(x))

plt.figure("cos wave")
plt.plot(x, np.cos(x))
# 保存图像到本地磁盘
plt.savefig('sin wave')

# 可以在一个画布画多个子图
# 让原有画布变成2行2列，可以画4张子图的画布
plt.subplot(3, 3, 1)
plt.xlim(-5, 15)
plt.ylim(-2, 2)
plt.ylabel("sin")
plt.plot(x, np.sin(x))
plt.subplot(3, 3, 2)
plt.axis([0, 15, -2, 2])
plt.ylabel('cos')
plt.plot(x, np.cos(x))

# 画不同种类、不同颜色的图
plt.subplot(3, 3, 3)
plt.plot(x, x + 0, '-g', label='-g')  # 实线、绿色
plt.plot(x, x + 1, '--c', label='--c')  # 虚线、浅蓝色

# 加上图例, 加上一点效果
plt.legend(loc='lower right', fancybox=True, framealpha=1, shadow=True, borderpad=1)

# 散点图
plt.subplot(3, 3, 4)
plt.plot(x, np.sin(x), 'o')
# plot画图速度是优于scatter的！scatter是一个点一个点去处理的！

plt.subplot(3, 3, 5)
plt.scatter(x, np.sin(x))

x = np.random.rand(100)
y = np.random.rand(100)
colors = np.random.rand(100)
size = 100 * np.random.rand(100)
plt.subplot(3, 3, 6)
plt.scatter(x, y, c=colors, s=size, alpha=0.7)

# 画等高线
x_ = np.linspace(-10, 10, 100)
y_ = np.linspace(-15, 15, 100)


# cbrt 开立方根
def f(x, y):
    return x ** 2 + (y - np.cbrt(x ** 2)) ** 2


X, Y = np.meshgrid(x_, y_)
Z = f(X, Y)

plt.subplot(3, 3, 7)
plt.contour(X, Y, Z)

plt.show()