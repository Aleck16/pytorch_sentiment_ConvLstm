# *_*coding:utf-8 *_* 
# Author:Aleck_
# @Time: 18-3-30 上午10:26

import matplotlib.pyplot as plt
import numpy as np

# 读取打印结果文件数据
f = open("logs/out20180329_155910.txt", 'r')

train_losses = []
train_acc = []
dev_losses = []
dev_acc = []

for line in f.readlines():
    if "train avg_loss:" in line:
        line = line.split(" ")
        train_losses.append(float(line[2].split(":")[1]))
        train_acc.append(float(line[-1].split(":")[1].split("\\")[0]))
    elif "dev avg_loss:" in line:
        line = line.split(" ")
        dev_losses.append(float(line[1].split(":")[1]))
        dev_acc.append(float(line[-1].split(":")[1].split("\\")[0]))

x = np.arange(21)

# view loss image
# marker数据点样式，linewidth线宽，linestyle线型样式，color颜色
plt.plot(x, train_losses, marker="*", linewidth=3, linestyle="--", color="orange")
plt.plot(x, dev_losses)
plt.title("loss title")
plt.xlabel("epoch")
plt.ylabel("loss")
# 设置图例
plt.legend(["train loss", "dev loss"], loc="upper right")
plt.grid(True)
plt.show()

# view accuracy image
# # marker数据点样式，linewidth线宽，linestyle线型样式，color颜色
# plt.plot(x, train_acc, marker="*", linewidth=3, linestyle="--", color="orange")
# plt.plot(x, dev_acc)
# plt.title("accuracy title")
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# # 设置图例
# plt.legend(["train acc", "dev acc"], loc="upper right")
# plt.grid(True)
# plt.show()
