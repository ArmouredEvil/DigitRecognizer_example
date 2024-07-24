import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import pandas as pd
import numpy as np
import CNN

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
# 输出训练集基本信息
train.head()
train.info()

# 合并训练数据和测试数据，统一处理
all_data = pd.concat([train.drop("label", axis=1), test], axis=0, ignore_index=True)
# 对像素数据归一化
all_data = all_data / 255.0  # 像素值的范围是0~255
# 将像素数据重塑为28x28的图像，通道数为1（灰度图像）
all_data = all_data.values.reshape(-1, 1, 28, 28)
# 将数据集划分为训练集和测试集
train_data = all_data[:train.shape[0]]
test_data = all_data[train.shape[0]:]
# 为训练数据和测试数据创建张量
train_image = torch.tensor(train_data, dtype=torch.float32)
test_image = torch.tensor(test_data, dtype=torch.float32)
# 为训练标签创建张量
train_label = torch.tensor(train["label"].values, dtype=torch.long)
# 创建完整的训练数据集
dataset = torch.utils.data.TensorDataset(train_image, train_label)
# 创建数据加载器
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# 定义模型
model = CNN.CNN().to(device)
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)
# 训练模型
for epoch in range(200):
    loss_epoch = 0.0
    for image_x, label_y in train_loader:
        # 将数据移动到设备上
        image_x, label_y = image_x.to(device), label_y.to(device)
        # 前向传播
        output = model(image_x)
        # 计算损失函数
        loss = criterion(output, label_y)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 累计损失
        loss_epoch += loss.item()
    # 输出平均损失
    print(f"Epoch {epoch + 1}, Loss: {loss_epoch / len(train_loader)}")

'''
测试模型，得到预测结果
'''
test_image = test_image.to(device)
# 预测结果
output = model(test_image)
# 返回概率值最大的维度（即我们预测的概率最大的整数），并把结果转化为Series类型
test_output = pd.Series(output.argmax(dim=1).cpu().numpy())
# 将结果变为提交的格式
test_result = pd.concat([pd.Series(range(1, 28001), name="ImageId"), test_output.rename("label")], axis=1)
# 将结果保存为csv文件
test_result.to_csv('submission.csv', index=False)
