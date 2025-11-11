import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建 XOR 数据集
x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
y_train = torch.tensor([0, 1, 1, 0], dtype=torch.long).to(device)

# 构建神经网络模型
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = XORNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    model.train()
    
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 评估模型
model.eval()
print("\n训练结果:")
with torch.no_grad():
    predictions = model(x_train)
    predicted_labels = torch.argmax(predictions, dim=1)
    
    for i, (x, y, pred) in enumerate(zip(x_train, y_train, predicted_labels)):
        print(f"输入: {x.cpu().numpy()} -> 预测: {pred.item()}, 实际: {y.item()}")