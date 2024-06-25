import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 输入通道为3，输出通道为32，卷积核大小为3x3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 输入通道为32，输出通道为64，卷积核大小为3x3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 输入通道为64，输出通道为128，卷积核大小为3x3
        # 定义池化层
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化，核大小为2x2
        # 定义全连接层
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # 输入特征数为128*28*28，输出特征数为512
        self.fc2 = nn.Linear(512, 5)  # 输出层，5个中药类别

    def forward(self, x):
        # 前向传播
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义训练器类
class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, num_epochs):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.train_losses = []  # 保存训练损失
        self.train_accuracies = []  # 保存训练准确率
        self.test_losses = []  # 保存测试损失
        self.test_accuracies = []  # 保存测试准确率

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._train_epoch()  # 训练模型
            test_loss, test_acc = self._test_epoch()  # 测试模型
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    def _train_epoch(self):
        self.model.train()  # 训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc

    def _test_epoch(self):
        self.model.eval()  # 测试模式
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_loss = running_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        return test_loss, test_acc

# 主函数
def main():
    start_time = time.time()
    # 定义数据预处理的转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 使用 ImageFolder 创建训练和测试数据集
    train_dataset = datasets.ImageFolder(root='train/', transform=transform)
    test_dataset = datasets.ImageFolder(root='test/', transform=transform)

    # 创建数据加载器
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建 Trainer 实例
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)

    # 训练模型
    trainer.train()

    # 绘制训练过程
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(trainer.num_epochs), trainer.train_losses, label='Train Loss')
    plt.plot(range(trainer.num_epochs), trainer.test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(trainer.num_epochs), trainer.train_accuracies, label='Train Accuracy')
    plt.plot(range(trainer.num_epochs), trainer.test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    end_time = time.time()
    print(f'Total running time: {end_time - start_time:.2f} seconds')

    plt.show()

if __name__ == "__main__":
    main()