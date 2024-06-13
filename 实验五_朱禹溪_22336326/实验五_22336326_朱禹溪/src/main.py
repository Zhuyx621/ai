import numpy as np
import matplotlib.pyplot as plt
class LogisticRegressionModel:
    def __init__(self, X_train, y_train, X_test, y_test, lr=0.01, epochs=10000):
        """  
        初始化逻辑回归模型。  
  
        参数:  
        X_train: 训练集特征数据  
        y_train: 训练集标签数据  
        X_test: 测试集特征数据  
        y_test: 测试集标签数据  
        lr: 学习率,默认为0.01  
        epochs: 迭代次数,默认为10000  
        """  
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.lr = lr
        self.epochs = epochs
        self.theta = np.zeros(X_train.shape[1])
        self.loss_history = []
    def sigmoid(self, z):
        """  
        Sigmoid激活函数,将输入值映射到0到1之间。  
  
        参数:  
        z: 输入值  
  
        返回:  
        激活后的输出值  
        """  
        return 1 / (1 + np.exp(-z))
    def binary_cross_entropy_loss(self, y_true, y_pred):
        """  
        计算二元交叉熵损失。  
  
        参数:  
        y_true: 真实标签值  
        y_pred: 预测概率值  
  
        返回:  
        损失值  
        """  
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    def fit(self):
        """  
        使用梯度下降算法训练模型。  
  
        无返回值,但会更新模型的权重theta和损失历史记录loss_history。  
        """  
        for _ in range(self.epochs):
            z = np.dot(self.X_train, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(self.X_train.T, (h - self.y_train)) / self.y_train.size
            self.theta -= self.lr * gradient
            loss = self.binary_cross_entropy_loss(self.y_train, h)
            self.loss_history.append(loss)
    def predict(self):
        """  
        对测试集进行预测,并返回预测结果。  
  
        返回:  
        预测结果数组  
        """  
        return np.round(self.sigmoid(np.dot(self.X_test, self.theta)))
    def calculate_accuracy(self, predictions):
        """  
        计算预测结果的准确率。  
  
        参数:  
        predictions: 预测结果数组  
  
        返回:  
        准确率  
        """  
        return np.mean(predictions == self.y_test)

    def plot(self, predictions):
        """  
        绘制测试集的散点图、决策边界和训练损失曲线。  
  
        参数:  
        predictions: 预测结果数组  
  
        无返回值,但会生成并显示图表。  
        """  
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_test[:, 1], self.X_test[:, 2], c=self.y_test, cmap='coolwarm', marker='o', label='Actual')
        plt.scatter(self.X_test[:, 1], self.X_test[:, 2], c=predictions, cmap='coolwarm', marker='o', label='Logistic Regression Prediction')
        theta0 = self.theta[0]
        theta1 = self.theta[1]
        theta2 = self.theta[2]
        x_values = np.linspace(-2, 2, 10)
        y_values = -(theta1 * x_values + theta0) / theta2
        plt.plot(x_values, y_values, color='red', linestyle='solid', linewidth=2, label='Decision Boundary')
        plt.xlabel('Age (Scaled)')
        plt.ylabel('Estimated Salary (Scaled)')
        plt.title('Logistic Regression Predictions vs Actual')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross-Entropy Loss')
        plt.title('Training Loss Curve')
        plt.show()


class PerceptronModel:
    def __init__(self, X_train, y_train, X_test, y_test, learning_rate=0.001, epochs=1000, lambda_param=0.01):
        """  
        初始化感知机模型。  
  
        参数:  
        X_train: 训练集特征数据  
        y_train: 训练集标签数据  
        X_test: 测试集特征数据  
        y_test: 测试集标签数据  
        learning_rate: 学习率,默认为0.01  
        epochs: 迭代次数,默认为1000  
        lambda_param: 正则化参数,默认为0.01  
        """  
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(X_train.shape[1])
        self.bias = 0
        self.loss_history = []
        self.lambda_param = lambda_param  # 正则化参数


    def fit(self):
        for _ in range(self.epochs):
            activation = np.dot(self.X_train, self.weights) + self.bias
            y_pred = np.where(activation >= 0, 1, 0)
            self.weights += self.learning_rate * (np.dot(self.X_train.T, (self.y_train - y_pred)) - self.lambda_param * self.weights)  # 添加L2正则化项
            self.bias += self.learning_rate * np.sum(self.y_train - y_pred)
            loss = np.mean(np.abs(self.y_train - y_pred)) + self.lambda_param / 2 * np.sum(self.weights ** 2)  # 添加L2正则化项
            self.loss_history.append(loss)

    def predict(self):
        return np.where(np.dot(self.X_test, self.weights) + self.bias >= 0, 1, 0)

    def calculate_accuracy(self, predictions):
        return np.mean(predictions == self.y_test)

    def plot(self, predictions):
        """  
        绘制测试集的散点图、决策边界和训练损失曲线（对于感知机,通常是误分类点的数量）。  
  
        参数:  
        predictions: 预测结果数组  
  
        无返回值,但会生成并显示图表。  
        """  
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_test[:, 1], self.X_test[:, 2], c=self.y_test, cmap='coolwarm', marker='o', label='Actual')
        plt.scatter(self.X_test[:, 1], self.X_test[:, 2], c=predictions, cmap='coolwarm', marker='o', label='Perceptron Prediction')
        theta0 = self.bias
        theta1, theta2 = self.weights[1], self.weights[2]
        x_values = np.linspace(-2, 2, 10)
        y_values = -(theta1 * x_values + theta0) / theta2
        plt.plot(x_values, y_values, color='red', linestyle='solid', linewidth=2, label='Decision Boundary')
        plt.xlabel('Age (Scaled)')
        plt.ylabel('Estimated Salary (Scaled)')
        plt.title('Perceptron Predictions vs Actual')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Number of Misclassifications')
        plt.title('Perceptron Loss Curve')
        plt.show()
def main():
    data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std
    X_scaled = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))
    np.random.seed(42)
    indices = np.random.permutation(X_scaled.shape[0])
    train_indices, test_indices = indices[:int(0.8*X_scaled.shape[0])], indices[int(0.8*X_scaled.shape[0]):]
    X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    lr_model = LogisticRegressionModel(X_train, y_train, X_test, y_test)
    perceptron_model = PerceptronModel(X_train, y_train, X_test, y_test)

    lr_model.fit()
    perceptron_model.fit()

    lr_pred = lr_model.predict()
    perceptron_pred = perceptron_model.predict()

    lr_accuracy = lr_model.calculate_accuracy(lr_pred)
    perceptron_accuracy = perceptron_model.calculate_accuracy(perceptron_pred)

    print("Logistic Regression Accuracy:", lr_accuracy)
    print("Perceptron Accuracy:", perceptron_accuracy)

    lr_model.plot(lr_pred)
    perceptron_model.plot(perceptron_pred)

if __name__ == '__main__':
    main()