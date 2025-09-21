import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
import glob
import os
import copy

# tensorboard导入
from torch.utils.tensorboard import SummaryWriter

# 步骤一、先初始化SummaryWriter，参数为写入文件的路径
log = r'log/language_recognition'
# 指定tensorboard存储数据的位置
writer = SummaryWriter(log)


# 所有类别
specises = ['100', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115',
            '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130',
            '131', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146',
            '147', '148', '149', '150']
# 数据集文件地址
current_dir = os.path.dirname(os.path.abspath(__file__))
image_folder_dir = os.path.join(current_dir, 'dataset', 'handwritten_50')

# 看看原来的目录下有多少照片
# all_img_count = 6419
# 所有img的路径
all_img_paths = []
# 所有img的标签
all_img_labels = []
all_img_count = 0

for spec in specises:
    spec_folder = os.path.join(image_folder_dir, spec)  #拼接访问路径到目录100，保存文件路径

    spec_folder_paths = glob.glob(spec_folder + '/*.png')   #模糊查询该目录下所有照片路径，列表保存完整路径

    # 便利每个目录下的照片路径存到总路径all_img_paths
    for spec_folder_path in spec_folder_paths:
        all_img_paths.append(spec_folder_path)
    # 便利每个目录下的照片对应的标签存到总路径all_img_labels
    for i in range(len(spec_folder_paths)):
        # 将标签都-100
        all_img_labels.append(int(spec) - 100)
    # print(len(all_pic_paths))
    #图片总数
    all_img_count += len(os.listdir(os.path.join(image_folder_dir, spec)))

    # print(spec, len(os.listdir(os.path.join(image_folder_dir, spec))))
# print(all_pic_count)
# 总的样本数6419
print(len(all_img_paths))
print(len(all_img_labels))

# 随机打乱全部图片顺序
index = np.random.permutation(len(all_img_paths))
print(index)
all_img_paths = np.array(all_img_paths)[index]
all_img_labels = np.array(all_img_labels)[index]

# 数据比例划分为1/5   训练集数5135
s = int(len(all_img_paths) * 0.8)
print(s)
# 获取训练集
train_imgs = all_img_paths[:s]  #取0-s
train_labels = all_img_labels[:s]
# 获取测试集
test_imgs = all_img_paths[s:]
test_labels = all_img_labels[s:]

# print(len(train_imgs))
# print(len(train_labels))
# print(len(test_imgs))
# print(len(test_labels))
# print(train_imgs[:20])
# print(train_labels[:20])


# 用from torchvision import transforms转换图片的大小

# transforms.ToTensor() 作用：
# 1、转化为一个tensor
# 2、转换到0-1之间（归一化）
# 3、会把，channel，放到第一个维度上
# 标准化

# 数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 压缩图片
    #  数据增强
    transforms.RandomCrop(192),  # 随机裁剪
    transforms.RandomHorizontalFlip(),  # 水平翻转图片
    transforms.RandomRotation(0.2),  # 随机旋转0.2
    transforms.ColorJitter(brightness=0.5),  # 增强图片亮度
    transforms.ColorJitter(contrast=0.5),   # 增强图片对比度
    transforms.ToTensor(),  # 归一化
])
test_transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor()
])

# 明确每次取出图片的数量
class MyDataset(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        # 当给定index时，返回的是图片的对象，而不是路径，
        # 所以返回之前，先进行读取，然后进行转化
        img = self.imgs[index]
        label = self.labels[index]
        # 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
        # pil_img = Image.open(img).convert('RGBA')
        pil_img = Image.open(img)
        np_img = np.asarray(pil_img, dtype=np.uint8)
        # 把pil_img转换成ndarray，查看是几维度的，如果是3维度的就不进行填充
        data = self.transforms(pil_img)
        return data, label

    def __len__(self):
        return len(self.imgs)

# 将预测出的图片所对应的标签与蒙语对应
base_label_dir = os.path.join(current_dir, 'dataset', 'Labels50.txt')
id_to_class = {}    #key:value
with open(base_label_dir, "r", encoding='utf-8') as f:  # 设置文件对象
    # 按行读入
    line_str = f.readline()
    while line_str:
        # 去掉末尾的'\n'
        line_str = line_str[:-1]
        # 用split()分割table空格
        str_list = line_str.split('\t')
        id_to_class[int(str_list[0])] = str_list[1]
        line_str = f.readline()

# print(len(id_to_class))
# print(id_to_class)
# 批次大小，每次训练所用图片的个数
baches = 32

#  划分测试数据和验证数据
train_ds = MyDataset(train_imgs, train_labels, train_transform)
test_ds = MyDataset(test_imgs, test_labels, test_transform)
train_dl = DataLoader(train_ds, batch_size=baches, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=baches)

imgs_batch, labels_batch = next(iter(train_dl))
print(imgs_batch.shape)
print(labels_batch.shape)


# 绘制读入的照片，检查照片与标签对不对

# plt.figure(figsize=(12, 8))
# for i, (img, label) in enumerate(zip(imgs_batch[:6], labels_batch[:6])):
#     img = img.permute(1, 2, 0).numpy()
#     plt.subplot(2, 3, i + 1)
#     # plt.title(id_to_class.get(label.item()))
#     print(label.item(), end=" ")
#     print(id_to_class.get(label.item()+100))
#     plt.imshow(img)
# plt.show()


class VGGNet(nn.Module):
    # 声明
    def __init__(self):
        super(VGGNet, self).__init__()
        # 输入通道为4，第一层使用16个3x3的卷积核
        self.conv1 = nn.Conv2d(4, 16, 3)
        # 使用BN层：参数为卷积核的个数：即特征数 16   一批标准化：加快过拟合，抑制过拟合
        self.bn1 = nn.BatchNorm2d(16)
        # 输入通道为16，第二层使用32个3x3的卷积核
        self.conv2 = nn.Conv2d(16, 32, 3)
        # 使用BN层：参数为卷积核的个数：即特征数 32
        self.bn2 = nn.BatchNorm2d(32)
        # 输入通道为32，第二层使用64个3x3的卷积核
        self.conv3 = nn.Conv2d(32, 64, 3)
        # 使用BN层：参数为卷积核的个数：即特征数 64
        self.bn3 = nn.BatchNorm2d(64)
        # 输入通道为64，第二层使用128个5x5的卷积核
        self.conv4 = nn.Conv2d(64, 128, 5)
        # 使用BN层：参数为卷积核的个数：即特征数 64
        self.bn4 = nn.BatchNorm2d(128)
        # 使用Dropout层
        self.drop = nn.Dropout(0.5)
        # 使用2*2的Pool层
        self.pool = nn.MaxPool2d((2, 2))
        # 第一个全联接层
        # self.fc1 = nn.Linear(128 * 11 * 11, 1024)
        self.fc1 = nn.Linear(128 * 9 * 9, 1024)
        # 使用BN层:参数为输出特征的个数：即特征数 1024
        self.bn_f1 = nn.BatchNorm1d(1024)
        # 第二个全联接层
        self.fc2 = nn.Linear(1024, 51)
        pass
    # 使用
    def forward(self, x):
        x = F.relu(self.conv1(x))  # relu激活函数
        # print("xv1:", x.size())
        x = self.pool(x)
        # print("xp1:", x.size())
        x = self.bn1(x) # 改变特征的大小，不改变特征数量
        x = F.relu(self.conv2(x))
        # print("xv2:", x.size())
        x = self.pool(x)
        # print("xp2:", x.size())
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        # print("xv3:", x.size())
        x = self.pool(x)
        # print("xp3:", x.size())
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        # print("xv4:", x.size())
        x = self.pool(x)
        # print("xp4:", x.size())
        x = self.bn4(x)
        # print(x.size())
        x = x.view(-1, 128 * 9 * 9)
        # x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = F.relu(self.fc1(x))
        # print("xq1:", x.size())
        x = self.bn_f1(x)
        x = self.drop(x)
        x = self.fc2(x)
        # print("xq2:", x.size())
        return x

# 创建神经网络模型
model = VGGNet()

# tensorboard用法二、显示模型
# writer.add_graph()是添加模型（网络）的方法
# 第一个参数为模型
# input_to_model第二个参数，表示输入的图片的样子,我们输入一个批次的数据
# tensorboard保存模型
writer.add_graph(model, imgs_batch)  #imgs_batch一批图片的路径

# 将模型放到GPU上
# if torch.cuda.is_available():
#     model.to('cuda')

# 交叉熵损失
loss_fn = nn.CrossEntropyLoss()
# Adam优化器
optim = torch.optim.Adam(model.parameters(), lr=0.001)  #模型内所有参数model.parameters，学习率lr


# 通用训练函数
def fit(epoch, model, trainloader, testloader):  #第几次训练，模型，训练集数据，测试集数据
    correct = 0
    total = 0
    running_loss = 0
    # 使用model.train()来告诉模型，当前处于训练模式，Dropout生效
    model.train()
    for x, y in trainloader:   #
        # 将数据放到GPU上跑
        # if torch.cuda.is_available():
        #     x, y = x.to('cuda'), y.to('cuda')
        y = torch.as_tensor(y, dtype=torch.long)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        #清除梯度
        optim.zero_grad()
        #反向传播
        loss.backward()
        #开始梯度下降
        optim.step()
        with torch.no_grad():  # 停止计算梯度
            y_pred = torch.argmax(y_pred, dim=1) # 找出概率最大的列，转化为预测值
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
    # 每个样本的平均loss
    epoch_loss = running_loss / len(trainloader.dataset)
    # 每个样本的平均acc
    epoch_acc = correct / total
    # 用tensorboard动态的显示loss
    # 第一个值，train_loss，第二个值，对应那个epoch
    # 保存训练集的损失函数和准确率
    writer.add_scalar('train_loss', epoch_loss, epoch + 1)
    writer.add_scalar('train_acc', epoch_acc, epoch + 1)

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    # 使用model.eval()来告诉模型，当前处于预测模式，Dropout不生效
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            # if torch.cuda.is_available():
            #     x, y = x.to('cuda'), y.to('cuda')
            y = torch.as_tensor(y, dtype=torch.long)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    # 每个test样本的平均loss
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    # 每个test样本的平均acc
    epoch_test_acc = test_correct / test_total
    # 用tensorboard动态的显示loss
    # 保存测试集的损失函数和准确率
    writer.add_scalar('test_loss', epoch_test_loss, epoch + 1)
    writer.add_scalar('test_acc', epoch_test_acc, epoch + 1)

    print("epoch: ", epoch, " loss:  ", round(epoch_loss, 3),
          " accuracy:  ", round(epoch_acc, 3),
          " test_loss  ", round(epoch_test_loss, 3),
          " test_accuracy:  ", round(epoch_test_acc, 3)
          )
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc


# 训练函数保存最优参数
# 获得模型的初始参数
# 深拷贝
best_model_wts = copy.deepcopy(model.state_dict())  #model.state_dict()模型全部参数
best_acc = 0
# 训练次数
epochs = 30
#保存训练参数信息
train_loss = []
train_acc = []
test_loss = []
test_acc = []
for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch, model, train_dl, test_dl)
    if epoch_test_acc > best_acc:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = epoch_test_acc
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
# 绘图
plt.plot(range(1, epochs + 1), train_loss, label='train_loss')
plt.plot(range(1, epochs + 1), test_loss, label='test_loss')
plt.legend()
plt.show()

# 保存模型的参数
path = os.path.join(current_dir, 'model', 'model/my_model.pth')
torch.save(best_model_wts, path)


new_model = VGGNet()
saved_state_dict = torch.load(path)
new_model.load_state_dict(saved_state_dict)

# 重新测试一下
test_correct = 0
test_total = 0
test_running_loss = 0
# 设置为预测模式

new_model.eval()
with torch.no_grad():
    for x, y in test_dl:
        # if torch.cuda.is_available():
        #     x, y = x.to('cuda'), y.to('cuda')
        y = torch.as_tensor(y, dtype=torch.long)
        y_pred = new_model(x)
        loss = loss_fn(y_pred, y)
        y_pred = torch.argmax(y_pred, dim=1)
        test_correct += (y_pred == y).sum().item()
        test_total += y.size(0)
        test_running_loss += loss.item()

# 每个test样本的平均loss
epoch_test_loss = test_running_loss / len(test_dl.dataset)
# 每个test样本的平均acc
epoch_test_acc = test_correct / test_total
print(" test_accuracy:  ", round(epoch_test_acc, 3))

#  0.032  accuracy:   0.802  test_loss   0.036  test_accuracy:   0.756
#  test_accuracy:   0.756


#epoch:  22  loss:   0.005  accuracy:   0.954  test_loss   0.005  test_accuracy:   0.967