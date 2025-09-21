import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import sys

# 模型定义
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        # 输入通道为3，第一层使用16个3x3的卷积核
        self.conv1 = nn.Conv2d(4, 16, 3)
        # 使用BN层：参数为卷积核的个数：即特征数 16
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
        # 使用BN 层：参数为卷积核的个数：即特征数 64
        self.bn4 = nn.BatchNorm2d(128)
        # 使用Dropout层
        self.drop = nn.Dropout(0.5)
        # 使用2*2的Pool层
        self.pool = nn.MaxPool2d((2, 2))
        # 第一个全联接层
        self.fc1 = nn.Linear(128 * 9 * 9, 1024)
        # 使用BN层:参数为输出特征的个数：即特征数 1024
        self.bn_f1 = nn.BatchNorm1d(1024)
        # 第二个全联接层
        self.fc2 = nn.Linear(1024, 51)
        pass

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.bn4(x)
        x = x.view(-1, 128 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = self.bn_f1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

# 图像预处理和预测函数
def predict_image(image_path, torchvision=None, transforms=None):
    # 定义图片转换
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor()
    ])
    
    # 打开并处理图片
    try:
        pil_img = Image.open(image_path)
        img_tensor = transform(pil_img)
        # 添加batch维度
        img_tensor_batch = torch.unsqueeze(img_tensor, 0)
        
        # 模型路径 - 现在直接位于根目录下
        model_path = os.path.join('model', 'my_model.pth')
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"错误：模型文件 '{model_path}' 不存在")
            return None
        
        # 创建模型并加载参数
        model = VGGNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        # 进行预测
        with torch.no_grad():
            pred = model(img_tensor_batch)
            y_pred = torch.argmax(pred, dim=1)
        
        # 加载标签 - 现在直接位于根目录下
        labels_path = os.path.join('dataset', 'Labels50.txt')
        
        # 检查标签文件是否存在
        if not os.path.exists(labels_path):
            print(f"错误：标签文件 '{labels_path}' 不存在")
            return None
        
        # 创建id_to_class字典
        id_to_class = {}
        with open(labels_path, "r", encoding='utf-8') as f:
            line_str = f.readline()
            while line_str:
                line_str = line_str.strip()
                if line_str:
                    str_list = line_str.split('\t')
                    id_to_class[int(str_list[0])] = str_list[1]
                line_str = f.readline()
        
        # 返回预测结果
        predicted_class = y_pred.item() + 100
        predicted_char = id_to_class.get(predicted_class, "未知字符")
        return predicted_char
        
    except Exception as e:
        print(f"处理图片时出错：{e}")
        return None

# 主函数
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 从命令行参数获取图片路径
        image_path = sys.argv[1]
    else:
        # 默认使用示例图片（如果存在）
        example_dir = os.path.join('dataset', 'handwritten_50', '100')
        if os.path.exists(example_dir):
            example_images = [f for f in os.listdir(example_dir) if f.endswith('.png')]
            if example_images:
                image_path = os.path.join(example_dir, example_images[0])
                print(f"未提供图片路径，使用示例图片：{image_path}")
            else:
                print("错误：未找到示例图片")
                sys.exit(1)
        else:
            print("错误：未找到示例图片目录")
            sys.exit(1)
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：图片文件 '{image_path}' 不存在")
        sys.exit(1)
    
    # 进行预测
    result = predict_image(image_path)
    if result:
        print(f"预测结果：{result}")
    else:
        print("预测失败")
