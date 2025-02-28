"""
前面设置了一些参数，主要是模型训练中需要的，都有注释。
函数：
main():主要是进行跳转，根据前面的参数mode进行训练模型或者跑模型。有四种选择，下面进行阐述。
1.train()：mode=1，设置mode为1，进行模型的训练，里面都是模型训练需要设置的参数，如学习率、训练轮次，最后保存模型
2.test_model(), 利用已经保存好的模型，进行计算mse，对给定的输入和输出，模型根据输入得到预测数据，之后和输出进行计算误差。
3.use_model(), 使用模型，只给输入数据，模型根据输入数据，得出预测的输出数据，但因为不导入输出数据，所以不会算mse误差。
4.look_data_label(), 查看mat文件的数据标签。根据这个结果来写labelx和labely

其他函数：
load_dataset(), 加载数据的函数。

class ResBlock(nn.Module):
class Network(nn.Module):
ResBlock和Network和定义的两个神经网络。
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import scipy
import os
from torchsummary import summary
import  h5py

mode = 2                                        #选择自己训练模型（1）、用模型跑mse（2）、用模型生成数据（3）, 查看mat数据标签（4）
Filter = 6                                     #过滤器F数量
intputfile = "./20dBinput.mat"      #输入文件x
outputfile = "./20dBoutput.mat"                 #输出文件y
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   #模型训练预测是用显卡（cuda）还是cpu
saved_model = "./small_model_F6.pth"                    #已经训练好的模型
labelx = "X_DL"                                 #输入文件的标签
labely = "Y_DL"                                 #输出文件的标签
output_model = "small_model_F"+ str(Filter) + ".pth"                      #自己训练保存的路径
image_train_losses = "small_model_F"+ str(Filter) + "_train_losses.png"
image_val_losses = "small_model_F"+ str(Filter) + "_val_losses.png"
criterion = nn.MSELoss()


def load_dataset(inputfile, outputfile):
    #读取文件
    with h5py.File(inputfile, 'r') as hf:
        data_x = torch.Tensor(hf[labelx][:]).to(device)
    with h5py.File(outputfile, 'r') as hf:
        data_y = torch.Tensor(hf[labely][:]).to(device)
    return data_x.permute(3, 2, 1, 0), data_y.permute(3, 2, 1, 0)

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, Filter, kernel_size=(3,3), stride=(1,1), padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')  # He initialization

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(Filter, Filter, kernel_size=(3,3), stride=(1,1), padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')  # He initialization



    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(2, Filter, kernel_size=(3,3), stride=(1,1), padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')  # He initialization
        self.resblocks = nn.ModuleList([
            ResBlock(Filter),
            ResBlock(Filter),
            ResBlock(Filter),
            ResBlock(Filter)
        ])
        self.conv2 = nn.Conv2d(Filter, Filter, kernel_size=(3,3), stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')  # He initialization

        self.upsample = nn.Upsample(size=(512, 280), mode='bilinear', align_corners=False)
        self.regression = nn.Identity()
        self.conv3 = nn.Conv2d(Filter, 2, kernel_size=(3,3), stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')  # He initialization


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x1
        for i, resblock in enumerate(self.resblocks):
            if i > 0:
                x2 += x1
            x2 = resblock(x2)
        x3 = self.conv2(x2) + x1
        x4 = self.upsample(x3)
        x5 = self.conv3(x4)
        out = self.regression(x5)
        return out

def main():
    if mode == 1:
        train()
    elif mode == 2:
        test_model(saved_model=saved_model)
    elif mode == 3:
        use_model(saved_model=saved_model)
    elif mode == 4:
        look_data_label()
    else:
        print("请选择正确的mode，自己训练模型（1）、用模型跑mse（2）、用模型生成数据（3）,查看mat数据标签（4）")
    
def train():
    #读取文件
    data_x, data_y = load_dataset(intputfile, outputfile)
    
    length = int(len(data_x) * 0.8)
    data_x = data_x.permute(0, 3, 1, 2)
    data_y = data_y.permute(0, 3, 1, 2)

    #把数据导入到需要的格式

    train_data = torch.utils.data.TensorDataset(data_x[:length], data_y[:length])

    validation_data = torch.utils.data.TensorDataset(data_x[length:], data_y[length:])
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=8, shuffle=False)
    
    # 开始初始化模型
    model = Network()
    model.to(device)
    criterion = nn.MSELoss()
    #优化器的一些配置，lr是学习率
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    #调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
   
    #训练的轮次，一般200就可以了
    num_epochs = 200
    
    #记录的训练集损失
    train_losses = []
    #记录的验证集的损失
    validation_losses = []
    
    #开始200轮寻来你
    for epoch in range(num_epochs):
        print("epoch",epoch)
        # Training
        model.train()
        train_loss = 0
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 进行计算损失值
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for x_validation, y_validation in validation_loader:
                outputs = model(x_validation)
                validation_loss += loss.item()
        validation_loss /= len(validation_loader)
        validation_losses.append(validation_loss)

        print('Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}'
            .format(epoch+1, num_epochs, train_loss, validation_loss))
        scheduler.step(validation_loss)

    # 保存模型
    torch.save(model.state_dict(), output_model)

    # 画出图片
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.legend()
    plt.savefig(image_train_losses)
    plt.plot(validation_losses, label='Validation Loss')
    plt.legend()
    plt.savefig(image_val_losses)

def test_model(saved_model):
    #读取文件
    data_x, data_y = load_dataset(intputfile, outputfile)

    data_x = data_x.permute(0, 3, 1, 2)
    data_y = data_y.permute(0, 3, 1, 2)
    data = torch.utils.data.TensorDataset(data_x, data_y)
    data_loader = DataLoader(data, batch_size=8, shuffle=False)
    
    #导入已经训练好的模型
    model = Network()
    model.load_state_dict(torch.load(saved_model, map_location=device))
    model.to(device)
    model.eval()
    
    #计算误差mse
    validation_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x_validation, y_validation in data_loader:
            output = model(x_validation)
            loss = criterion(output, y_validation)
            validation_loss += loss.item()
    validation_loss /= len(data_loader)
    print('Validation Loss: {:.4f}'.format(validation_loss))

    #输出模型的信息
    summary(model, input_size=data_x.shape[1:])
    
def use_model(saved_model):
    #读取文件
    data_x = torch.Tensor(scipy.io.loadmat(intputfile)[labelx]).to(device)
    data_x = data_x.permute(0, 3, 1, 2)
    data_y = torch.zeros_like(data_x)
    data = torch.utils.data.TensorDataset(data_x, data_y)
    data_loader = DataLoader(data, batch_size=8, shuffle=False)

    #导入训练好的模型
    model = Network()
    model.to(device)
    model.load_state_dict(torch.load(saved_model, map_location=device))
    model.eval()
    
    #用训练好的模型计算预测的数据
    ans = None
    with torch.no_grad():
        for x_validation, y_validation in data_loader:
            if ans is None:
                ans = model(x_validation)
            else:
                ans = torch.cat((ans, model(x_validation)))
    ans = ans.permute(0,2,3,1)

    # 将得出的数据写入到outputfile文件中
    with h5py.File(outputfile, 'w') as f:
        f.create_dataset(labely, data = ans)
    
    summary(model, input_size=data_x.shape[1:])
        
def look_data_label():
    if os.path.exists(intputfile):
        with h5py.File(intputfile, 'r') as hf:
            print(hf.keys())
    else:
        print("输入文件不存在！")
    if os.path.exists(outputfile):
        with h5py.File(outputfile, 'r') as hf:
            print(hf.keys())
    else:
        print("输出文件不存在！")

if __name__ == '__main__':
    main()