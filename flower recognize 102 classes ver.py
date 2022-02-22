import os
import matplotlib.pyplot as plt
#%matplotlib inline              # 在jupyter notebook中显示图片，不需要显示图片的时候可以注释掉
import numpy as np              # numpy模块
import torch                    # 导入pytorch  
from torch import nn            # 导入pytorch的nn模块
import torch.optim as optim     # 导入pytorch的优化器模块
import torchvision              # 导入pytorch的图像处理模块
#if torchvision uninstalld, please run: pip install torchvision
from torchvision import datasets, models, transforms    # 导入pytorch的数据集模块
#pytorch web: https://pytorch.org/docs/stable/torchvision/index.html
import imageio                  # 图片读取
import time                     # 计时
import warnings                 # 警告
import random                   # 随机数
import sys                      # 系统
import copy                     # 深拷贝
import json                     # json
from PIL import Image           # 图片处理

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"       # 关闭环境变量警告,删除后会提示OMP警告

#导入数据集
data_dir = 'C:/Users\JasonFeng\Desktop\CNN Framework\Flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
#test_dir = data_dir + '/test'  # 如果没有测试集，可以注释掉

#数据预处理
    #图片预处理
data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(45), # 随机旋转
            transforms.RandomResizedCrop(224),                 # 随机裁剪
            transforms.RandomHorizontalFlip(p = 0.5),          # 随机水平翻转
            transforms.RandomVerticalFlip(p = 0.5),            # 随机垂直翻转
            transforms.ColorJitter(brightness = 0.2, contrast = 0.1, saturation = 0.1, hue = 0.1),   # 随机调整图片亮度,对比度,饱和度,色调
            transforms.RandomGrayscale(p = 0.025),               # 随机灰度
            transforms.ToTensor(),                              # 将图片转换为Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化,均值，标准差
    ]),
    'valid': transforms.Compose([transforms.Resize(256),    # 缩放
            transforms.CenterCrop(224),                     # 中心裁剪
            transforms.ToTensor(),                          # 将图片转换为Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化,均值，标准差
    ]),
}

#批量大小
batch_size = 8  # 批量大小,训练时候的batch大小,显存不够时候可以调小

#训练集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}   # 创建数据集
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8,
                                            pin_memory = True) for x in ['train', 'valid']}   # 创建数据加载器
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}   # 数据集大小
class_names = image_datasets['train'].classes   # 类别名称

#print(image_datasets)   # 打印数据集,可以看到有两个数据集,不需要的时候可以注释掉

#print(dataloaders)   # 打印数据加载器,可以看到有两个数据加载器,不需要的时候可以注释掉

#print(dataset_sizes)   # 打印数据集大小,可以看到有两个数据集大小,不需要的时候可以注释掉

#多线程的原因，这里加上补充代码
#读取标签对应的类别名称
with open('C:/Users\JasonFeng\Desktop\python\CV\Flower-Recognition-master\cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
#简单测试
#print(cat_to_name)   # 打印标签对应的类别名称,可以看到有102个类别,不需要的时候可以注释掉

#展示数据
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()    # 将tensor转换为cpu,并且复制一份,然后取消关联
    image = image.numpy().squeeze()              # 将tensor转换为numpy,并且去掉维度
    image = image.transpose(1, 2, 0)             # 转换维度
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))   # 归一化
    image = image.clip(0, 1)                     # 限制像素值在0-1之间

    return image                                 # 返回图片

#
fig = plt.figure(figsize = (20, 12))             # 创建一个图像
columns = 5                                      # 列数
rows = 3                                         # 行数

dataiter = iter(dataloaders['valid'])            # 创建一个迭代器
inputs, classes = dataiter.next()                # 获取迭代器中的数据

# 出bug了，回看
# for idx in range (columns * rows):
#     ax = fig.add_subplot(rows, columns, idx , xticks = [], yticks = [])   # 创建子图
#     ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])           # 展示标题
#     plt.imshow(im_convert(inputs[idx]))                                      # 展示图片
# plt.show()                                                

#加载模型
model_name = 'resnet'
feature_extract = True

#是否使用GPU
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('CUDA is available.  Training on GPU ...')
else:
    print('CUDA is not available!  Training on CPU ...')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

#迁移学习
def set_parameter_requires_grad(model, feature_extracting):  # 参数是否需要更新
    if feature_extracting:                                   # 如果是特征提取
        for param in model.parameters():                     # 将所有参数设置为不需要更新
            param.requires_grad = False                      # 参数不需要更新

model_ft = models.resnet152()                                # 加载模型
#print(model_ft)                                             # 打印模型

def initialize_model ( model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，并且初始化，不同的模型有不同的初始化方式
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        model_ft = models.resnet18(pretrained=use_pretrained)           # 加载模型
        set_parameter_requires_grad(model_ft, feature_extract)      # 参数是否需要更新
        num_ftrs = model_ft.fc.in_features                          # 获取输出维度
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102),
                                    nn.LogSoftmax(dim=1))           # 将输出转换为类别
        input_size = 224                                            # 输入大小

    elif model_name == "alexnet":                                   # 如果是AlexNet
        """ Alexnet 
        """
        model_ft = models.alexnet(pretrained=use_pretrained)        # 加载模型
        set_parameter_requires_grad(model_ft, feature_extract)      # 参数是否需要更新
        num_ftrs = model_ft.classifier[6].in_features               # 获取输出维度
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)   # 将输出转换为类别
        input_size = 224                                            # 输入大小

    elif model_name == "vgg":                                       # 如果是VGG
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)       # 加载模型
        set_parameter_requires_grad(model_ft, feature_extract)      # 参数是否需要更新
        num_ftrs = model_ft.classifier[6].in_features               # 获取输出维度
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)   # 将输出转换为类别
        input_size = 224                                            # 输入大小

    elif model_name == "squeezenet":                                # 如果是SqueezeNet
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)  # 加载模型
        set_parameter_requires_grad(model_ft, feature_extract)      # 参数是否需要更新
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1)) # 将输出转换为类别
        model_ft.num_classes = num_classes                          # 类别数
        input_size = 224                                            # 输入大小

    elif model_name == "densenet":                                  # 如果是DenseNet
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)    # 加载模型
        set_parameter_requires_grad(model_ft, feature_extract)      # 参数是否需要更新
        num_ftrs = model_ft.classifier.in_features                  # 获取输出维度
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)      # 将输出转换为类别
        input_size = 224                                            # 输入大小

    elif model_name == "inception":                                  # 如果是Inception
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)   # 加载模型
        set_parameter_requires_grad(model_ft, feature_extract)      # 参数是否需要更新
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features                # 获取输出维度
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)    # 将输出转换为类别
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features                          # 获取输出维度
        model_ft.fc = nn.Linear(num_ftrs, num_classes)              # 将输出转换为类别
        input_size = 299                                            # 输入大小

    else:                                                           # 如果是其他模型
        print("Invalid model name, exiting...")                     # 打印错误信息
        exit()                                                      # 退出

    return model_ft, input_size                                     # 返回模型和输入大小

# 加载模型
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)  # 初始化模型，获取模型和输入大小，种类多少，是否使用预训练模型

#GPU计算
model_ft = model_ft.to(device)                                                                          # 将模型加载到GPU

#模型保存
filename = 'checkpoint.pth'                                                                             # 模型保存路径

#是否训练所有层
params_to_update = model_ft.parameters()                            # 获取参数
print("Params to learn:")                                           # 打印参数
if feature_extract:                                                 # 如果是特征提取
    params_to_update = []                                           # 则参数为空
    for name,param in model_ft.named_parameters():                  # 且遍历参数
        if param.requires_grad == True:                             # 如果参数需要更新
            params_to_update.append(param)                          # 则将参数加入列表
            print("\t",name)                                        # 并打印参数名称
else:                                                               # 如果不是特征提取
    for name,param in model_ft.named_parameters():                  # 则遍历参数
        if param.requires_grad == True:                             # 如果参数需要更新
            print("\t",name)                                        # 则打印参数名称

#可以在此处再打印一次模型，看看是否更新了需要训练的参数
#print(model_ft)

#优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)            # 创建优化器，自行设置学习率
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 创建学习率调整器，自行设置学习率调整步长和衰减率,step_size表示学习率调整的步长，gamma表示衰减率,默认7个epoch衰减为之前的0.1倍
#因为网络的最后一层设置了LogSoftmax，所以损失函数设置不能为交叉熵损失函数，而是NLLLoss
criterion = nn.NLLLoss()                                        # 创建损失函数

#训练模块
def train_model(model,dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename=filename):
    since = time.time()                                          # 获取当前时间
    best_acc = 0                                               # 初始化最佳准确率
    """
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    """
    model.to(device)                                             # 将模型加载到GPU

    val_acc_history = []                                    # 初始化验证集准确率列表
    train_acc_history = []                                  # 初始化训练集准确率列表
    train_losses = []                                       # 初始化训练集损失列表
    val_losses = []                                         # 初始化验证集损失列表
    LRs = [optimizer.param_groups[0]['lr']]                 # 初始化学习率列表

    best_model_wts = copy.deepcopy(model.state_dict())       # 初始化最佳模型参数

    for epoch in range(num_epochs):                         # 循环epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))  # 打印当前迭代次数
        print('-' * 10)                                      # 打印分隔符

        #训练与验证
        for phase in ['train', 'valid']:                       # 循环训练和验证
            if phase == 'train':                             # 如果是训练阶段
                model.train()                                 # 将模型设置为训练模式
            else:
                model.eval()                                  # 否则将模型设置为测试模式

            running_loss = 0.0                                # 初始化迭代损失
            running_corrects = 0                             # 初始化迭代正确数

            #循环所有数据
            for inputs, labels in dataloaders[phase]:        # 循环数据
                inputs = inputs.to(device)                   # 将数据加载到GPU
                labels = labels.to(device)                   # 将标签加载到GPU

                #梯度清零
                optimizer.zero_grad()                        # 将梯度清零
                #仅在训练阶段计算与更新梯度
                with torch.set_grad_enabled(phase == 'train'):      # 如果是训练阶段
                    if is_inception and phase == 'train':           # 如果是Inception模型，并且是训练阶段
                        outputs, aux_outputs = model(inputs)        # 获取输出和辅助输出
                        loss1 = criterion(outputs, labels)          # 计算输出损失
                        loss2 = criterion(aux_outputs, labels)      # 计算辅助输出损失
                        loss = loss1 + 0.4*loss2                  # 计算总损失
                    else:
                        #resnet执行此处
                        outputs = model(inputs)                     # 获取输出
                        loss = criterion(outputs, labels)           # 计算损失

                    _, preds = torch.max(outputs, 1)               # 获取预测类别

                        #训练阶段更新权重
                    if phase == 'train':
                        loss.backward()                          # 反向传播
                        optimizer.step()                         # 更新权重

                #计算迭代损失和正确数
                running_loss += loss.item() * inputs.size(0)          # 计算迭代损失
                running_corrects += torch.sum(preds == labels.data)     # 计算迭代正确数

            #计算平均损失和正确率
            epoch_loss = running_loss / len(dataloaders[phase].dataset) # 计算平均损失
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) # 计算平均正确率


            #记录时间
            time_elapsed = time.time() - since                        # 计算时间
            print('Time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印时间

            #记录损失和正确率
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) # 打印损失和正确率


            #得到最佳模型参数
            if phase == 'valid' and epoch_acc > best_acc:           # 如果是验证阶段，并且准确率大于最佳准确率
                best_acc = epoch_acc                                # 更新最佳准确率
                best_model_wts = copy.deepcopy(model.state_dict())  # 更新最佳模型参数
                state = {
                    'state_dict': model.state_dict(),             # 模型参数
                    'optimizer': optimizer.state_dict(),          # 参数优化器
                    'lr': optimizer.param_groups[0]['lr'],        # 学习率
                    'epoch': epoch,                               # 当前迭代次数
                    'best_acc': best_acc,                         # 最佳准确率
                    'time': time_elapsed                          # 时间
                }
                torch.save(state, filename) # 保存模型参数
            if phase == 'valid':                                # 如果是验证阶段
                val_acc_history.append(epoch_acc)               # 将验证准确率添加到验证准确率列表
                val_losses.append(epoch_loss)                   # 将验证损失添加到验证损失列表
                scheduler.step(epoch_loss)                      # 更新学习率
            if phase == 'train':
                train_acc_history.append(epoch_acc)             # 将训练准确率添加到训练准确率列表
                train_losses.append(epoch_loss)                 # 将训练损失添加到训练损失列表

        print('Optimizer learning rate: {:.7f}'.format(optimizer.param_groups[0]['lr'])) # 打印学习率
        LRs.append(optimizer.param_groups[0]['lr'])             # 将学习率添加到学习率列表
        print()

    time_elapsed = time.time() - since                                                          # 计算时间
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印时间
    print('Best val Acc: {:4f}'.format(best_acc))                                               # 打印最佳准确率

    # # 绘制训练和验证准确率曲线,可选
    # fig = plt.figure()                                                                        # 创建图表
    # plt.plot(LRs, train_acc_history, label='train')                                          # 绘制训练准确率曲线
    # plt.plot(LRs, val_acc_history, label='val')                                             # 绘制验证准确率曲线
    # plt.xlabel('Learning rate')                                                             # 横坐标名称
    # plt.ylabel('Accuracy')                                                                # 纵坐标名称
    # plt.title('Training and validation accuracy')                                        # 图表标题
    # plt.legend(loc='lower right')                                                       # 图例位置
    # plt.show()                                                                        # 显示图表

    # 训练完后，使用最佳模型作为最终结果    
    model.load_state_dict(best_model_wts)                                              # 加载最佳模型参数
    return model, val_acc_history, train_acc_history, train_losses, val_losses, LRs # 返回最佳模型，验证准确率，训练准确率，训练损失，验证损失，学习率

#开始训练
#若太慢，把epochs调小
#训练时，损失是否下降，准确率是否上升，训练与验证差距大不大？
#若差距大，就是模型过拟合
model_ft, val_acc_history, train_acc_history,valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,criterion, optimizer_ft, num_epochs=5, is_inception=(model_name=="inception"))  # 训练模型
#以上训的fc

###########################################################################################################
#如果要训练所有层，可以把下面的注释打开
# for param in model_ft.parameters():
#     param.requires_grad = True

# # 再继续训练所有的参数，学习率调小一点
# optimizer = optim.Adam(params_to_update, lr=1e-4)           # 创建优化器
# scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)     # 创建学习率调度器

# # 损失函数
# criterion = nn.NLLLoss()        

# # Load the checkpoint

# checkpoint = torch.load(filename)   
# best_acc = checkpoint['best_acc']   
# model_ft.load_state_dict(checkpoint['state_dict'])  
# optimizer.load_state_dict(checkpoint['optimizer'])  
# #model_ft.class_to_idx = checkpoint['mapping']  

# model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=10, is_inception=(model_name=="inception"))   

#测试网络效果
#加载训练好的模型
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)  # 初始化模型

# GPU模式
model_ft = model_ft.to(device)  # 将模型放到GPU上

# 保存文件的名字
filename='C:\\Users\JasonFeng\Desktop\python\CV\Flower-Recognition-master\seriouscheckpoint.pth'    #保存文件的名字

# 加载模型
checkpoint = torch.load(filename)   
best_acc = checkpoint['best_acc']   
model_ft.load_state_dict(checkpoint['state_dict'])  

def process_image(image_path):
    # 读取测试数据
    img = Image.open(image_path)
    # Resize,thumbnail方法只能进行缩小，所以进行了判断
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop操作
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,top_margin))
    # 相同的预处理方法
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # 注意颜色通道应该放在第一个位置
    img = img.transpose((2, 0, 1))
    
    return img

