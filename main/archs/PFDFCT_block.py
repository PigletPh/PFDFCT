import torch.nn as nn
import torch.nn.functional as F #函数，由def function ()定义，是一个固定的运算公式
# from basicsr.archs import SGBlock,FNet,Spartial_Attention,SwinT
from basicsr.archs import SwinT
import functools
# from basicsr.archs import block as B
import torch



#定义卷积层 输入通道 输出通道 卷积核尺寸 步长 扩展 组
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation #计算padding
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)
def conv_layer2(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    return nn.Sequential(#400epoch 32.726 28.623
        nn.Conv2d(in_channels, int(in_channels * 0.5), 1, stride, bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5 * 0.5), 1, 1, bias=True),
        nn.Conv2d(int(in_channels * 0.5 * 0.5), int(in_channels * 0.5), (1, 3), 1, (0, 1),
                           bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5), (3, 1), 1, (1, 0), bias=True),
        nn.Conv2d(int(in_channels * 0.5), out_channels, 1, 1, bias=True)
    )


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)

#定义激活函数 类型
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

#融合特征模块
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)  #Conv2d(40, 12, kernel_size=(1, 1), stride=(1, 1))
        self.conv_f = conv(f, f, kernel_size=1) #Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1))
        self.conv_max = conv(f, f, kernel_size=3, padding=1) #Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0) #Conv2d(12, 12, kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = conv(f, f, kernel_size=3, padding=1) #Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_ = conv(f, f, kernel_size=3, padding=1) #Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = conv(f, n_feats, kernel_size=1) #Conv2d(12, 40, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid = nn.Sigmoid() #Sigmoid()
        self.relu = nn.ReLU(inplace=True) #ReLU(inplace=True)

    def forward(self, x): #输入特性x
        c1_ = (self.conv1(x)) #x通过1×1卷积提取特征
        c1 = self.conv2(c1_) #Conv(S=2) 步长为2的卷积层
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3) #最大池化
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) #双线性插值
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf) #逐元素相加
        m = self.sigmoid(c4) #Sigmoid激活函数

        return x * m #返回x乘m

#中间级联的块（模型中的子块）
class FEB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(FEB, self).__init__()
        self.rc = self.remaining_channels = in_channels  #剩余通道=输入通道
        # self.c = B.conv_block(in_channels, in_channels, kernel_size=1, act_type='lrelu')  # 激活函数
        self.lcf = LCF(num_feat=50, compress_ratio=3, squeeze_factor=30)  # LCF 局部特征提取
        self.swinT = SwinT.SwinT() #深度为2的Transformer层
        self.c1_r = conv_layer(in_channels, self.rc, 3)  # 输入通道数 剩余通道数 卷积核为3
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        shortcut = input ###加上残差结构
        # local context 局部特征提取
        LCF = self.lcf(input) #局部特征提取
        input = self.swinT(LCF)   #通过Swin Transformer Layer 深度为2
        input = input + shortcut #相加之后的结果
        out_fused = self.esa(self.c1_r(input)) #融合模块
        # out_fused = self.c1_r(out_fused)  # 融合模块
        return out_fused



#双三次上采样
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride) #Conv2d(50, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    pixel_shuffle = nn.PixelShuffle(upscale_factor) #PixelShuffle(upscale_factor=4)
    return sequential(conv, pixel_shuffle)


# 局部特征提取
class ECAM(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ECAM, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  #全局平均池化
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),  #卷积1×1
            nn.ReLU(inplace=True),  #激活函数
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),  #卷积1×1
            nn.Sigmoid()) #激活函数
    def forward(self, x):
        y = self.attention(x)
        return x * y  #通道注意力作用到特征图上  变形版注意力

# 局部特征提取块
class LCF(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(LCF, self).__init__()
        #局部特征提取
        self.lcf = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),  #2d卷积 3×3
            nn.GELU(),  #激活函数
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),  #2d卷积 3×3
            ECAM(num_feat, squeeze_factor) #增强注意力块
            )
    def forward(self, x):
        return self.lcf(x)


class CALayer(nn.Module):
    def __init__(self, num_fea):
        super(CALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_fea, num_fea // 8, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_fea // 8, num_fea, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, fea):
        return self.conv_du(fea)

##  DCAB 模块
class DCABlock(nn.Module):
    def __init__(self, num_fea):
        super(DCABlock, self).__init__()
        self.channel1=num_fea//2
        self.channel2=num_fea-self.channel1
        self.convblock = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
        )
        self.A_att_conv = CALayer(self.channel1)
        self.B_att_conv = CALayer(self.channel2)

        self.fuse1 = nn.Conv2d(num_fea, self.channel1, 1, 1, 0)
        self.fuse2 = nn.Conv2d(num_fea, self.channel2, 1, 1, 0)
        self.fuse = nn.Conv2d(num_fea, num_fea, 1, 1, 0)

    def forward(self, x):
        x1,x2=torch.split(x,[self.channel1,self.channel2],dim=1)

        x1 = self.convblock(x1)

        A = self.A_att_conv(x1)
        P = torch.cat((x2, A*x1),dim=1)

        B = self.B_att_conv(x2)
        Q = torch.cat((x1, B*x2),dim=1)

        c=torch.cat((self.fuse1(P),self.fuse2(Q)),dim=1)
        out=self.fuse(c)
        return out

#attention fuse
class AF(nn.Module):
    def __init__(self, num_fea):
        super(AF, self).__init__()
        self.CA1=CALayer(num_fea)
        self.CA2=CALayer(num_fea)
        self.fuse=nn.Conv2d(num_fea*2,num_fea,1)
    def forward(self,x1,x2):
        x1=self.CA1(x1)*x1
        x2=self.CA2(x2)*x2
        return self.fuse(torch.cat((x1,x2),dim=1))

####  PN 模块
ls=[]
class PN(nn.Module):
    def __init__(self, num_fea):
        super(PN, self).__init__()
        self.CB1=DCABlock(num_fea)
        self.CB2=DCABlock(num_fea)
        self.CB3=DCABlock(num_fea)
        self.AF1=AF(num_fea)
        self.AF2=AF(num_fea)
    def forward(self,x):
        x1=self.CB1(x)
        x2=self.CB2(x1)
        x3=self.CB3(x2)
        f1=self.AF1(x3,x2)
        f2=self.AF2(f1,x1)
        ls.append(torch.mean(x1[0].data,dim=0))
        ls.append(torch.mean(x2[0].data,dim=0))
        ls.append(torch.mean(x3[0].data,dim=0))
        ls.append(torch.mean(f1[0].data,dim=0))
        ls.append(torch.mean(f2[0].data,dim=0))
        ls.append(torch.mean((x+f2)[0].data,dim=0))
        return x+f2


 
