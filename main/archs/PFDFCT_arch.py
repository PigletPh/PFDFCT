import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs import PFDFCT_block as PFDFCT_block


#@ARCH_REGISTRY.register()  #这里是注册  单独调试代码的时候可以注释这行  但是整体运行的时候记得去掉注释
class PFDFCT(nn.Module):
    #主网络
    def __init__(self,
                 num_in_ch, #输入的通道数为3。
                 num_out_ch, #输出的通道数为3。
                 num_feat, #中间特征的通道数50。
                 upscale,
                 ): #图像平均值以RGB顺序表示。
        super(PFDFCT, self).__init__()


        nf = num_feat #中间特征的通道数。


        ###############  PN  部分  ####################
        self.PN=PFDFCT_block.PN(nf)  #边缘提取部分

        self.upscale = upscale
        self.fea_conv = PFDFCT_block.conv_layer(num_in_ch, nf, kernel_size=3) #输入通道为RGB3通道 输出特征为50通道中间特征通道数 3×3的卷积
        # num_modules = 5   #模块数量
        num_modules = 4  # 模块数量
        self.B1 = PFDFCT_block.FEB(in_channels=nf) #特征提取块
        self.B2 = PFDFCT_block.FEB(in_channels=nf) #特征提取块
        self.B3 = PFDFCT_block.FEB(in_channels=nf) #特征提取块
        self.B4 = PFDFCT_block.FEB(in_channels=nf) #特征提取块
        # self.B5 = LightSR_block.FEB(in_channels=nf) #特征提取块
        self.c = PFDFCT_block.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu') #激活函数
        self.LR_conv = PFDFCT_block.conv_layer(nf, nf, kernel_size=3) #Conv2d(50, 50, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        upsample_block = PFDFCT_block.pixelshuffle_block #双三次上采样模块
        self.upsampler = upsample_block(nf, num_out_ch, upscale_factor=upscale)  #上采样模块
        self.scale_idx = 0



    def forward(self, input):
        bi = F.interpolate(input, scale_factor=self.upscale, mode='bicubic', align_corners=False) #这里把FeNet的上采样加上去了

        out_fea = self.fea_conv(input) #3×3的卷积     提取浅层特征

        PN =self.PN(out_fea) #通过边缘提取网路得到边缘细节图

        out_B1 = self.B1(out_fea) #特征提取块
        out_B2 = self.B2(out_B1) #特征提取块
        out_B3 = self.B3(out_B2) #特征提取块
        out_B4 = self.B4(out_B3) #特征提取块


        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))  # concatenation 拼接
        out_lr = self.LR_conv(out_B) + out_fea #逐元素相加

        ########加上PM部分#############
        out_lr = out_lr + PN #通过特征提取块之后在加上边缘特征


        output = self.upsampler(out_lr) #Pixel上采样

        return output+bi #加上了增强

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

    # 模型参数计算
def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))


