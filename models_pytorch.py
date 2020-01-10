import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method {} is not implemented in pytorch'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialized network with {} initialization'.format(init_type))
    net.apply(init_func)
    
    
def save_model(model,optimizer,name,scheduler=None):
    if scheduler==None:
        checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    else:
        checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()}

    torch.save(checkpoint,name)
    

def load_model(filename,model,optimizer=None,scheduler=None):
    checkpoint=torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    print("Done loading")
    if  optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(optimizer.state_dict()['param_groups'][-1]['lr'],' : Learning rate')
    if  scheduler:
        scheduler.load_state_dict(checkpoint['optimizer'])
        print(scheduler.state_dict()['param_groups'][-1]['lr'],' : Learning rate')
        
        
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.up = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x



    
class ASM(nn.Module):
    def __init__(self,F_ip,F_int):
        super().__init__()
#         self.GlobalPool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.W_1x1 = nn.Conv2d(F_ip, F_int, kernel_size=1,stride=1,padding=0,bias=True)
        self.batch_norm=nn.BatchNorm2d(F_int)
      

    def forward(self,map_1_fm,map_2_fm):
        
        x=torch.cat((map_1_fm,map_2_fm),dim=1)
#         psi=self.GlobalPool(x)
        x1=self.W_1x1(x)
        psi=torch.sigmoid(x1)
        psi=self.batch_norm(psi)
        

        return x1*psi
    
    
class FFM(nn.Module):
    
    def __init__(self,F_int):
        super().__init__()
        self.W_x = nn.Sequential(
        nn.Conv2d(F_int, F_int, kernel_size=3,stride=1,padding=1,bias=True),
        nn.BatchNorm2d(F_int),
        nn.ReLU(inplace=True)
        )
        self.GlobalPool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.W_1x1 = nn.Conv2d(F_int, F_int, kernel_size=1,stride=1,padding=0,bias=True)
        self.batch_norm=nn.BatchNorm2d(F_int)
      

    def forward(self,map_1_fm,map_2_fm):
        
        x=torch.cat((map_1_fm,map_2_fm),dim=1)
        x=self.W_x(x)
        x=self.GlobalPool(x)
        psi=self.W_1x1(x)
        psi=torch.sigmoid(psi)
        
        x1=x*psi
        

        return x1+x
    
    
class DualEncoding_U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1_encoding_1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2_encoding_1 = conv_block(ch_in=64,ch_out=128)
        self.Conv3_encoding_1 = conv_block(ch_in=128,ch_out=256)
        self.Conv4_encoding_1 = conv_block(ch_in=256,ch_out=512)
        
        
        
        self.Conv1_encoding_2 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2_encoding_2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3_encoding_2 = conv_block(ch_in=128,ch_out=256)
        self.Conv4_encoding_2 = conv_block(ch_in=256,ch_out=512)
        
        self.ffm=FFM(1024)
        
        self.asm4=ASM(128,64)
        self.asm3=ASM(256,128)
        self.asm2=ASM(512,256)
        self.asm1=ASM(1024,512)
      
        self.dropout=nn.Dropout(0.45)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = nn.Sequential(nn.Conv2d(1024,512,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True))

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = nn.Sequential(nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = nn.Sequential(nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True))

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = nn.Sequential(nn.Conv2d(128,64,kernel_size=1,stride=1,padding=0,bias=True),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x_h,x_e):
        # encoding path
        x_h1 = self.Conv1_encoding_1(x_h)
        # N*64*512*512

        x_h2 = self.Maxpool(x_h1)
        x_h2 = self.Conv2_encoding_1(x_h2)
        # N*128*256*256

        x_h3 = self.Maxpool(x_h2)
        x_h3 = self.Conv3_encoding_1(x_h3)
        # N*256*128*128

        x_h4 = self.Maxpool(x_h3)
        x_h4 = self.Conv4_encoding_1(x_h4)
        # N*512*64*64
        
        x_e1 = self.Conv1_encoding_2(x_e)
        # N*64*512*512

        x_e2 = self.Maxpool(x_e1)
        x_e2 = self.Conv2_encoding_2(x_e2)
        # N*128*256*256

        x_e3 = self.Maxpool(x_e2)
        x_e3 = self.Conv3_encoding_2(x_e3)
        # N*256*128*128

        x_e4 = self.Maxpool(x_e3)
        x_e4 = self.Conv4_encoding_2(x_e4)
        # N*512*64*64
    
        x5=self.ffm(x_h4,x_e4)
        # N*1024*32*32
      
        
        # decoding + concat path
        
        d5 = self.Up5(x5)
        # N*512*64*64
        
        x4=self.asm1(x_h4,x_e4)
       
        # N*512*64*64
        d5 = torch.cat((x4,d5),dim=1)
        # N*1024*64*64
        d5 = self.Up_conv5(d5)
        # N*512*64*64
        d4 = self.Up4(d5)
        # N*256*128*128
        x3=self.asm2(x_h3,x_e3)
        # N*256*128*128
        d4 = torch.cat((x3,d4),dim=1)
        # N*512*128*128
        d4 = self.Up_conv4(d4)
        # N*256*128*128

        d3 = self.Up3(d4)
        # N*128*256*256
        x2=self.asm3(x_h2,x_e2)
        # N*128*256*256
        
        d3 = torch.cat((x2,d3),dim=1)
        # N*256*256*256
        d3 = self.Up_conv3(d3)
        # N*128*256*256

        d2 = self.Up2(d3)
        # N*64*512*512
        x1=self.asm4(x_h1,x_e1)
        # N*64*512*512
        d2 = torch.cat((x1,d2),dim=1)
        # N*128*512*512
        d2 = self.Up_conv2(d2)
        # N*64*512*512

        d1 = self.Conv_1x1(d2)
        # N*1*512*512

        return d1

    

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super().__init__()
        self.W_g = nn.Sequential(
        nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
        nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
        nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
        nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
        nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
        nn.BatchNorm2d(1),
        nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class AttnUNet(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,dropout=0.45):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout=torch.nn.Dropout(dropout)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5=self.dropout(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

    

