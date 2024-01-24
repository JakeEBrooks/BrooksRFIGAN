import torch
import torch.nn as nn

class UNetConvBlock(nn.Module):
    def __init__(self, kernel_size, infilters, outfilters):
        super(UNetConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(infilters,outfilters,kernel_size,bias=False,padding='same'),
            nn.BatchNorm2d(outfilters),
            nn.ReLU(inplace=False),
            nn.Conv2d(outfilters,outfilters,kernel_size,bias=False,padding='same'),
            nn.BatchNorm2d(outfilters),
            nn.ReLU(inplace=False)
        )
    def forward(self, input):
        return self.main(input)

class UNetDecodeBlock(nn.Module):
    def __init__(self, kernel_size, infilters, outfilters):
        super(UNetDecodeBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(infilters,outfilters,kernel_size,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(outfilters),
            nn.ReLU(inplace=False)
        )
    def forward(self, input):
        return self.main(input)

class UNet(nn.Module):
    def __init__(self, kernel_size=3, convfilters=32, dropout_p=0.1):
        super(UNet, self).__init__()
        self.mpool = nn.MaxPool2d(2)

        self.e1 = UNetConvBlock(kernel_size, 1, convfilters)
        self.do1 = nn.Dropout(p=dropout_p,inplace=False)

        self.e2 = UNetConvBlock(kernel_size, convfilters, convfilters*2)
        self.do2 = nn.Dropout(p=dropout_p,inplace=False)

        self.e3 = UNetConvBlock(kernel_size, convfilters*2, convfilters*3)
        self.do3 = nn.Dropout(p=dropout_p,inplace=False)

        self.e4 = UNetConvBlock(kernel_size, convfilters*3, convfilters*4)
        self.do4 = nn.Dropout(p=dropout_p,inplace=False)

        self.bottleneck = UNetConvBlock(kernel_size, convfilters*4, convfilters*5)

        # Extra *2 is because the skip connections double the features
        self.d5 = UNetDecodeBlock(kernel_size, convfilters*5, convfilters*4)
        self.do5 = nn.Dropout(p=dropout_p,inplace=False)
        self.dc5 = UNetConvBlock(kernel_size, convfilters*4*2, convfilters*4)

        self.d6 = UNetDecodeBlock(kernel_size, convfilters*4, convfilters*3)
        self.do6 = nn.Dropout(p=dropout_p,inplace=False)
        self.dc6 = UNetConvBlock(kernel_size, convfilters*3*2, convfilters*3)

        self.d7 = UNetDecodeBlock(kernel_size, convfilters*3, convfilters*2)
        self.do7 = nn.Dropout(p=dropout_p,inplace=False)
        self.dc7 = UNetConvBlock(kernel_size, convfilters*2*2, convfilters*2)

        self.d8 = UNetDecodeBlock(kernel_size, convfilters*2, convfilters)
        self.do8 = nn.Dropout(p=dropout_p,inplace=False)
        self.dc8 = UNetConvBlock(kernel_size, convfilters*2, convfilters)
    
        self.outconv = nn.Conv2d(convfilters,1,1)
        self.outact = nn.Sigmoid()
    
    def forward(self, input):
        enc1 = self.e1(input)
        x = self.mpool(enc1)
        x = self.do1(x)

        enc2 = self.e2(x)
        x = self.mpool(enc2)
        x = self.do2(x)

        enc3 = self.e3(x)
        x = self.mpool(enc3)
        x = self.do3(x)

        enc4 = self.e4(x)
        x = self.mpool(enc4)
        x = self.do4(x)

        mid = self.bottleneck(x)

        dec5 = self.d5(mid)
        x = torch.cat((enc4,dec5),dim=1)
        x = self.do5(x)
        x = self.dc5(x)

        dec6 = self.d6(x)
        x = torch.cat((enc3,dec6),dim=1)
        x = self.do6(x)
        x = self.dc6(x)

        dec7 = self.d7(x)
        x = torch.cat((enc2,dec7),dim=1)
        x = self.do7(x)
        x = self.dc7(x)

        dec8 = self.d8(x)
        x = torch.cat((enc1,dec8),dim=1)
        x = self.do8(x)
        x = self.dc8(x)

        x = self.outconv(x)
        return self.outact(x)

class ConvNet(nn.Module):
    def __init__(self, kernel_size=3, convfilters=32, input_shape=(128,1024)):
        super(ConvNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1,convfilters,kernel_size,bias=False,stride=2,padding=1),
            nn.BatchNorm2d(convfilters),
            nn.ReLU(inplace=False),

            nn.Conv2d(convfilters,convfilters*2,kernel_size,bias=False,stride=2,padding=1),
            nn.BatchNorm2d(convfilters*2),
            nn.ReLU(inplace=False),

            nn.Conv2d(convfilters*2,convfilters*3,kernel_size,bias=False,stride=2,padding=1),
            nn.BatchNorm2d(convfilters*3),
            nn.ReLU(inplace=False),

            nn.Flatten(),
            nn.Linear(convfilters*3*(input_shape[0]//8)*(input_shape[1]//8),convfilters*3),
            nn.Linear(convfilters*3,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)