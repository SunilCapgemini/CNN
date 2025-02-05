import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels//2, 1),
            ConvBlock(channels//2, channels, 3)
        )
        
    def forward(self, x):
        return x + self.block(x)

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        
        # Initial conv
        self.conv1 = ConvBlock(3, 32, 3)
        
        # Downsample 1: 416 -> 208
        self.conv2 = ConvBlock(32, 64, 3, stride=2)
        self.res1 = ResidualBlock(64)
        
        # Downsample 2: 208 -> 104
        self.conv3 = ConvBlock(64, 128, 3, stride=2)
        self.res2 = nn.ModuleList([ResidualBlock(128) for _ in range(2)])
        
        # Downsample 3: 104 -> 52
        self.conv4 = ConvBlock(128, 256, 3, stride=2)
        self.res3 = nn.ModuleList([ResidualBlock(256) for _ in range(8)])
        
        # Downsample 4: 52 -> 26
        self.conv5 = ConvBlock(256, 512, 3, stride=2)
        self.res4 = nn.ModuleList([ResidualBlock(512) for _ in range(8)])
        
        # Downsample 5: 26 -> 13
        self.conv6 = ConvBlock(512, 1024, 3, stride=2)
        self.res5 = nn.ModuleList([ResidualBlock(1024) for _ in range(4)])
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        
        x = self.conv3(x)
        for res in self.res2:
            x = res(x)
            
        x = self.conv4(x)
        for res in self.res3:
            x = res(x)
        skip_52 = x
        
        x = self.conv5(x)
        for res in self.res4:
            x = res(x)
        skip_26 = x
        
        x = self.conv6(x)
        for res in self.res5:
            x = res(x)
            
        return x, skip_26, skip_52

class YOLOHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YOLOHead, self).__init__()
        self.head = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            ConvBlock(out_channels, out_channels * 2, 3),
            ConvBlock(out_channels * 2, out_channels, 1),
            ConvBlock(out_channels, out_channels * 2, 3),
            ConvBlock(out_channels * 2, out_channels, 1)
        )
        
    def forward(self, x):
        return self.head(x)

class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        
        # Backbone
        self.darknet = Darknet53()
        
        # Detection heads
        self.head1 = YOLOHead(1024, 512)
        self.head2 = YOLOHead(1024, 256)  # 512 + 512 from skip connection
        self.head3 = YOLOHead(512, 128)   # 256 + 256 from skip connection
        
        # Detection layers (3 anchors * (5 + num_classes))
        # 5 = objectness + 4 box coordinates
        self.det1 = nn.Conv2d(512, 3 * (5 + num_classes), 1)
        self.det2 = nn.Conv2d(256, 3 * (5 + num_classes), 1)
        self.det3 = nn.Conv2d(128, 3 * (5 + num_classes), 1)
        
        # Additional conv layers for feature reduction before upsampling
        self.conv_before_upsample1 = ConvBlock(512, 256, 1)
        self.conv_before_upsample2 = ConvBlock(256, 128, 1)
        
    def forward(self, x):
        # Input shape: (batch_size, 3, 416, 416)
        x, skip_26, skip_52 = self.darknet(x)
        
        # First detection head (13x13 grid)
        head1 = self.head1(x)
        out1 = self.det1(head1)
        
        # Second detection head (26x26 grid)
        x = self.conv_before_upsample1(head1)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, skip_26], dim=1)  # 256 + 512 = 768 channels
        head2 = self.head2(x)
        out2 = self.det2(head2)
        
        # Third detection head (52x52 grid)
        x = self.conv_before_upsample2(head2)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, skip_52], dim=1)  # 128 + 256 = 384 channels
        head3 = self.head3(x)
        out3 = self.det3(head3)
        
        return out1, out2, out3

if __name__ == "__main__":
    # Example usage
    model = YOLOv3(num_classes=80)
    x = torch.randn(1, 3, 416, 416)  # Example input
    outputs = model(x)
    
    # Print output shapes
    for i, output in enumerate(outputs):
        print(f"Output {i+1} shape:", output.shape)

    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
