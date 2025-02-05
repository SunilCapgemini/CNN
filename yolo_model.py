import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class YOLOv3:
    def __init__(self, input_shape=(416, 416, 3), num_classes=80):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def conv_block(self, x, filters, kernel_size, strides=1):
        """Convolution block with batch normalization and leaky ReLU"""
        x = layers.Conv2D(filters, kernel_size, strides=strides, 
                         padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x
    
    def residual_block(self, x, filters):
        """Residual block with two convolutions and a skip connection"""
        shortcut = x
        x = self.conv_block(x, filters // 2, 1)
        x = self.conv_block(x, filters, 3)
        return layers.Add()([shortcut, x])
    
    def darknet53(self, x):
        """Darknet-53 feature extractor"""
        # Initial convolution
        x = self.conv_block(x, 32, 3)
        
        # Downsample 1: 416 -> 208
        x = self.conv_block(x, 64, 3, strides=2)
        x = self.residual_block(x, 64)
        
        # Downsample 2: 208 -> 104
        x = self.conv_block(x, 128, 3, strides=2)
        for _ in range(2):
            x = self.residual_block(x, 128)
            
        # Downsample 3: 104 -> 52
        x = self.conv_block(x, 256, 3, strides=2)
        for _ in range(8):
            x = self.residual_block(x, 256)
        skip_52 = x
        
        # Downsample 4: 52 -> 26
        x = self.conv_block(x, 512, 3, strides=2)
        for _ in range(8):
            x = self.residual_block(x, 512)
        skip_26 = x
        
        # Downsample 5: 26 -> 13
        x = self.conv_block(x, 1024, 3, strides=2)
        for _ in range(4):
            x = self.residual_block(x, 1024)
            
        return x, skip_26, skip_52
    
    def yolo_head(self, x, filters):
        """YOLO detection head"""
        x = self.conv_block(x, filters, 1)
        x = self.conv_block(x, filters * 2, 3)
        x = self.conv_block(x, filters, 1)
        x = self.conv_block(x, filters * 2, 3)
        x = self.conv_block(x, filters, 1)
        
        return x
    
    def detection_layer(self, x, num_anchors=3):
        """Detection layer for object detection"""
        return layers.Conv2D(num_anchors * (5 + self.num_classes), 1,
                           padding='same')(x)
    
    def build_model(self):
        """Build the complete YOLOv3 model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Darknet-53 backbone
        x, skip_26, skip_52 = self.darknet53(inputs)
        
        # First detection head (13x13 grid)
        x = self.yolo_head(x, 512)
        output_13 = self.detection_layer(x)
        
        # Second detection head (26x26 grid)
        x = self.conv_block(x, 256, 1)
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skip_26])
        x = self.yolo_head(x, 256)
        output_26 = self.detection_layer(x)
        
        # Third detection head (52x52 grid)
        x = self.conv_block(x, 128, 1)
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skip_52])
        x = self.yolo_head(x, 128)
        output_52 = self.detection_layer(x)
        
        return Model(inputs, [output_13, output_26, output_52])

if __name__ == "__main__":
    # Example usage
    yolo = YOLOv3()
    model = yolo.build_model()
    model.summary()
