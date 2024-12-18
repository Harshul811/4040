from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D,
    Activation, Flatten, Dense, Add, Multiply, BatchNormalization, Dropout, Lambda, AveragePooling2D
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class ResidualAttentionNetwork:
    def __init__(self, input_shape, n_classes, activation='softmax', p=1, t=2, r=1, kernel_size=(3, 3), dropout_rate=0.5):
        """
        Initialize the Residual Attention Network.

        Args:
            input_shape (tuple): Shape of input images (H, W, C).
            n_classes (int): Number of output classes.
            activation (str): Activation function for the final layer.
            p (int): Number of residual units before and after the attention module.
            t (int): Number of residual units in the trunk branch.
            r (int): Number of residual units between pooling and upsampling layers in the mask branch.
            kernel_size (tuple): Kernel size for convolution layers.
            dropout_rate (float): Dropout rate for fully connected layers.
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.activation = activation
        self.p = p
        self.t = t
        self.r = r
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

    def build_model(self):
        """
        Build the Residual Attention Network model.
        """
        input_data = Input(shape=self.input_shape)

        # Initial padding for small input dimensions
        if self.input_shape[0] < 32 or self.input_shape[1] < 32:
            pad_x = (32 - self.input_shape[0]) // 2
            pad_y = (32 - self.input_shape[1]) // 2
            input_data = ZeroPadding2D(padding=(pad_x, pad_y))(input_data)

        # Initial Conv and Normalization
        x = Conv2D(32, kernel_size=self.kernel_size, strides=(1, 1), padding='same')(input_data)
        x = BatchNormalization()(x)
        
        #Downsampling residual unit (output = 16x16x64)
        x = self.residual_unit(x, 32, 64, stride=2)
        
        #Attention Stage 1 (stage 2 from original paper): 2 modules 
        x = self.attention_module_stage_1(x, input_chan=64, output_chan=64, size=(2,2))
        x = self.attention_module_stage_1(x, input_chan=64, output_chan=64, size=(2,2))
        
        #Downsampling residual unit (output = 8x8x128)
        x = self.residual_unit(x, input_chan=64, output_chan=128, stride=2)
        
        #Attention Stage 2 (stage 3 from original paper): 1 module
        x = self.attention_module_stage_2(x, input_chan=128, output_chan=128, size=(2,2))
        
        #Downsampling residual unit (output = 4x4x128)
        x = self.residual_unit(x, input_chan=128, output_chan=256, stride=2)
        
        # Final Residual Units after attention module(s)
        for _ in range(2):
            x = self.residual_unit(x, input_chan=256, output_chan=256)
        
        #Average Pooling, Dropout, and FC Layer to get softmax output vector
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.n_classes, activation=self.activation)(x)

        return Model(inputs=input_data, outputs=x)

    def residual_unit(self, x, input_chan, output_chan, stride=1):
        """
        Residual unit with bottleneck architecture and identity mapping.

        Args:
            x (Tensor): Input tensor.
            input_chan (int): Number of channels for the input
            output_chan (int): Number of channels for the output
        """
        residual = x
        self.input_chan = input_chan
        self.output_chan = output_chan

        # First convolution
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(output_chan//4, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)

        # Second convolution
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(output_chan//4, kernel_size=self.kernel_size, strides=stride, padding='same')(x)

        # Third convolution
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(output_chan, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)

        # Adjust dimensions of residual connection if necessary
        if (self.input_chan != self.output_chan):
            residual = Conv2D(output_chan, kernel_size=(1, 1), strides=stride, padding='same')(residual)
        
        #Add residual to output
        return Add()([x, residual])

    def attention_module_stage_1(self, x, input_chan, output_chan, size):
        """
        Attention module combining trunk and mask branches.

        Args:
            x (Tensor): Input tensor.
            input_chan (int): Number of channels for the input
            output_chan (int): Number of channels for the output
            size (tuple): The upsampling factors for rows and columns
        """
        # p Residual units before trunk and mask branches
        for _ in range(self.p):
            x = self.residual_unit(x, input_chan, output_chan)
        
        #Split into trunk and mask branches
        trunk = self.trunk_branch(x, input_chan, output_chan)
        
        mask = self.mask_branch_stage_1(x, input_chan, output_chan, size)
        
        # Use mask for 'soft attention' on trunk branch, weighting relevant features
        out =  (1 + mask) * trunk
        
        # p Residual units after combining trunk and mask branches
        for _ in range(self.p):
            out = self.residual_unit(out, input_chan, output_chan)

        return out

    def trunk_branch(self, x, input_chan, output_chan):
        """
        Trunk branch for core feature processing.

        Args:
            x (Tensor): Input tensor.
            input_chan (int): Number of channels for the input
            output_chan (int): Number of channels for the output
        """
        
        #t residual units in trunk branch
        for _ in range(self.t):
            x = self.residual_unit(x, input_chan, output_chan)
        return x

    def mask_branch_stage_1(self, x, input_chan, output_chan, size):
        """
        Mask branch for bottom-up and top-down processing.

        Args:
            x (Tensor): Input tensor.
            input_chan (int): Number of channels for the input
            output_chan (int): Number of channels for the output
            size (tuple): The upsampling factors for rows and columns
        """
        skip_connection = self.residual_unit(x, input_chan, output_chan)
        
        # Downsampling via max pooling
        mask = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

        # 2*r Middle residual units
        for _ in range(2 * self.r):
            mask = self.residual_unit(mask, input_chan, output_chan)

        # Upsampling + skip connection
        mask = UpSampling2D(size=size)(mask)
        
        out = mask + skip_connection
        
        #Conv and Normalization layers
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv2D(output_chan, kernel_size=(1, 1), strides=(1, 1), padding='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv2D(output_chan, kernel_size=(1, 1), strides=(1, 1), padding='same')(out)
        
        #Sigmoid activation to ensure values of mask are between 0 and 1
        out_sigmoid = Activation('sigmoid')(out)
        
        return out_sigmoid
    
    def attention_module_stage_2(self, x, input_chan, output_chan, size):
        """
        Attention module for stage 2.

        Args:
            x (Tensor): Input tensor.
            input_chan (int): Number of channels for the input
            output_chan (int): Number of channels for the output
            size (tuple): The upsampling factors for rows and columns
        """
        # p Residual units before trunk and mask branches
        for _ in range(self.p):
            x = self.residual_unit(x, input_chan, output_chan)
        
        #Split into trunk and mask branches
        trunk = self.trunk_branch(x, input_chan, output_chan)
        
        mask = self.mask_branch_stage_2(x, input_chan, output_chan, size)
        
        # Use mask for 'soft attention' on trunk branch, weighting relevant features
        out =  (1 + mask) * trunk
        
        # p Residual units after combining trunk and mask branches
        for _ in range(self.p):
            out = self.residual_unit(out, input_chan, output_chan)

        return out

    
    def mask_branch_stage_2(self, x, input_chan, output_chan, size):
        """
        Mask branch for stage 2.

        Args:
            x (Tensor): Input tensor.
            input_chan (int): Number of channels for the input
            output_chan (int): Number of channels for the output
            size (tuple): The upsampling factors for rows and columns
        """
        
        # Downsampling via max pooling
        mask = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

        # 2*r Middle residual units
        for _ in range(2 * self.r):
            mask = self.residual_unit(mask, input_chan, output_chan)

        # Upsampling 
        out = UpSampling2D(size=size)(mask)
        
        #Conv and Normalization layers
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv2D(output_chan, kernel_size=(1, 1), strides=(1, 1), padding='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv2D(output_chan, kernel_size=(1, 1), strides=(1, 1), padding='same')(out)
        
        #Sigmoid activation to ensure values of mask are between 0 and 1
        out_sigmoid = Activation('sigmoid')(out)
        
        return out_sigmoid
       