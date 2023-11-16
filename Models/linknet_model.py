import torch
import torch.nn as nn


class linkNet(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], nbClasses = 1):
        super(linkNet, self).__init__()
        
        self.channels = channels
        self._mconv_1 = conv_block(3, channels[0], 7, 2)
        self._mmax_pool = nn.MaxPool2d(3, 2, padding=1)
        
        self._mencoder_1 = encoder(channels[0], channels[0])
        self._mencoder_2 = encoder(channels[0], channels[1])
        self._mencoder_3 = encoder(channels[1], channels[2])
        self._mencoder_4 = encoder(channels[2], channels[3])
        
        self._mdecoder_1 = decoder(channels[0], channels[0])
        self._mdecoder_2 = decoder(channels[1], channels[0])
        self._mdecoder_3 = decoder(channels[2], channels[1])
        self._mdecoder_4 = decoder(channels[3], channels[2])
        
        self._deconv_1 = deconv_block(channels[0], 32)
        self._mconv_2 = conv_block(32, 32, 3)
        self._deconv_2 = deconv_block(32, nbClasses, 2, 2, 0)
        
    
    def forward(self, x):
                                        
        conv_down_out = self._mconv_1(x)
        max_pool_out = self._mmax_pool(conv_down_out)
                
        encoder_1_out = self._mencoder_1(max_pool_out)
        encoder_2_out = self._mencoder_2(encoder_1_out)                
        encoder_3_out = self._mencoder_3(encoder_2_out)        
        encoder_4_out = self._mencoder_4(encoder_3_out)
        
        
        decoder_4_out = self._mdecoder_4(encoder_4_out, encoder_3_out.size()) + encoder_3_out
        decoder_3_out = self._mdecoder_3(decoder_4_out, encoder_2_out.size()) + encoder_2_out
        decoder_2_out = self._mdecoder_2(decoder_3_out, encoder_1_out.size()) + encoder_1_out
        decoder_1_out = self._mdecoder_1(decoder_2_out, max_pool_out.size())
        
        
        deconv_out = self._deconv_1(decoder_1_out, conv_down_out.size())
        conv_2_out = self._mconv_2(deconv_out)
        out = self._deconv_2(conv_2_out, x.size())
        
                
        return out
        


class conv_block(nn.Module):
    def __init__(self, in_map, out_map, kernel = 3, stride = 1, activation = True):
        super(conv_block, self).__init__()
        
        self._mconv = nn.Sequential(
            nn.Conv2d(in_map, out_map, kernel, stride, (kernel)//2),
            nn.BatchNorm2d(out_map)            
            )
        
        if(activation):
            self._mconv.add_module("conv_block_relu", nn.ReLU(inplace=True))
        
    def forward(self, x):        
        out = self._mconv(x)
        
        return out
    
class deconv_block(nn.Module):
    def __init__(self, in_map, out_map, kernel = 3, stride = 2, padding = 1):
        super(deconv_block, self).__init__()
                
        self._conv_trans_2d =  nn.ConvTranspose2d(in_map, out_map, kernel, stride, padding)
        self._batch_norm_2d = nn.BatchNorm2d(out_map)
        self._relu = nn.ReLU(inplace=True)        
        
    def forward(self, x, output_size):
        out = self._conv_trans_2d(x, output_size = output_size)
        
        return out
    
class res_block(nn.Module):
    def __init__(self, in_map, out_map, downsample = False):
        super(res_block, self).__init__()
                        
        self._mconv_2 = conv_block(out_map, out_map, 3, 1, False)        
        
        if downsample == True:
            stride = 2            
        else:
            stride = 1            
            
        self._mconv_1 = conv_block(in_map, out_map, 3, stride)
        self._mdownsample = nn.Sequential(
                nn.Conv2d(in_map, out_map, 1, stride),
                nn.BatchNorm2d(out_map)
                )
        self._relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        residual = x
        out = self._mconv_1(x)
        out = self._mconv_2(out)
        residual = self._mdownsample(x)
        #print("residual size,", residual.size())
        out = residual + out
        out = self._relu(out)
        
        return out
    
class encoder(nn.Module):
    def __init__(self, in_map, out_map):
        super(encoder, self).__init__()
        
        self._mres_1 = res_block(in_map, out_map, True)
        self._mres_2 = res_block(out_map, out_map)
        
        
    def forward(self, x):
        out = self._mres_1(x)
        out = self._mres_2(out)
        
        return out
    
class decoder(nn.Module):
    def __init__(self, in_map, out_map, padding = 1):
        super(decoder, self).__init__()
        
        self._mconv_1 = conv_block(in_map, in_map//4, 1)
        self._mdeconv_1 = deconv_block(in_map//4, in_map//4, 3, 2, padding)
        self._mconv_2 = conv_block(in_map//4, out_map, 1)        
        
    def forward(self, x, output_size):
        out = self._mconv_1(x)
        out = self._mdeconv_1(out, output_size = output_size)
        out = self._mconv_2(out)
        
        return out        