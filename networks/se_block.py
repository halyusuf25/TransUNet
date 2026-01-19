# copied from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py

from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, verbose=False): #input_type can be 'feature' or 'attention_map'
        super(SELayer, self).__init__()
        self.avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.avg_pool1d = nn.AdaptiveAvgPool1d(1)  # expects (B, C, N) -> (B, C, 1)
        self.verbose = verbose

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 3: # for feature map with shape (B, C, N)
            if self.verbose:
                print(f"SELayer input shape: {x.shape}, will process as 3D tensor (INPUT FEATURE)")
                
            b, n, c = x.size()
            x_bcn = x.transpose(1, 2)            # (B, C, N) -> (B, N, C)
            s = self.avg_pool1d(x_bcn)           # (B, C, 1)
            s = s.squeeze(-1)                    # (B, C)
            y = self.fc(s)                       # (B, C)
            y = y.reshape(b, 1, c)               # (B, 1, C)  (reshape is safer than view)
            
        elif x.dim() == 4: # for attention map with shape (B, C, H, W)
            if self.verbose:
                print(f"SELayer input shape: {x.shape}, will process as 4D tensor (ATTENTION MAP)")
            
            b, c, _, _ = x.size()               # (B, C, H, W)
            y = self.avg_pool2d(x).view(b, c)   # (B, C)
            y = self.fc(y).view(b, c, 1, 1)     # (B, C, 1, 1)
        else:
            raise ValueError("Input tensor must be 3D or 4D but got {}D".format(x.dim()))
        
        x = x * y.expand_as(x)
        if self.verbose:
            print(f"SELayer scaling factors shape: {y.shape} and output (x) shape: {x.shape}")
    
        return x, y