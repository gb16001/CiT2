from torch import nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class STN(nn.Module):
    def __init__(self,L_net:nn.Module,L_channels:int,outSize,theta_0=[1, 0, 0, 0, 1, 0],detach:bool=True,fc_loc_init="zeros"):
        '''
        STN=localization net+grid affine+sample
        L_net:fm=>(bz,L_channels)=(fc_loc)=>theta:(bz,6)
        grid affine:theta=>pos grid
        theta_0:initial theta
        detach: detach L_net backprop to fm
        '''
        super().__init__()
        self.outSize,self.detach=outSize,detach
        self.Lnet=L_net
        self.fc_loc = nn.Sequential(
            nn.Linear(L_channels, 3 * 2)
        )
        if fc_loc_init=="normal":
            nn.init.normal_(self.fc_loc[0].weight, mean=0.0, std=1e-1)
        elif fc_loc_init=='kaiming':
            nn.init.kaiming_normal_(self.fc_loc[0].weight, nonlinearity='relu')
        elif fc_loc_init=="xavier":
            nn.init.xavier_normal_(self.fc_loc[0].weight, gain=1.0)
        elif fc_loc_init=="zeros":
            self.fc_loc[0].weight.data.zero_()
        self.fc_loc[0].bias.data.copy_(torch.tensor(theta_0, dtype=torch.float))
        return
    
    def forward(self,x):
        xs = self.Lnet(x.detach() if self.detach else x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        x=self.crop_img(x, theta)
        return x,theta.detach()
    def forward_img_pos_mask(self,img,pos,mask):
        img,theta=self.forward(img)
        pos=self.crop_img(pos,theta)
        mask=self.crop_mask(mask,theta)
        return img,pos,mask

    def crop_img(self, img, theta):
        grid = F.affine_grid(theta, [*img.shape[0:2],*self.outSize])
        # sample
        img = F.grid_sample(img, grid)
        return img
    def crop_mask(self, mask, theta):
        mask = mask.float().unsqueeze(1)  # (B, 1, H, W)
        mask=self.crop_img(mask,theta).squeeze(1)
        # grid = F.affine_grid(theta, [mask.size(0), mask.size(1),*self.outSize])
        # mask= F.grid_sample(mask,grid)
        mask=mask>0.5
        return mask
    @staticmethod
    def gen_Theta0_relative(p1, p2):
        '''
        p1,p2:(y,x)~(-1,1).rectfangle crop down=object box relative pos in origin img
        o1,o2=(-1,-1) (1,1)
        '''
        y, x = [(p2[i] + p1[i]) / 2 for i in range(len(p1))]
        b, a = ((p2[i] - p1[i]) / 2 for i in range(len(p1)))
        return [a,0,x,
                0,b,y]#grid point:(x,y)～[-1,1]
    pass 
class Neck:
    @staticmethod
    def STN_s32g276(d_fm:int=512,stn_detach:bool=True):
        '''
        input fm[37,23] from img[290,720]
        resnet18/34 stride 32 fm.
        out fm size [12,23]'''
        L_net=nn.Sequential(# s16
                # nn.AdaptiveAvgPool2d((37,23)),
                nn.Conv2d(d_fm,128,3,2,1), #s64
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,32,3,2,1), #s128
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32,16,3,2,1), #s256
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(16, 6, kernel_size=1),
                nn.BatchNorm2d(6),
                nn.ReLU(True),#6,5,3
                nn.Flatten(),

                nn.Linear(6*5*3,6*6),
                nn.Tanh(),
            )
        L_channels=6*6
        outSize=[12,23]
        theta_0=STN.gen_Theta0_relative(p1=(-1,-1/3),p2=(1,1/3))
        # (inSize=(19,45),p1=(6,0),p2=(12,45))
        stn = STN(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return stn
    @staticmethod
    def STN_s16g270(d_fm:int=256,stn_detach:bool=True):
        '''
        input fm[19,46] from img[290,720]
        resnet18/34 stride 16 fm.
        out fm size [6,45]'''
        L_net=nn.Sequential(# s16
                # nn.AdaptiveAvgPool2d((19,45)),
                nn.Conv2d(d_fm,64,3,2,1), #s32
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,32,3,2,1), #s64
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32,16,3,2,1), #s128
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(16, 2, kernel_size=1),
                nn.BatchNorm2d(2),
                nn.ReLU(True),
                nn.Flatten(),
            )
        L_channels=36
        outSize=[6,45]
        theta_0=STN.gen_Theta0_relative(p1=(-1,-1/3),p2=(1,1/3))
        # (inSize=(19,45),p1=(6,0),p2=(12,45))
        stn = STN(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return stn
    pass
class Encoder:
    
    pass