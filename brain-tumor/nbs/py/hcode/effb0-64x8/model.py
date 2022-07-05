#model.py
from common import *
from timm.models.efficientnet import *

class Net(torch.nn.Module):
    def __init__(self, verbouse=False):
        super(Net, self).__init__()

        e = efficientnet_b0(pretrained=True, drop_rate=0.2, drop_path_rate=0.2)
        self.verbouse=verbouse
        self.b0 = torch.nn.Sequential(
            e.conv_stem,
            e.bn1,
            e.act1,
        )
        self.b1 = e.blocks[0]
        self.b2 = e.blocks[1]
        self.b3 = e.blocks[2]
        self.b4 = e.blocks[3]
        self.b5 = e.blocks[4]
        self.b6 = e.blocks[5]
        self.b7 = e.blocks[6]
        self.b8 = torch.nn.Sequential(
            e.conv_head,
            e.bn2,
            e.act2,
        )

        self.logit = torch.nn.Linear(1280,1)
       
    # @torch.cuda.amp.autocast()
    def forward(self, image):
        batch_size = len(image)
        x = 2*image-1   
        #if (self.verbouse): print(f'i {x.size()}') 

        x = self.b0(x) 
        #if (self.verbouse): print(f's {x.size()}')
        x = self.b1(x) 
        #if (self.verbouse): print(f'0 {x.size()}')
        x = self.b2(x) 
        #if (self.verbouse): print(f'1 {x.size()}')
        x = self.b3(x)
        #if (self.verbouse): print(f'2 {x.size()}')
        x = self.b4(x) 
        #if (self.verbouse): print(f'3 {x.size()}')
        x = self.b5(x) 
        #if (self.verbouse): print(f'4 {x.size()}')
        x = self.b6(x) 
        #if (self.verbouse): print(f'5 {x.size()}')
        x = self.b7(x) 
        #if (self.verbouse): print(f'6 {x.size()}')
        x = self.b8(x) 
        #if (self.verbouse): print(f'f {x.size()}')
        x = torch.nn.functional.adaptive_avg_pool2d(x,1).reshape(batch_size,-1) 
        #if (self.verbouse): print(f'a {x.size()}')
        x = torch.nn.functional.dropout(x, 0.5, training=self.training)
        #if (self.verbouse): print(f'd {x.size()}')
        logit = self.logit(x)
        return logit

def run_check_net():
    batch_size = 2
    C, H, W = 3, 64, 64
    image = torch.randn(batch_size, C, H, W).cuda()
    net = Net().cuda()
    logit = net(image)

    print('image: ',image.shape)
    print('logit: ',logit.shape)