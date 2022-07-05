from common import *
from siim import *

from timm.models.efficientnet import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        e = efficientnet_b3a(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)
        self.b0 = nn.Sequential(
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
        self.b8 = nn.Sequential(
            e.conv_head, #384, 1536
            e.bn2,
            e.act2,
        )

        self.logit = nn.Linear(1536,num_study_label)
        self.mask = nn.Sequential(
            nn.Conv2d(136, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )


    # @torch.cuda.amp.autocast()
    def forward(self, image):
        batch_size = len(image)
        x = 2*image-1     # ; print('input ',   x.shape)

        x = self.b0(x) #; print (x.shape)  # torch.Size([2, 40, 256, 256])
        x = self.b1(x) #; print (x.shape)  # torch.Size([2, 24, 256, 256])
        x = self.b2(x) #; print (x.shape)  # torch.Size([2, 32, 128, 128])
        x = self.b3(x) #; print (x.shape)  # torch.Size([2, 48, 64, 64])
        x = self.b4(x) #; print (x.shape)  # torch.Size([2, 96, 32, 32])
        x = self.b5(x) #; print (x.shape)  # torch.Size([2, 136, 32, 32])
        #------------
        mask = self.mask(x)
        #-------------
        x = self.b6(x) #; print (x.shape)  # torch.Size([2, 232, 16, 16])
        x = self.b7(x) #; print (x.shape)  # torch.Size([2, 384, 16, 16])
        x = self.b8(x) #; print (x.shape)  # torch.Size([2, 1536, 16, 16])
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        #x = F.dropout(x, 0.5, training=self.training)
        logit = self.logit(x)
        return logit, mask





# check #################################################################

def run_check_net():
    batch_size = 2
    C, H, W = 3, 512, 512
    #C, H, W = 3, 640, 640
    image = torch.randn(batch_size, C, H, W).cuda()
    mask  = torch.randn(batch_size, num_study_label, H, W).cuda()

    net = Net().cuda()
    logit, mask = net(image)

    print(image.shape)
    print(logit.shape)
    print(mask.shape)


# main #################################################################
if __name__ == '__main__':
    run_check_net()


