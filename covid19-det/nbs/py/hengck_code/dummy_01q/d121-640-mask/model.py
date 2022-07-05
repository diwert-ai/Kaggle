from common import *
from siim import *

from timm.models.densenet import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        e = densenet121(pretrained=True, drop_rate=0.4)
        self.f = e.features
        self.logit = nn.Linear(1024,num_study_label)
        self.mask = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=3, padding=1),
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
        x = 2*image-1       #;print('input: ',   x.shape) # torch.Size([2, 3, 640, 640])

        x = self.f (x)    #;print ('features: ',x.shape)  # torch.Size([2, 1024, 20, 20])
        #------------
        mask = self.mask(x)
        #-------------
        
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)   #; print('out: ',x.shape) # torch.Size([2, 1024])
        #x = F.dropout(x, 0.5, training=self.training)
        logit = self.logit(x)
        return logit, mask




# check #################################################################

def run_check_net():
    batch_size = 2
    #C, H, W = 3, 600, 600
    C, H, W = 3, 640, 640
    image = torch.randn(batch_size, C, H, W).cuda()
    mask  = torch.randn(batch_size, num_study_label, H, W).cuda()

    net = Net().cuda()
    logit, mask = net(image)

    print('image: ',image.shape)
    print('logit: ',logit.shape)
    print('mask: ',mask.shape)

# main #################################################################
if __name__ == '__main__':
    run_check_net()


