from common import *
from siim import *

from mlp_mixer_pytorch import MLPMixer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model =  MLPMixer(image_size = 640,channels = 3,patch_size = 16,dim = 512,depth = 12,num_classes = num_study_label)
       
    # @torch.cuda.amp.autocast()
    def forward(self, image):
        batch_size = len(image)
        x = 2*image-1      #; print('\nx: ',   x.shape) #  torch.Size([2, 3, 640, 640])
        logit = self.model(x)  #; print ('\nlogit(x): ',logit.shape) #torch.Size([2, 4])
        return logit
 




# check #################################################################

def run_check_net():
    batch_size = 2
    #C, H, W = 3, 600, 600
    C, H, W = 3, 640, 640
    image = torch.randn(batch_size, C, H, W).cuda()
    #mask  = torch.randn(batch_size, num_study_label, H, W).cuda()

    net = Net().cuda()
    logit = net(image)

    print(image.shape)
    print(logit.shape)
    #print(mask.shape)


# main #################################################################
if __name__ == '__main__':
    run_check_net()


