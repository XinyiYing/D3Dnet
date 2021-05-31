from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *

from model import *
from evaluation import psnr2, ssim
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import argparse, os

parser = argparse.ArgumentParser(description="PyTorch D3Dnet")
parser.add_argument("--scale_factor", type=int, default=4, help="scale")
parser.add_argument("--test_dataset_dir", default='./data', type=str, help="test_dataset dir")
parser.add_argument("--model", default='./log/D3Dnet.pth.tar', type=str, help="checkpoint")
parser.add_argument("--inType", type=str, default='y', help="RGB input or y input")
parser.add_argument("--batchSize", type=int, default=1, help="Test batch size")
parser.add_argument("--gpu", type=int, default=0, help="Test batch size")
parser.add_argument("--datasets", type=str, default=['Vid4'], help="Test batch size")

global opt, model
opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)

def demo_test(net, test_loader, scale_factor, dataset_name, video_name):
    PSNR_list = []
    SSIM_list = []
    with torch.no_grad():
        for idx_iter, (LR, SR_buicbic) in enumerate(test_loader):
            LR = Variable(LR).cuda()
            SR = net(LR)
            SR = torch.clamp(SR, 0, 1)

            if not os.path.exists('results/' + dataset + '/' + video_name):
                os.makedirs('./results/' + dataset + '/' + video_name)
            # ## save y images
            # SR_img = transforms.ToPILImage()(SR[0, :, :, :].cpu())
            # SR_img.save('results/' + dataset_name + '/' + video_name + '/sr_y_' + str(idx_iter+1).rjust(2, '0') + '.png')
            #
            # ## save rgb images
            # SR_buicbic[:, 0, :, :] = SR[:, 0, :, :].cpu()
            # SR_rgb = (ycbcr2rgb(SR_buicbic[0,:,:,:].permute(2,1,0))).permute(2,1,0)
            # SR_rgb = torch.clamp(SR_rgb, 0, 1)
            # SR_img = transforms.ToPILImage()(SR_rgb)
            # SR_img.save('results/' + dataset_name + '/' + video_name + '/sr_rgb_' + str(idx_iter+1).rjust(2, '0') + '.png')


def main(dataset_name):
    net = Net(opt.scale_factor).cuda()
    model = torch.load(opt.model)
    net.load_state_dict(model['state_dict'])

    PSNR_dataset = []
    SSIM_dataset = []

    video_list = os.listdir(opt.test_dataset_dir + '/' + dataset_name)
    for i in range(0, len(video_list)):
        video_name = video_list[i]
        test_set = InferLoader(opt.test_dataset_dir + '/' + dataset_name + '/' + video_name, scale_factor=opt.scale_factor)
        test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
        demo_test(net, test_loader, opt.scale_factor, dataset_name, video_name)    
            
if __name__ == '__main__':
    for i in range(len(opt.datasets)):
        dataset = opt.datasets[i]
        if not os.path.exists('results/' + dataset):
            os.makedirs('./results/' + dataset)
        import time
        start = time.time()
        main(dataset)
        end = time.time()
        print(end-start)
