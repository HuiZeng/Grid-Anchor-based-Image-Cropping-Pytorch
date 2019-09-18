from croppingDataset import GAICD
from croppingModel import build_crop_model
import time
import math
import sys
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
from scipy.stats import spearmanr, pearsonr

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--dataset_root', default='dataset/GAIC/', help='Dataset root directory path')
parser.add_argument('--image_size', default=256, type=int, help='Batch size for training')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, help='Use CUDA to train model')
parser.add_argument('--net_path', default='weights/ablation/cropping/mobilenetv2/downsample4_multi_Aug1_Align9_Cdim8/23_0.625_0.583_0.553_0.525_0.785_0.762_0.748_0.723_0.783_0.806.pth_____',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


data_loader = data.DataLoader(GAICD(image_size=args.image_size, dataset_dir=args.dataset_root, set='test'), args.batch_size, num_workers=args.num_workers, shuffle=False)

def test():

    net = build_crop_model(scale='multi', alignsize=9, reddim=8, loadweight=True, model='mobilenetv2', downsample=4)

    net.load_state_dict(torch.load(args.net_path))

    if args.cuda:
        net = torch.nn.DataParallel(net,device_ids=[0])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        net = net.cuda()

    net.eval()

    acc4_5 = []
    acc4_10 = []
    wacc4_5 = []
    wacc4_10 = []
    srcc = []
    pcc = []
    for n in range(4):
        acc4_5.append(0)
        acc4_10.append(0)
        wacc4_5.append(0)
        wacc4_10.append(0)

    for id, sample in enumerate(data_loader):
        image = sample['image']
        bboxs = sample['bbox']
        MOS = sample['MOS']

        roi = []

        for idx in range(0,len(bboxs['xmin'])):
            roi.append((0, bboxs['xmin'][idx],bboxs['ymin'][idx],bboxs['xmax'][idx],bboxs['ymax'][idx]))

        if args.cuda:
            image = Variable(image.cuda())
            roi = Variable(torch.Tensor(roi))
        else:
            image = Variable(image)
            roi = Variable(torch.Tensor(roi))

        t0 = time.time()
        out = net(image,roi)
        t1 = time.time()
        print('timer: %.4f sec.' % (t1 - t0))

        id_MOS = sorted(range(len(MOS)), key=lambda k: MOS[k], reverse=True)
        id_out = sorted(range(len(out)), key=lambda k: out[k], reverse=True)

        rank_of_returned_crop = []
        for k in range(4):
            rank_of_returned_crop.append(id_MOS.index(id_out[k]))

        for k in range(4):
            temp_acc_4_5 = 0.0
            temp_acc_4_10 = 0.0
            for j in range(k+1):
                if MOS[id_out[j]] >= MOS[id_MOS[4]]:
                    temp_acc_4_5 += 1.0
                if MOS[id_out[j]] >= MOS[id_MOS[9]]:
                    temp_acc_4_10 += 1.0
            acc4_5[k] += temp_acc_4_5 / (k+1.0)
            acc4_10[k] += temp_acc_4_10 / (k+1.0)

        for k in range(4):
            temp_wacc_4_5 = 0.0
            temp_wacc_4_10 = 0.0
            temp_rank_of_returned_crop = rank_of_returned_crop[:(k+1)]
            temp_rank_of_returned_crop.sort()
            for j in range(k+1):
                if temp_rank_of_returned_crop[j] <= 4:
                    temp_wacc_4_5 += 1.0 * math.exp(-0.2*(temp_rank_of_returned_crop[j]-j))
                if temp_rank_of_returned_crop[j] <= 9:
                    temp_wacc_4_10 += 1.0 * math.exp(-0.1*(temp_rank_of_returned_crop[j]-j))
            wacc4_5[k] += temp_wacc_4_5 / (k+1.0)
            wacc4_10[k] += temp_wacc_4_10 / (k+1.0)


        MOS_arr = []
        out = torch.squeeze(out).cpu().detach().numpy()
        for k in range(len(MOS)):
            MOS_arr.append(MOS[k].numpy()[0])
        srcc.append(spearmanr(MOS_arr,out)[0])
        pcc.append(pearsonr(MOS_arr,out)[0])


    for k in range(4):
        acc4_5[k] = acc4_5[k] / 200.0
        acc4_10[k] = acc4_10[k] / 200.0
        wacc4_5[k] = wacc4_5[k] / 200.0
        wacc4_10[k] = wacc4_10[k] / 200.0

    avg_srcc = sum(srcc) / 200.0
    avg_pcc = sum(pcc) / 200.0

    sys.stdout.write('[%.3f, %.3f, %.3f, %.3f] [%.3f, %.3f, %.3f, %.3f]\n' % (acc4_5[0],acc4_5[1],acc4_5[2],acc4_5[3],acc4_10[0],acc4_10[1],acc4_10[2],acc4_10[3]))
    sys.stdout.write('[%.3f, %.3f, %.3f, %.3f] [%.3f, %.3f, %.3f, %.3f]\n' % (wacc4_5[0],wacc4_5[1],wacc4_5[2],wacc4_5[3],wacc4_10[0],wacc4_10[1],wacc4_10[2],wacc4_10[3]))
    sys.stdout.write('[Avg SRCC: %.3f] [Avg PCC: %.3f]\n' % (avg_srcc,avg_pcc))


if __name__ == '__main__':
    test()
