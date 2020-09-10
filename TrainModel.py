from croppingModel import build_crop_model
from croppingDataset import GAICD
import os
import sys
import time
import math
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import argparse
import numpy as np
import random
from scipy.stats import spearmanr, pearsonr

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser(description="Grid anchor based image cropping")
parser.add_argument(
    "--dataset_root", default="dataset/GAIC/", help="Dataset root directory path"
)
parser.add_argument("--base_model", default="mobilenetv2", help="Pretrained base model")
parser.add_argument(
    "--scale", default="multi", type=str, help="choose single or multi scale"
)
parser.add_argument("--downsample", default=4, type=int, help="downsample time")
parser.add_argument(
    "--augmentation", default=1, type=int, help="choose single or multi scale"
)
parser.add_argument(
    "--image_size", default=256, type=int, help="Batch size for training"
)
parser.add_argument(
    "--align_size", default=9, type=int, help="Spatial size of RoIAlign and RoDAlign"
)
parser.add_argument(
    "--reduced_dim", default=8, type=int, help="Spatial size of RoIAlign and RoDAlign"
)
parser.add_argument("--batch_size", default=1, type=int, help="Batch size for training")
parser.add_argument(
    "--resume",
    default=None,
    type=str,
    help="Checkpoint state_dict file to resume training from",
)
parser.add_argument(
    "--start_iter", default=0, type=int, help="Resume training at this iter"
)
parser.add_argument(
    "--num_workers", default=0, type=int, help="Number of workers used in dataloading"
)
parser.add_argument(
    "--lr", "--learning-rate", default=1e-4, type=float, help="initial learning rate"
)
parser.add_argument(
    "--save_folder",
    default="weights/ablation/cropping/",
    help="Directory for saving checkpoint models",
)
args = parser.parse_args()

args.save_folder = (
    args.save_folder
    + args.base_model
    + "/"
    + "downsample"
    + str(args.downsample)
    + "_"
    + args.scale
    + "_Aug"
    + str(args.augmentation)
    + "_Align"
    + str(args.align_size)
    + "_Cdim"
    + str(args.reduced_dim)
)

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

cuda = True if torch.cuda.is_available() else False

if cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")


data_loader_train = data.DataLoader(
    GAICD(
        image_size=args.image_size,
        dataset_dir=args.dataset_root,
        set="train",
        augmentation=args.augmentation,
    ),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    worker_init_fn=random.seed(SEED),
)

data_loader_test = data.DataLoader(
    GAICD(image_size=args.image_size, dataset_dir=args.dataset_root, set="test"),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
)

net = build_crop_model(
    scale=args.scale,
    alignsize=args.align_size,
    reddim=args.reduced_dim,
    loadweight=True,
    model=args.base_model,
    downsample=args.downsample,
)

# fix the batch normalization in mobilenet and shufflenet because batchsize = 1
net.eval()

if cuda:
    net = torch.nn.DataParallel(net, device_ids=[0])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # cudnn.benchmark = True
    net = net.cuda()

optimizer = optim.Adam(net.parameters(), lr=args.lr)


def test():
    acc4_5 = []
    acc4_10 = []
    wacc4_5 = []
    wacc4_10 = []
    srcc = []
    pcc = []
    total_loss = 0
    avg_loss = 0
    for n in range(4):
        acc4_5.append(0)
        acc4_10.append(0)
        wacc4_5.append(0)
        wacc4_10.append(0)

    for id, sample in enumerate(data_loader_test):
        image = sample["image"]
        bboxs = sample["bbox"]
        MOS = sample["MOS"]

        roi = []

        for idx in range(0, len(bboxs["xmin"])):
            roi.append(
                (
                    0,
                    bboxs["xmin"][idx],
                    bboxs["ymin"][idx],
                    bboxs["xmax"][idx],
                    bboxs["ymax"][idx],
                )
            )

        if cuda:
            image = Variable(image.cuda())
            roi = Variable(torch.Tensor(roi))
        else:
            image = Variable(image)
            roi = Variable(roi)

        # t0 = time.time()
        out = net(image, roi)
        loss = torch.nn.SmoothL1Loss(reduction="elementwise_mean")(
            out.squeeze(), torch.Tensor(MOS)
        )
        total_loss += loss.item()
        avg_loss = total_loss / (id + 1)

        id_MOS = sorted(range(len(MOS)), key=lambda k: MOS[k], reverse=True)
        id_out = sorted(range(len(out)), key=lambda k: out[k], reverse=True)
        for k in range(4):
            temp_acc_4_5 = 0.0
            temp_acc_4_10 = 0.0
            for j in range(k + 1):
                if MOS[id_out[j]] >= MOS[id_MOS[4]]:
                    temp_acc_4_5 += 1.0
                if MOS[id_out[j]] >= MOS[id_MOS[9]]:
                    temp_acc_4_10 += 1.0
            acc4_5[k] += temp_acc_4_5 / (k + 1.0)
            acc4_10[k] += temp_acc_4_10 / (k + 1.0)

        rank_of_returned_crop = []
        for k in range(4):
            rank_of_returned_crop.append(id_MOS.index(id_out[k]))

        for k in range(4):
            temp_wacc_4_5 = 0.0
            temp_wacc_4_10 = 0.0
            temp_rank_of_returned_crop = rank_of_returned_crop[: (k + 1)]
            temp_rank_of_returned_crop.sort()
            for j in range(k + 1):
                if temp_rank_of_returned_crop[j] <= 4:
                    temp_wacc_4_5 += 1.0 * math.exp(
                        -0.2 * (temp_rank_of_returned_crop[j] - j)
                    )
                if temp_rank_of_returned_crop[j] <= 9:
                    temp_wacc_4_10 += 1.0 * math.exp(
                        -0.1 * (temp_rank_of_returned_crop[j] - j)
                    )
            wacc4_5[k] += temp_wacc_4_5 / (k + 1.0)
            wacc4_10[k] += temp_wacc_4_10 / (k + 1.0)

        MOS_arr = []
        out = torch.squeeze(out).cpu().detach().numpy()
        for k in range(len(MOS)):
            MOS_arr.append(MOS[k].numpy()[0])
        srcc.append(spearmanr(MOS_arr, out)[0])
        pcc.append(pearsonr(MOS_arr, out)[0])

        # t1 = time.time()

        # print('timer: %.4f sec.' % (t1 - t0))
    for k in range(4):
        acc4_5[k] = acc4_5[k] / 200.0
        acc4_10[k] = acc4_10[k] / 200.0
        wacc4_5[k] = wacc4_5[k] / 200.0
        wacc4_10[k] = wacc4_10[k] / 200.0

    avg_srcc = sum(srcc) / 200.0
    avg_pcc = sum(pcc) / 200.0

    return acc4_5, acc4_10, avg_srcc, avg_pcc, avg_loss, wacc4_5, wacc4_10


def train():

    for epoch in range(0, 80):
        total_loss = 0
        for id, sample in enumerate(data_loader_train):

            image = sample["image"]
            bboxs = sample["bbox"]

            roi = []
            MOS = []

            random_ID = range(0, len(bboxs["xmin"]))
            random.shuffle(random_ID)

            for idx in random_ID[:64]:
                roi.append(
                    (
                        0,
                        bboxs["xmin"][idx],
                        bboxs["ymin"][idx],
                        bboxs["xmax"][idx],
                        bboxs["ymax"][idx],
                    )
                )
                MOS.append(sample["MOS"][idx])

            if cuda:
                image = Variable(image.cuda())
                roi = Variable(torch.Tensor(roi))
                MOS = torch.Tensor(MOS)
            else:
                image = Variable(image)
                roi = Variable(roi)

            # forward

            out = net(image, roi)
            loss = torch.nn.SmoothL1Loss(reduction="elementwise_mean")(
                out.squeeze(), MOS
            )
            total_loss += loss.item()
            avg_loss = total_loss / (id + 1)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Train Loss: %.4f]"
                % (epoch, 79, id, len(data_loader_train), avg_loss)
            )

        acc4_5, acc4_10, avg_srcc, avg_pcc, test_avg_loss, wacc4_5, wacc4_10 = test()
        sys.stdout.write(
            "[Test Loss: %.4f] [%.3f, %.3f, %.3f, %.3f] [%.3f, %.3f, %.3f, %.3f] [SRCC: %.3f] [PCC: %.3f]\n"
            % (
                test_avg_loss,
                acc4_5[0],
                acc4_5[1],
                acc4_5[2],
                acc4_5[3],
                acc4_10[0],
                acc4_10[1],
                acc4_10[2],
                acc4_10[3],
                avg_srcc,
                avg_pcc,
            )
        )
        sys.stdout.write(
            "[%.3f, %.3f, %.3f, %.3f] [%.3f, %.3f, %.3f, %.3f]\n"
            % (
                wacc4_5[0],
                wacc4_5[1],
                wacc4_5[2],
                wacc4_5[3],
                wacc4_10[0],
                wacc4_10[1],
                wacc4_10[2],
                wacc4_10[3],
            )
        )
        torch.save(
            net.module.state_dict(),
            args.save_folder
            + "/"
            + repr(epoch)
            + "_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f"
            % (
                acc4_5[0],
                acc4_5[1],
                acc4_5[2],
                acc4_5[3],
                acc4_10[0],
                acc4_10[1],
                acc4_10[2],
                acc4_10[3],
                avg_srcc,
                avg_pcc,
            )
            + ".pth",
        )


if __name__ == "__main__":
    train()
