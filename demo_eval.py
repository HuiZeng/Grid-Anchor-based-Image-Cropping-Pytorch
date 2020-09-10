from croppingModel import build_crop_model
from croppingDataset import setup_test_dataset
import os
import torch
import cv2
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
import time


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description="Grid anchor based image cropping With Pytorch"
)
parser.add_argument(
    "--input_dir",
    default="dataset/GAIC/images/test",
    help="root directory path of testing images",
)
parser.add_argument(
    "--output_dir",
    default="dataset/test_result",
    help="root directory path of testing images",
)
parser.add_argument("--batch_size", default=1, type=int, help="Batch size for training")
parser.add_argument(
    "--num_workers", default=0, type=int, help="Number of workers used in dataloading"
)
parser.add_argument(
    "--cuda", default=False, type=str2bool, help="Use CUDA to train model"
)
parser.add_argument(
    "--net_path",
    default="pretrained_model/mobilenet_0.625_0.583_0.553_0.525_0.785_0.762_0.748_0.723_0.783_0.806.pth",
    help="Directory for saving checkpoint models",
)
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    if not args.cuda:
        print(
            "WARNING: It looks like you have a CUDA device, but aren't "
            + "using CUDA.\nRun with --cuda for optimal training speed."
        )
        torch.set_default_tensor_type("torch.FloatTensor")

else:
    torch.set_default_tensor_type("torch.FloatTensor")

dataset = setup_test_dataset(dataset_dir=args.input_dir)


def test():

    net = build_crop_model(
        scale="multi",
        alignsize=9,
        reddim=8,
        loadweight=True,
        model="mobilenetv2",
        downsample=4,
    )
    net.load_state_dict(torch.load(args.net_path, map_location="cpu"), strict=False)
    net.eval()

    if args.cuda:
        net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True
        net = net.cuda()

    data_loader = data.DataLoader(
        dataset,
        args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    for id, sample in enumerate(data_loader):
        imgpath = sample["imgpath"]
        image = sample["image"]
        bboxes = sample["sourceboxes"]
        resized_image = sample["resized_image"]
        tbboxes = sample["tbboxes"]

        if len(tbboxes["xmin"]) == 0:
            continue

        roi = []

        for idx in range(0, len(tbboxes["xmin"])):
            roi.append(
                (
                    0,
                    tbboxes["xmin"][idx],
                    tbboxes["ymin"][idx],
                    tbboxes["xmax"][idx],
                    tbboxes["ymax"][idx],
                )
            )

        if args.cuda:
            resized_image = Variable(resized_image.cuda())
            roi = Variable(torch.Tensor(roi))
        else:
            resized_image = Variable(resized_image)
            roi = Variable(torch.Tensor(roi))

        t0 = time.time()
        for r in range(0, 100):
            out = net(resized_image, roi)
        t1 = time.time()
        print("timer: %.4f sec." % (t1 - t0))

        out = net(resized_image, roi)

        id_out = sorted(range(len(out)), key=lambda k: out[k], reverse=True)
        image = image.cpu().numpy().squeeze(0)

        for i in range(4):
            top1_box = bboxes[id_out[i]]
            top1_box = [
                top1_box[0].numpy()[0],
                top1_box[1].numpy()[0],
                top1_box[2].numpy()[0],
                top1_box[3].numpy()[0],
            ]
            top1_crop = image[
                int(top1_box[0]) : int(top1_box[2]), int(top1_box[1]) : int(top1_box[3])
            ]
            imgname = imgpath[0].split("/")[-1]
            cv2.imwrite(
                args.output_dir + "/" + imgname[:-4] + "_" + str(i) + imgname[-4:],
                top1_crop[:, :, (2, 1, 0)],
            )


if __name__ == "__main__":
    test()
