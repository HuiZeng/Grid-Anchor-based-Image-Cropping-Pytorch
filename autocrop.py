from croppingModel import build_crop_model
from croppingDataset import setup_test_dataset

import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
import argparse

from tqdm import tqdm

networks = {
    "mobilenetv2_0.5": {
        "model": "mobilenetv2",
        "path": "mobilenetv2_0.5-eaa6f9ad.pth",
    },
    "mobilenetv2_0.75": {
        "model": "mobilenetv2",
        "path": "mobilenetv2_0.75-dace9791.pth",
    },
    "mobilenetv2_1.0": {
        "model": "mobilenetv2",
        "path": "mobilenetv2_1.0-0c6065bc.pth",
    },
    "shufflenetv2_x0.5": {
        "model": "shufflenetv2",
        "path": "shufflenetv2_x0.5_60.646_81.696.pth.tar",
    },
    "shufflenetv2_x1": {
        "model": "shufflenetv2",
        "path": "shufflenetv2_x1_69.402_88.374.pth.tar",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="input_image_dir")
    parser.add_argument(dest="output_image_dir")
    parser.add_argument(
        "--network", choices=sorted(networks), default="mobilenetv2_1.0"
    )
    args = parser.parse_args()

    os.makedirs(args.output_image_dir, exist_ok=True)

    net_conf = networks[args.network]
    net = build_crop_model(
        scale="multi",
        alignsize=9,
        reddim=8,
        loadweight=True,
        model=net_conf["model"],
        downsample=4,
    )
    net.load_state_dict(
        torch.load(
            os.path.join("pretrained_model", net_conf["path"]), map_location="cpu"
        ),
        strict=False,
    )
    net.eval()

    dataset = setup_test_dataset(dataset_dir=args.input_image_dir)
    data_loader = DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False, pin_memory=True
    )

    for sample in tqdm(data_loader):
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

        resized_image = Variable(resized_image)
        roi = Variable(torch.Tensor(roi))

        """
        t0 = time.time()
        for r in range(0,100):
            out = net(resized_image,roi)
        t1 = time.time()
        print('timer: %.4f sec.' % (t1 - t0))
        """

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
            imgname = os.path.basename(imgpath[0])
            filename, file_extension = os.path.splitext(imgname)
            cv2.imwrite(
                args.output_image_dir + "/" + filename + "_" + str(i) + file_extension,
                top1_crop[:, :, (2, 1, 0)],
            )


if __name__ == "__main__":
    main()
