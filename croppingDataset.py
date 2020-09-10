import os
import torch.utils.data as data
import cv2
import math
import numpy as np
from augmentations import CropAugmentation

MOS_MEAN = 2.95
MOS_STD = 0.8
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


class TransformFunction(object):
    def __call__(self, sample, image_size):
        image, annotations = sample["image"], sample["annotations"]

        scale = image_size / min(image.shape[:2])
        h = round(image.shape[0] * scale / 32.0) * 32
        w = round(image.shape[1] * scale / 32.0) * 32
        resized_image = cv2.resize(image, (int(w), int(h))) / 256.0
        rgb_mean = np.array(RGB_MEAN, dtype=np.float32)
        rgb_std = np.array(RGB_STD, dtype=np.float32)
        resized_image = resized_image.astype(np.float32)
        resized_image -= rgb_mean
        resized_image = resized_image / rgb_std

        scale_height = float(resized_image.shape[0]) / image.shape[0]
        scale_width = float(resized_image.shape[1]) / image.shape[1]

        transformed_bbox = {}
        transformed_bbox["xmin"] = []
        transformed_bbox["ymin"] = []
        transformed_bbox["xmax"] = []
        transformed_bbox["ymax"] = []
        MOS = []
        for annotation in annotations:
            transformed_bbox["xmin"].append(
                math.floor(float(annotation[1]) * scale_width)
            )
            transformed_bbox["ymin"].append(
                math.floor(float(annotation[0]) * scale_height)
            )
            transformed_bbox["xmax"].append(
                math.ceil(float(annotation[3]) * scale_width)
            )
            transformed_bbox["ymax"].append(
                math.ceil(float(annotation[2]) * scale_height)
            )

            MOS.append((float(annotation[-1]) - MOS_MEAN) / MOS_STD)

        resized_image = resized_image.transpose((2, 0, 1))
        return {"image": resized_image, "bbox": transformed_bbox, "MOS": MOS}


class GAICD(data.Dataset):
    def __init__(
        self,
        image_size=256,
        dataset_dir="dataset/GAIC/",
        set="train",
        transform=TransformFunction(),
        augmentation=False,
    ):
        self.image_size = float(image_size)
        self.dataset_dir = dataset_dir
        self.set = set
        image_lists = os.listdir(self.dataset_dir + "images/" + set)
        self._imgpath = list()
        self._annopath = list()
        for image in image_lists:
            self._imgpath.append(os.path.join(self.dataset_dir, "images", set, image))
            self._annopath.append(
                os.path.join(self.dataset_dir, "annotations", set, image[:-3] + "txt")
            )
        self.transform = transform
        if augmentation:
            self.augmentation = CropAugmentation()
        else:
            self.augmentation = None

    def __getitem__(self, idx):
        image = cv2.imread(self._imgpath[idx])

        with open(self._annopath[idx], "r") as fid:
            annotations_txt = fid.readlines()

        annotations = list()
        for annotation in annotations_txt:
            annotation_split = annotation.split()
            if float(annotation_split[4]) != -2:
                annotations.append(
                    [
                        float(annotation_split[0]),
                        float(annotation_split[1]),
                        float(annotation_split[2]),
                        float(annotation_split[3]),
                        float(annotation_split[4]),
                    ]
                )

        if self.augmentation:
            image, annotations = self.augmentation(image, annotations)

        # to rgb
        image = image[:, :, (2, 1, 0)]

        sample = {"image": image, "annotations": annotations}

        if self.transform:
            sample = self.transform(sample, self.image_size)

        return sample

    def __len__(self):
        return len(self._imgpath)


class TransformFunctionTest(object):
    def __call__(self, image, image_size):

        scale = image_size / min(image.shape[:2])
        h = round(image.shape[0] * scale / 32.0) * 32
        w = round(image.shape[1] * scale / 32.0) * 32
        resized_image = cv2.resize(image, (int(w), int(h))) / 256.0
        rgb_mean = np.array(RGB_MEAN, dtype=np.float32)
        rgb_std = np.array(RGB_STD, dtype=np.float32)
        resized_image = resized_image.astype(np.float32)
        resized_image -= rgb_mean
        resized_image = resized_image / rgb_std

        scale_height = image.shape[0] / float(resized_image.shape[0])
        scale_width = image.shape[1] / float(resized_image.shape[1])

        bboxes = generate_bboxes(resized_image)

        transformed_bbox = {}
        transformed_bbox["xmin"] = []
        transformed_bbox["ymin"] = []
        transformed_bbox["xmax"] = []
        transformed_bbox["ymax"] = []
        source_bboxes = list()

        for bbox in bboxes:
            source_bboxes.append(
                [
                    round(bbox[0] * scale_height),
                    round(bbox[1] * scale_width),
                    round(bbox[2] * scale_height),
                    round(bbox[3] * scale_width),
                ]
            )
            transformed_bbox["xmin"].append(bbox[1])
            transformed_bbox["ymin"].append(bbox[0])
            transformed_bbox["xmax"].append(bbox[3])
            transformed_bbox["ymax"].append(bbox[2])

        resized_image = resized_image.transpose((2, 0, 1))
        return resized_image, transformed_bbox, source_bboxes


def generate_bboxes(image):

    bins = 12.0
    h = image.shape[0]
    w = image.shape[1]
    step_h = h / bins
    step_w = w / bins
    annotations = list()
    for x1 in range(0, 4):
        for y1 in range(0, 4):
            for x2 in range(8, 12):
                for y2 in range(8, 12):
                    if (
                        (x2 - x1) * (y2 - y1) > 0.4999 * bins * bins
                        and (y2 - y1) * step_w / (x2 - x1) / step_h > 0.5
                        and (y2 - y1) * step_w / (x2 - x1) / step_h < 2.0
                    ):
                        annotations.append(
                            [
                                float(step_h * (0.5 + x1)),
                                float(step_w * (0.5 + y1)),
                                float(step_h * (0.5 + x2)),
                                float(step_w * (0.5 + y2)),
                            ]
                        )

    return annotations


def generate_bboxes_16_9(image):

    h = image.shape[0]
    w = image.shape[1]
    h_step = 9
    w_step = 16
    annotations = list()
    for i in range(14, 30):
        out_h = h_step * i
        out_w = w_step * i
        if out_h < h and out_w < w and out_h * out_w > 0.4 * h * w:
            for w_start in range(0, w - out_w, w_step):
                for h_start in range(0, h - out_h, h_step):
                    annotations.append(
                        [
                            float(h_start),
                            float(w_start),
                            float(h_start + out_h - 1),
                            float(w_start + out_w - 1),
                        ]
                    )
    return annotations


def generate_bboxes_4_3(image):

    h = image.shape[0]
    w = image.shape[1]
    h_step = 12
    w_step = 16
    annotations = list()
    for i in range(14, 30):
        out_h = h_step * i
        out_w = w_step * i
        if out_h < h and out_w < w and out_h * out_w > 0.4 * h * w:
            for w_start in range(0, w - out_w, w_step):
                for h_start in range(0, h - out_h, h_step):
                    annotations.append(
                        [
                            float(h_start),
                            float(w_start),
                            float(h_start + out_h - 1),
                            float(w_start + out_w - 1),
                        ]
                    )
    return annotations


def generate_bboxes_1_1(image):

    h = image.shape[0]
    w = image.shape[1]
    h_step = 12
    w_step = 12
    annotations = list()
    for i in range(14, 30):
        out_h = h_step * i
        out_w = w_step * i
        if out_h < h and out_w < w and out_h * out_w > 0.4 * h * w:
            for w_start in range(0, w - out_w, w_step):
                for h_start in range(0, h - out_h, h_step):
                    annotations.append(
                        [
                            float(h_start),
                            float(w_start),
                            float(h_start + out_h - 1),
                            float(w_start + out_w - 1),
                        ]
                    )
    return annotations


class setup_test_dataset(data.Dataset):
    def __init__(
        self,
        image_size=256.0,
        dataset_dir="testsetDir",
        transform=TransformFunctionTest(),
    ):
        self.image_size = float(image_size)
        self.dataset_dir = dataset_dir
        image_lists = sorted(os.listdir(self.dataset_dir))
        self._imgpath = list()
        self._annopath = list()
        for image in image_lists:
            self._imgpath.append(os.path.join(self.dataset_dir, image))
        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(self._imgpath[idx])

        # to rgb
        image = image[:, :, (2, 1, 0)]

        if self.transform:
            resized_image, transformed_bbox, source_bboxes = self.transform(
                image, self.image_size
            )

        sample = {
            "imgpath": self._imgpath[idx],
            "image": image,
            "resized_image": resized_image,
            "tbboxes": transformed_bbox,
            "sourceboxes": source_bboxes,
        }

        return sample

    def __len__(self):
        return len(self._imgpath)
