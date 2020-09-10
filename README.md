# Grid-Anchor-based-Image-Cropping-Pytorch
This code includes several extensions we have made to our conference version. Please read the [paper](https://drive.google.com/open?id=1Bd1VaqYVycB7Npv5OdXKl-znKs_APl4n) for details.

## Change Log
### 2020-09-10
- Add an `autocrop.py` driver for batch processing on image folder.
- Overrides `_roi_align.so` and `_roi_align.so` with corresponding CPU version. (I'm using a MAC without nvidia gpu :( ). 
- Removed a lot of `*.pyc` files
- Add a `requirements.txt` with correct (old) dependencies.
- Fix formatting issues with black in pre-commit hook

## Requirements
python 2.7, pytorch 0.4.1, numpy, cv2, scipy. 

## Usage
1. Download the source code, the [dataset](https://drive.google.com/open?id=1X9xK5O9cx4_MvDkWAs5wVuM-mPWINaqa) and the [pretrained model](https://drive.google.com/open?id=1kaNWvfIdtbh2GIPNSWXdxqyS-d2DR1F3).

2. Run ``TrainModel.py`` to train a new model on our dataset or Run ``demo_eval.py`` to test the pretrained model on any images.

3. To change the aspect ratio of generated crops, please change the ``generate_bboxes`` function in ``croppingDataset.py`` (line 115).

## Annotation software
The executable annotation software can be found [here](https://github.com/lld533/Grid-Anchor-based-Image-Cropping-Pytorch).

## Other implementation
1. [PyTorch 1.0 or later](https://github.com/lld533/Grid-Anchor-based-Image-Cropping-Pytorch)
2. [Matlab (conference version)](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping)
