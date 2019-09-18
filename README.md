# Grid-Anchor-based-Image-Cropping-Pytorch
This code includes several extensions we have made to our conference version. Please read the [paper](https://drive.google.com/open?id=1Bd1VaqYVycB7Npv5OdXKl-znKs_APl4n) for details.


### Requirements
python 2.7, pytorch 0.4.2, numpy, cv2, scipy. 

### Usage
1. Download the source code, the [dataset](https://drive.google.com/open?id=1X9xK5O9cx4_MvDkWAs5wVuM-mPWINaqa) and the [pretrained model](https://drive.google.com/open?id=1kaNWvfIdtbh2GIPNSWXdxqyS-d2DR1F3).

2. Run TrainModel.py to train a new model on our dataset or Run demo_eval.py to test the pretrained model on any images.

To change the aspect ratio of generated crop, please change the generate_bboxes function in croppingDataset.py (line 115).

