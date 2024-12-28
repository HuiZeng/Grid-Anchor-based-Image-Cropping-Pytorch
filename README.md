# Grid-Anchor-based-Image-Cropping-Pytorch
The extension of this work has been accepted by TPAMI. Please read the [paper](https://www4.comp.polyu.edu.hk/~cslzhang/paper/GAIC-PAMI.pdf) for details.


### Requirements
python 2.7, pytorch 0.4.1, numpy, cv2, scipy. 

### Usage

1. Download the source code, the datasets [[conference version](https://drive.google.com/file/d/1KhmyjoimsQVXqPnLjKZiU4iXNKNyyxqW/view?usp=sharing)], [[journal version](https://drive.google.com/file/d/1tDdQqDe8dMoMIVi9Z0WWI5vtRViy01nR/view?usp=sharing)] and the pretrained models [[conference version](https://drive.google.com/file/d/1OvLT_ul17zCK4ljAi4myGAgA50PmLy3Y/view?usp=sharing)] [[journal version](https://drive.google.com/file/d/1KWYQdL6R5hmOC9toTymbMORZDThpiEW4/view?usp=sharing)]

2. Run ``TrainModel.py`` to train a new model on our dataset or Run ``demo_eval.py`` to test the pretrained model on any images.

3. To change the aspect ratio of generated crops, please change the ``generate_bboxes`` function in ``croppingDataset.py`` (line 115).

### Annotation software
The executable annotation software can be found [here](https://github.com/lld533/Grid-Anchor-based-Image-Cropping-Pytorch).

### Other implementation
1. [PyTorch 1.0 or later](https://github.com/lld533/Grid-Anchor-based-Image-Cropping-Pytorch)
2. [Matlab (conference version)](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping)


### Citation
```
@inproceedings{zhang2019deep,
  title={Reliable and Efficient Image Cropping: A Grid Anchor based Approach},
  author={Zeng, Hui, Li, Lida， Cao, Zisheng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
@article{zeng2020cropping,
  title={Grid Anchor based Image Cropping: A New Benchmark and An Efficient Model},
  author={Zeng, Hui and Li, Lida and Cao, Zisheng and Zhang, Lei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={},
  number={},
  pages={},
  year={2020},
  publisher={IEEE}
}
```
