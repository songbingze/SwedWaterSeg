This is a program for water segmentation useing SWED dataset.

dataset link: [SWED](https://openmldata.ukho.gov.uk/)

```
conda install pytorch torchvision
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install ftfy regex
```

Now, its work on HRNet model.
```
CUDA_VISIBLE_DEVICES=0 python tools/train.py config/hrnet/fcn_hr18_4xb2-160k_swed-256x256.py
```