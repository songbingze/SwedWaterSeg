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

work list:
- [x] 1. data pipline to mmsegmentation
- [x] 2. data pipline to normalize to 0-255
