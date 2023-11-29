_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/swed.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)