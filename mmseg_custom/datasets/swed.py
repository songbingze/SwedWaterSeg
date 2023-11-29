from typing import List
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import os.path as osp
import mmengine
import mmengine.fileio as fileio

@DATASETS.register_module()
class SWEDDataset(BaseSegDataset):
    METAINFO = dict(
        classes= ('background', 'water'),
        palette= [[0], [1]],
    )
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.npy', seg_map_suffix='.npy', **kwargs)

    def load_data_list(self) -> List[dict]:
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        
        _suffix_len = len(self.img_suffix)
        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args):
            data_info = dict(img_path=osp.join(img_dir, img))
            if ann_dir is not None:
                seg_map = img[:-_suffix_len][:61] + "chip" + img[:-_suffix_len][66:] + self.seg_map_suffix
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)
        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
        