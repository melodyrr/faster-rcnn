import os
import xml.etree.ElementTree as ET

import numpy as np
import util



class DOTABboxDataset:
    """Bounding box dataset for PASCAL `VOC`_."""

    def __init__(self, data_dir, split='train', hbb=True, 
                 use_difficult=False, return_difficult=False):


        id_list_file = os.listdir(os.path.join(data_dir,'{0}/hbbxml/'.format(split)))
        #print(os.path.join(data_dir,'{0}/hbbxml/'.format(split)))

        self.ids = [id_.split('.')[0] for id_ in id_list_file]
        self.data_dir = data_dir
        self.split = split
        self.is_hbb = hbb
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = DOTAv10_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example."""
        id_ = self.ids[i]
        if self.is_hbb:
            anno = ET.parse(os.path.join(self.data_dir, self.split, 'hbbxml', id_ + '.xml'))
        else:  
            anno = ET.parse(os.path.join(self.data_dir, self.split, 'obbxml', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in [('ymin', 'xmin', 'ymax', 'xmax'), ('y0', 'x0', 'y1', 'x1', 'y2', 'x2', 'y3', 'x3')][not self.is_hbb]])
            name = obj.find('name').text.lower().strip()
            label.append(DOTAv10_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, self.split, 'images', id_ + '.png')
        img = util.read_image(img_file, color=True)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    __getitem__ = get_example


DOTAv10_BBOX_LABEL_NAMES = (
    'plane',
    'baseball-diamond',
    'bridge',
    'ground-track-field',
    'small-vehicle',
    'large-vehicle',
    'ship',
    'tennis-court',
    'basketball-court',
    'storage-tank',
    'soccer-ball-field',
    'roundabout',
    'harbor',
    'swimming-pool',
    'helicopter'
)

