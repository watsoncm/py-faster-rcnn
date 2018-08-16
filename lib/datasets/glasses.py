import os
import json
import pickle as pickle

import scipy
import scipy.io as sio
import numpy as np
from IPython import embed

from datasets.imdb import imdb
from fast_rcnn.config import cfg


class glasses(imdb):
    def __init__(self, image_set, devkit_path=None, sg_path=None):
        imdb.__init__(self, 'person_wearing_glasses_{}'.format(image_set))
        self._image_set = image_set
        self._devkit_path = (self._get_default_path()
                             if devkit_path is None else devkit_path)
        self._sg_path = (self._get_default_sg_path()
                         if sg_path is None else sg_path)
        self._classes = ['__background__', 'person', 'glasses']
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._roidb_handler = self.rpn_roidb
 
        train_anno_path = os.path.join(self._devkit_path, '*.json')
        self.train_annotations = []
        for name in sorted(glob.glob(train_anno_path)):
            with open(name) as f:
                self.train_annotations.append(json.load(f))

        test_anno_path = os.path.join(self._sg_path,
                                      'sg_test_annotations.json')
        with open(test_anno_path) as f:
            self.test_annotations = _filter_annotations(json.load(f))

        self.train_size = len(self.train_annotations)
        self.test_size = len(self.test_annotations)
        self.data_size = self.train_size + self.test_size
        if image_set is 'train':
            self._image_index = range(self.train_size)
        elif image_set is 'test':
            self._image_index = range(self.train_size, self.data_size)
        else:
            error_format = ('invalid image set name \'{}\': only'
                            '\'train\' and \'test\' are allowed')
            raise ValueError(error_format.format(image_set))

    def image_path_at(self, i):
        return self.image_path_from_index(self.image_index[i])

    def image_path_from_index(self, index):
        if 0 <= index < self.train_size:
            anno_name = self.train_annotations[index]
        elif self.train_size <= index < self.data_size:
            anno_name = self.test_annotations[index - len(self.train_size)]
        else:
            raise ValueError('index must be between 0 and {}'
                             .format(self.data_size - 1))
        return anno_name.replace('.json', '.jpg')

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def selective_search_roidb(self):
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))
        return roidb

    def rpn_roidb(self):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, None)

    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR, 'sg_dataset')

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        raw_data = sio.loadmat(filename)['boxes'].ravel()
        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

     def _filter_annotations(self, annotations):
        for entry in annotations:
            entry['objects'] = [obj for obj in entry['objects']
                                if obj['names'][0] in self._classes]
        return annotations


    def _load_annotation(self, index):
        if 0 <= index < self.train_size:
            anno = self.train_annotations[index]
            gt_class_list = [obj['desc'] for obj in anno['objects']]
            bbox_list = [obj['box_xywh'] for obj in anno['objects']]
            image_size = (anno['im_w'], anno['im_h'])
        elif self.train_size <= index < self.data_size:
            anno = self.test_annotations[index - self.train_size]
            gt_class_list = [obj['names'][0] for obj in anno['objects']]
            bbox_list = [np.array(obj['bbox']['x'], obj['bbox']['y'],
                                  obj['bbox']['w'], obj['bbox']['h'])
                         for obj in anno['objects']]
            image_size = (anno['width'], anno['height'])

        boxes = np.zeros((len(bbox_list), 4), dtype=np.uint16)
        gt_classes = np.zeros(len(gt_class_list), dtype=np.int32)
        overlaps = np.zeros((len(gt_class_list), self.num_classes), dtype=np.float32)
        for i, (class_name, bbox) in enumerate(zip(gt_class_list, bbox_list)):
            x1 = np.clip(bbox[0], a_min=0.0, a_max=image_size[0] - 1)
            y1 = np.clip(bbox[1], a_min=0.0, a_max=image_size[1] - 1)
            x2 = np.clip(bbox[0] + bbox[2], a_min=0.0, a_max=image_size[0] - 1)
            y2 = np.clip(bbox[1] + bbox[3], a_min=0.0, a_max=image_size[1] - 1)
            boxes[i, :] = [x1, y1, x2, y2]
            obj_class = self._class_to_ind[class_name]
            gt_classes[i] = obj_class
            overlaps[i, obj_class] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

    def evaluate_detections(self, all_boxes, output_dir):
        for class_index, class_name in enumerate(self.classes):
            if class_name == '__background__':
                continue
            print('Writing {} results file'.format(class_name))
            output_name = '{}_results.csv'.format(class_name)
            output_path = os.path.join(output_dir, output_name)
            with open(output_path, 'w') as f:
                for image_index_index, index in enumerate(self.image_index):
                    dets = all_boxes[class_index][image_index_index]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{} {:.6f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))

    def competition_mode(self, on):
        pass  # guess we're not competing today
