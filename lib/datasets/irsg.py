import os
import json
import pickle as pickle

import scipy
import scipy.io as sio
import numpy as np
from IPython import embed

from datasets.imdb import imdb
from fast_rcnn.config import cfg


class irsg(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'IRSG_{}'.format(image_set))
        self._image_set = image_set
        self._devkit_path = (self._get_default_path() if devkit_path is None
                             else devkit_path)
        self.train_image_path = os.path.join(self._devkit_path,
                                             'sg_train_images')
        self.test_image_path = os.path.join(self._devkit_path,
                                            'sg_test_images')
        self._classes = ('__background__', 'man', 'helmet', 'bottle', 'table',
                         'horse', 'picture', 'wall', 'sign', 'building',
                         'bench', 'woman', 'controller', 'phone', 'skateboard',
                         'shoes', 'sign', 'pole', 'laptop', 'monitor',
                         'desk', 'sunglasses')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._roidb_handler = self.rpn_roidb
        train_anno_path = os.path.join(self._devkit_path,
                                       'sg_train_annotations.json')
        test_anno_path = os.path.join(self._devkit_path,
                                      'sg_test_annotations.json')

        with open(train_anno_path) as f:
            print('loading train annotations...')
            self.train_annotations = self._filter_annotations(json.load(f))
        with open(test_anno_path) as f:
            print('loading test annotations...')
            self.test_annotations = self._filter_annotations(json.load(f))

        if image_set is 'train':
            self.annotations = self.train_annotations
            self.image_path = self.train_image_path
            self._image_index = list(range(len(self.train_annotations)))
        elif image_set is 'test':
            self.annotations = self.test_annotations
            self.image_path = self.test_image_path
            self._image_index = list(range(len(self.train_annotations),
                                           len(self.test_annotations)))
        else:
            error_format = ('invalid image set name \'{}\': only'
                            '\'train\' and \'test\' are allowed')
            raise ValueError(error_format.format(image_set))

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        annos, index, image_dir = self._index_to_data(index)
        return os.path.join(image_dir, annos[index]['filename'])

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_annotation(index)
                    for index in self._image_index]
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

    def _filter_annotations(self, annotations):
        result = []
        for entry in annotations:
            entry['objects'] = [obj for obj in entry['objects']
                                if obj['names'][0] in self._classes]
        return annotations

    def _index_to_data(self, index):
        num_train = len(self.train_annotations)
        if index < num_train:
            image_dir = self.train_image_path
            annos = self.train_annotations
        else:
            index -= num_train
            image_dir = self.test_image_path
            annos = self.test_annotations
        return annos, index, image_dir

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

    def _load_annotation(self, index):
        objs = self.annotations[index]['objects']
        image_size = (self.annotations[index]['width'],
                      self.annotations[index]['height'])
        boxes = np.zeros((len(objs), 4), dtype=np.uint16)
        gt_classes = np.zeros(len(objs), dtype=np.int32)
        overlaps = np.zeros((len(objs), self.num_classes), dtype=np.float32)
        for i, obj in enumerate(objs):
            x1 = np.clip(obj['bbox']['x'], a_min=0.0, a_max=image_size[0] - 1)
            y1 = np.clip(obj['bbox']['y'], a_min=0.0, a_max=image_size[1] - 1)
            x2 = np.clip(obj['bbox']['x'] + obj['bbox']['w'], 
                         a_min=0.0, a_max=image_size[0] - 1)
            y2 = np.clip(obj['bbox']['y'] + obj['bbox']['h'],
                         a_min=0.0, a_max=image_size[1] - 1)
            boxes[i, :] = [x1, y1, x2, y2]
            obj_class = self._class_to_ind[obj['names'][0]]
            gt_classes[i] = obj_class
            overlaps[i, obj_class] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

    def evaluate_detections(self, all_boxes, output_dir):
        pass

    def competition_mode(self, on):
        pass


if __name__ == '__main__':
    image_db = irsg('train', None)
    embed()
