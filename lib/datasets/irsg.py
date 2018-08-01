import os
import json

from datasets.imdb import imdb


class irsg(imdb):
    def __init__(self, image_set, devkit_path):
        imdb.__init__(self, 'IRSG_{}'.format(image_set))
        self._image_set = image_set
        self.train_image_path = os.path.join(devkit_path, 'sg_train_images')
        self.test_image_path = os.path.join(devkit_path, 'sg_test_images')
        train_anno_path = os.path.join(devkit_path, 'train_annotations.json')
        test_anno_path = os.path.join(devkit_path, 'test_annotations.json')

        with open(train_anno_path) as f:
            self.train_annotations = json.load(f)
        with open(test_anno_path) as f:
            self.test_annotations = json.load(f)

        if image_set is 'train':
            self.annotations = self.train_annotations
            self.image_path = self.train_image_path
        elif image_set is 'test':
            self.annotations = self.test_annotations
            self.image_path = self.test_image_path
        else:
            error_format = ('invalid image set name \'{}\': only'
                            '\'train\' and \'test\' are allowed')
            raise ValueError(error_format.format(image_set))

    def image_path_at(self, i):
        return os.path.join(self.image_path, self.annotations[i]['filename'])

    def image_path_from_index(self, index):
        num_train = len(self.train_annotations)
        if index < num_train:
            image_dir = self.train_image_path
            annos = self.train_annotations
        else:
            index -= num_train
            image_dir = self.test_image_path
            annos = self.test_annotations
        return os.path.join(image_dir, annos[index]['filename'])

    def gt_roidb(self):
        pass

    def selective_search_roidb(self):
        pass

    def rpn_roidb(self):
        pass

    def competition_mode(self, on):
        pass


if __name__ == '__main__':
    image_db = irsg('train', None)
    print(image_db)
