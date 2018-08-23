import os
import sys
import json
import glob
import hashlib


def get_hash(image_path):
    h = hashlib.md5()
    with open(image_path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

sg_path = 'sg_dataset'
sg_anno_path = os.path.join(sg_path, 'sg_test_annotations.json')
sg_test_path = os.path.join(sg_path, 'sg_test_images')

if len(sys.argv) <= 1:
    print('usage: {} situation-name'.format(sys.argv[0]))
    sys.exit()

sit_name = sys.argv[1]
sit_path = 'situate'
sit_pos_path = os.path.join(sit_path, '{}-test'.format(sit_name))
sit_neg_path = os.path.join(sit_path, '{}-negative'.format(sit_name))

with open(sg_anno_path) as f:
    annos = json.load(f)

image_to_anno = {}
for anno in annos:
    image_to_anno[anno['filename']] = anno
        
hash_to_image = {}
for image_path in glob.glob(os.path.join(sg_test_path, '*.jpg')):
    hash_to_image[get_hash(image_path)] = os.path.basename(image_path)

for path in (sit_pos_path, sit_neg_path):
    annos = []
    for image_path in glob.glob(os.path.join(path, '*.jpg')):
        anno = image_to_anno[hash_to_image[get_hash(image_path)]]  
        anno['filename'] = os.path.basename(image_path)
        annos.append(anno)
    with open(os.path.join(path, 'annotations.json'), 'w') as f:
        json.dump(annos, f)

