from tqdm import tqdm
import os
import shutil
import argparse

import pandas as pd
import cv2

import yaml

FISH_NAMES = ['Albacore', 'Bigeye tuna', 'Blue marlin', 'Escolar', 'Great barracuda', 
            'Long snouted lancetfish', 'Mahi mahi', 'Opah', 'Shark', 'Shortbill spearfish', 
            'Skipjack tuna', 'Striped marlin', 'Swordfish', 'Tuna', 'Wahoo', 'Yellowfin tuna']

RANDOM_STATE = 42

CONFIG = dict(
    train = "",
    val = "",
    test = "",
    is_coco = False,
    nc = len(FISH_NAMES),
    names = FISH_NAMES
)

def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def gen_data(df, images_source_path, output_path, split):

    images_path = os.path.join(output_path, 'images', split)
    labels_path = os.path.join(output_path, 'labels', split)

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
 
    columns = ["img_id", "x_min", "x_max", "y_min", "y_max", "label_l1"]
    df = df[columns]

    img_ids = df.img_id.unique().tolist()

    for id in tqdm(img_ids, desc=f"Generate data {split}"):
        detail = df[df.img_id == id]

        filename = f'{id}.jpg'

        file = cv2.imread(os.path.join(images_source_path, filename), cv2.IMREAD_GRAYSCALE)
        h, w = file.shape

        with open(os.path.join(labels_path, f'{id}.txt'), 'w') as f:
            for img_id, xmin, xmax, ymin, ymax, label in detail.values:
                bbox = convert_box((w, h), [xmin, xmax, ymin, ymax])
                # print(img_id, xmin, xmax, ymin, ymax, label)
                f.write(" ".join([str(a) for a in (FISH_NAMES.index(label), *bbox)]) + '\n')

        ori_image_path = os.path.join(images_source_path, filename)
        new_image_path = os.path.join(images_path, filename)
        shutil.copy(ori_image_path, new_image_path)
    
    CONFIG[split] = images_path

def main(args):
    data_path = args.data_path
    image_path = os.path.join(data_path, 'images')

    output_path = args.output_path

    if output_path is None:
        output_path = f'{image_path}_result'

    os.makedirs(output_path, exist_ok=True)

    csv_path = os.path.join(data_path, args.csv_name)
    df = pd.read_csv(csv_path)

    # FISH_NAMES = df.label_l1.unique().tolist()
    # FISH_NAMES.sort()
    # print(FISH_NAMES)
    # CONFIG['names'] = FISH_NAMES
    # CONFIG['nc'] = len(FISH_NAMES)

    split = ['train', 'val', 'test']

    print("Generating data ...")
    for s in split:
        gen_data(df[df[s]], image_path, output_path, s)
    print("Generate data success!")

    with open(r'./data/fishnet.yaml', 'w') as file:
        conf = yaml.dump(CONFIG, file)
        print(conf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path', type=str, help='Data path to images', required=True)
    parser.add_argument('-o', '--output_path', type=str, default=None, help='Output data path to images')
    parser.add_argument('--csv-name', type=str, default='foid_labels_v100.csv', help='CSV filename')

    args = parser.parse_args()
    print(args)

    main(args)
