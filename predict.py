import argparse
import numpy as np
import pandas as pd
import torch
import utils
import os, json
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from engine import train_one_epoch
import utils
import transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import tensor
import utils
import transforms as T
from pprint import pprint



class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = T.ToTensor()
        self.example_filenames = sorted(os.listdir(os.path.join(self.root)))
        files, images, bboxs, masks = [], [], [], []
        
    def __getitem__(self, idx):
        filename = self.example_filenames[idx]
        image_id, _, _ = filename.strip(".jpg").split("__")
            
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        bbox = torch.as_tensor([[0, 0, 1, 1], ], dtype=torch.float32)
        labels = torch.as_tensor([False, ], dtype=torch.int64) + 1


        image_id = torch.tensor([idx])

        target = {}
        target["filename"] = filename
        target["image_id"] = image_id
        target["boxes"] = bbox
        target["labels"] = labels
        
        img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.example_filenames)



def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        res_list = []

        for images, targets in tqdm(data_loader):
            images = list(img.to(device) for img in images)
            
            targets = list({k: v.to(device) if k not in ['image_id', 'filename'] else v 
                            for k, v in target.items()} 
                           for target in targets)

            outputs = model(images, targets)

            # image_id: res, image_id: tru
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            
            for k, output in res.items():  # complete missing bbox
                if output['boxes'].nelement() == 0:
                    res[k] = {'boxes': torch.tensor([[0, 0, 200, 200]]).to(device), 
                              'labels': torch.tensor([1]).to(device), 'scores': torch.tensor([1]).to(device)}

            for image_id, output in res.items():  # for each image
                max_score_indx = output['scores'].argmax()  # select detection with highest score
                res[image_id]['boxes'] = output['boxes'][max_score_indx]
                res[image_id]['labels'] = output['labels'][max_score_indx]
                res[image_id]['scores'] = output['scores'][max_score_indx]

                # to x, y, w, h
                res[image_id]['boxes'][2] = res[image_id]['boxes'][2] - res[image_id]['boxes'][0]
                res[image_id]['boxes'][3] = res[image_id]['boxes'][3] - res[image_id]['boxes'][1]
                
                
            for image_id, image_res in res.items():
                for target in targets:
                    if target['image_id'] == image_id:
                        cor_filename = target['filename']
                image_res['boxes'] = [val.item() for val in image_res['boxes']]
                image_res['labels'] = (image_res['labels'] == 2).item()
                res_list.append(dict({'filename': cor_filename, 'proper_mask': image_res['labels']},
                                    **dict(zip(['x', 'y', 'w', 'h'], image_res['boxes']))))

    pd.DataFrame(res_list).to_csv("prediction.csv", index=False, header=True)



# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = MaskDataset(args.input_folder)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True,
    collate_fn=utils.collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1RQBkbPHDZp5T2Iu0nR4mwwyWGzA_SXvv', dest_path='./model_9',showsize=True)

model.load_state_dict(torch.load('model_9'))
model.to(device)



evaluate(model, data_loader)

