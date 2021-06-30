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
from torch import nn


class CustomResnet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=False, transform_input=False, final_pooling=None):
        super(CustomResnet, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Linear(512 * BasicBlock.expansion, 2)
        self.bb = nn.Linear(512 * BasicBlock.expansion, 4)

        
    def forward(self, x):
        x = self.resnet(x)
        
        return self.classifier(x), self.bb(x)



class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.example_filenames = sorted(os.listdir(os.path.join('data', self.root)))
        files, images, bboxs, masks = [], [], [], []
        
    def __getitem__(self, idx):
        filename = self.example_filenames[idx]
        image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
            
        bbox = json.loads(bbox)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]

        proper_mask = True if proper_mask.lower() == "true" else False

        img = Image.open(os.path.join('data', self.root, filename)).convert("RGB")

        bbox = torch.as_tensor([bbox, ], dtype=torch.float32)
        labels = torch.as_tensor([proper_mask, ], dtype=torch.int64) + 1
        
        image_id = torch.tensor([idx])
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = bbox
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
            
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


        files, images, bboxs, masks = map(tensor, [files, images, bboxs, masks])
        map(lambda x: print(type(x), x.shape), [files, images, bboxs, masks])

        return files, images, bboxs, masks

    def __len__(self):
        return len(self.example_filenames)


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip())
        transforms.append(T.RandomIoUCrop())
        transforms.append(T.RandomPhotometricDistort())
    return T.Compose(transforms)


def plot(res_dict, name):
    plt.plot(res_dict["train_iou"], label="train iou")
    plt.plot(res_dict["test_iou"], label="test iou")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.savefig(f'results/{name}/iou.png')
    open(f'results/{name}/train_iou.txt', 'w').write(str(res_dict["train_iou"]))
    open(f'results/{name}/test_iou.txt', 'w').write(str(res_dict["test_iou"]))
    
    plt.clf()
    plt.plot(res_dict["train_acc"], label="train acc")
    plt.plot(res_dict["test_acc"], label="test acc")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.savefig(f'results/{name}/acc.png')
    open(f'results/{name}/train_acc.txt', 'w').write(str(res_dict["train_acc"]))
    open(f'results/{name}/test_acc.txt', 'w').write(str(res_dict["test_acc"]))
    
    plt.clf()
    plt.plot(res_dict["train_loss"], label="train loss")
    plt.plot(res_dict["test_loss"], label="test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.savefig(f'results/{name}/loss.png')
    plt.clf()
    open(f'results/{name}/train_loss.txt', 'w').write(str(res_dict["train_loss"]))
    open(f'results/{name}/test_loss.txt', 'w').write(str(res_dict["test_loss"]))


def iou(bbox_a, bbox_b):
    x1, y1, w1, h1 = list(bbox_a)
    x2, y2, w2, h2 = list(bbox_b)
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0.0 or h_intersection <= 0.0:  # No overlap
        return 0.0
    intersection = w_intersection * h_intersection
    union = w1 * h1 + w2 * h2 - intersection    # Union = Total Area - Intersection
    return intersection / union


def evaluate(model, dataloader, res_dict, t):
    model.eval()
    with torch.no_grad():
        iou_sum = 0
        acc_sum = 0
        count = 0
        loss_sum = 0

        for images, targets in tqdm(dataloader):
            images = list(img.to(device) for img in images)            
            targets = list({k: v.to(device) for k, v in target.items()} for target in targets)

            loss_dict, outputs = model(images, targets)
            loss_sum += sum(loss for loss in loss_dict.values()).item()

            # image_id: res, image_id: tru
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            tru = {target["image_id"].item(): {"boxes": target["boxes"][0], 
                                               "labels": target["labels"][0]} 
                                               for target in targets}
            
            for k, output in res.items():  # complete missing bbox
                if output['boxes'].nelement() == 0:
                    res[k] = {'boxes': torch.tensor([[0, 0, 0, 0]]).to(device), 
                              'labels': torch.tensor([1]).to(device), 'scores': torch.tensor([1]).to(device)}

            for image_id, output in res.items():  # for each image
                max_score_indx = output['scores'].argmax()  # select detection with highest score
                res[image_id]['boxes'] = output['boxes'][max_score_indx]
                res[image_id]['labels'] = output['labels'][max_score_indx]
                res[image_id]['scores'] = output['scores'][max_score_indx]

                # to x, y, w, h
                res[image_id]['boxes'][2] = res[image_id]['boxes'][2] - res[image_id]['boxes'][0]
                res[image_id]['boxes'][3] = res[image_id]['boxes'][3] - res[image_id]['boxes'][1]

                tru[image_id]['boxes'][2] = tru[image_id]['boxes'][2] - tru[image_id]['boxes'][0]
                tru[image_id]['boxes'][3] = tru[image_id]['boxes'][3] - tru[image_id]['boxes'][1]
                
            if res.keys() != tru.keys():
                print('WARNING!! different keys...')
                
            for image_id in res.keys():
                iou_sum += iou(res[image_id]['boxes'], tru[image_id]['boxes'])
                acc_sum += res[image_id]['labels'] == tru[image_id]['labels'].item()

            count += len(res)
            
#             if count > 100:
#                 break

        print('acc', t, acc_sum, count)
        print('iou', t, iou_sum, count)
        print('los', t, loss_sum, count)

        res_dict[f"{t}_acc"].append(acc_sum / count)
        res_dict[f"{t}_iou"].append(iou_sum / count)
        res_dict[f"{t}_loss"].append(2 * loss_sum / count)
#     print(results)

    return res_dict

# TODO: left-est bbox


# use our dataset and defined transformations
dataset = MaskDataset('train', get_transform(train=True))
dataset_vali = MaskDataset('train', get_transform(train=False))
dataset_test = MaskDataset('test', get_transform(train=False))


data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True,
    collate_fn=utils.collate_fn)

data_loader_vali = torch.utils.data.DataLoader(
    dataset_vali, batch_size=2, shuffle=False,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False,
    collate_fn=utils.collate_fn)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


res_dict = {"train_loss":[], "test_loss":[], "train_iou":[], "test_iou":[], "train_acc":[], "test_acc":[]}

# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    print('train loss', train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=500))
    lr_scheduler.step()

    torch.save(model.state_dict(), f"results/model_faster_test/model_{epoch}")

#     evaluate(model, data_loader_test, res_dict, 'test')
#     evaluate(model, data_loader_vali, res_dict, 'train')
    print(res_dict)
    
    plot(res_dict, 'model_faster_test')

