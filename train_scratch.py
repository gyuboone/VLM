import torch
import torch.nn.functional as F
import numpy as np
import random
import os

from dataloader.dataset import CLIP_COCO_dataset
from dataloader.data_loaders import get_dataloader

from CLIP import CLIP
from utils.simple_tokenizer import SimpleTokenizer
from utils import set_seed, mkdir

import time

from torch.optim import Adam, AdamW # both are same but AdamW has a default weight decay
import gc
gc.collect()
torch.cuda.empty_cache()

################## setting start ##################
lr = 0.001
batch_size = 96
epochs = 35


# fixing seed
seed = 17
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# text tokenizer (text encoder에 사용)
tokenizer = SimpleTokenizer()


# setting model
model_params = {'embed_dim' : 1024,
  'image_resolution' : 224,
  'vision_layers' : [3, 4, 6, 3],
  'vision_width': 64,
  'vision_patch_size' : 0 ,# ideally it should be none
  'context_length' : 77,
  'vocab_size' : 49408,
  'transformer_width' : 512,
  'transformer_heads' : 8,
  'transformer_layers' : 6,# 12 in CLIP
}
model_params['vision_layers'] = tuple(model_params['vision_layers'])
model_params['vision_patch_size'] = None
model = CLIP(**model_params).to(device)


# setting dataset
train_img_dir = 'data/mscoco/train2017'
train_annotation_file = 'data/mscoco/annotations/captions_train2017.json'

train_dataset = CLIP_COCO_dataset(train_annotation_file, train_img_dir, tokenizer)
train_dataloader = get_dataloader(train_dataset, batch_size, is_train=True)

# setting optimizer
optimizer = AdamW(model.parameters(), lr=lr)


# setting loss function
def loss_function(logits_img, logits_txt):

    labels = torch.arange(logits_img.shape[0]).to(device)

    loss_i = F.cross_entropy(logits_img, labels)
    loss_t = F.cross_entropy(logits_txt, labels)
    return (loss_i + loss_t) / 2

################## setting end ##################




################ training epoch start #################
start = time.time()

for epoch in range(epochs):
    print(f"{epoch}th epoch starting.")
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        img, txt = batch

        img = img.to(device)
        txt = txt.to(device)

        logits_img, logits_txt = model(img, txt)

        optimizer.zero_grad()
        loss = loss_function(logits_img,logits_txt)
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()

    print(f"[Epoch {epoch}th] loss: {running_loss/len(train_dataloader):.4f}")
end = time.time()
################ training epoch end #################
print(f"Time ellapsed in training is: {end-start}")