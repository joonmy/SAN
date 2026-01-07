from ctypes import util
from cv2 import IMREAD_GRAYSCALE
import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
import math
from torchvision import transforms
from PIL import Image
import cv2
import os
import random
import numpy as np
import lmdb
import io
import time
from vidaug import augmentors as va
from augmentation import *
import pickle
from loguru import logger
from hpman.m import _
from textaugment import EDA
# global definition
from definition import *

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        if isinstance(Image, PIL.Image.Image):
            Image = np.asarray(Image, dtype=np.uint8)
        new_video_x = (Image - 127.5) / 128
        return new_video_x

class SomeOf(object):
    """
    Selects one augmentation from a list.
    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
    """

    def __init__(self, transforms1, transforms2):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, clip):
        select = random.choice([0, 1, 2])
        if select == 0:
            return clip
        elif select == 1:
            if random.random() > 0.5:
                return self.transforms1(clip)
            else:
                return self.transforms2(clip)
        else:
            clip = self.transforms1(clip)
            clip = self.transforms2(clip)
            return clip

class S2T_Dataset(Dataset.Dataset):
    def __init__(self,path,tokenizer,config,args,phase, training_refurbish=False):
        self.config = config
        self.args = args
        self.training_refurbish = training_refurbish
        
        self.raw_data = utils.load_dataset_file(path)
        self.tokenizer = tokenizer
        self.img_path = config['data']['img_path']
        self.phase = phase
        self.max_length = config['data']['max_length']
        
        self.list = [key for key,value in self.raw_data.items()]   
        self.emb = EDA()
        self.hard_negative_table = utils.load_dataset_file(args.neg_table_name)

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]
        tgt_sample = sample['text']
        name_sample = sample['name']

        feature, valid_length = self.load_features(name_sample)

        return name_sample, feature, valid_length, tgt_sample

    def load_features(self, name_sample):
        paths = os.path.join(self.img_path, self.phase, name_sample+'.pkl')
        with open (paths, 'rb') as f:
            video_feat = pickle.load(f)
        
        feature = video_feat['feature']  
        feature_len = 64  
        
        # follow the Cico
        video_len = len(feature)
        if video_len >= feature_len:
            indices = np.linspace(0, video_len-1, feature_len, dtype=int)
            sampled_feature = torch.from_numpy(feature[indices]).float() 
            valid_mask = torch.ones((feature_len,), dtype=torch.int)
        else:
            sampled_feature = torch.zeros((feature_len, feature.shape[1]), dtype=torch.float) 
            sampled_feature[:video_len] = torch.from_numpy(feature).float()
            valid_mask = torch.zeros((feature_len,), dtype=torch.int)
            valid_mask[:video_len] = 1

        return sampled_feature, valid_mask
    
    def generate_hard_negatives(
        self,
        tgt_batch,
        hard_negative_table,
        num_negatives=4
    ):
        """
        tgt_batch: List[str]
        hard_negative_table: Dict[str, List[str]]
        return: List[List[str]]  # B x num_negatives
        """

        hard_tgt_batch = []

        for sentence in tgt_batch:
            words = sentence.split()

            negatives_for_sentence = []
            candidate_words = [i for i in words if i in hard_negative_table]

            if len(candidate_words) == 0:
                pool = [s for s in tgt_batch if s != sentence]
                hard_tgt_batch.append(random.choices(
                    pool,
                    k=num_negatives
                ))
                continue

            for _ in range(num_negatives):
                k = min(2, len(candidate_words))
                ws = random.sample(candidate_words, k=k)

                replace = {
                    w: random.choice(list(hard_negative_table[w]))
                    for w in ws
                }
                negatives_for_sentence.append(
                    " ".join(replace.get(word, word) for word in words)
                )

            hard_tgt_batch.append(negatives_for_sentence)

        return hard_tgt_batch
    
    def collate_fn(self,batch):
        
        tgt_batch, feature_batch, feature_mask, name_batch = [],[],[],[]

        for name_sample, feature, valid_mask, tgt_sample in batch:

            name_batch.append(name_sample)

            feature_batch.append(feature)

            feature_mask.append(valid_mask)
            
            tgt_batch.append(tgt_sample)


        feature_batch = torch.stack(feature_batch, dim=0)  # [B, 64, 1024]
        feature_mask_batch = torch.stack(feature_mask, dim=0)  # [B, 64]

        if self.phase == 'train':
            hard_tgt_batch = self.generate_hard_negatives(
                tgt_batch,
                self.hard_negative_table,
                num_negatives=5
            )
            flat_hard_tgt_batch = [neg for sublist in hard_tgt_batch for neg in sublist] # B x num_negatives
        else:
            flat_hard_tgt_batch = []

        emb_aug = self.emb.random_swap
        if random.random() > 0.5 and self.phase == 'train':
            tgt_batch = [emb_aug(sentence) for sentence in tgt_batch]
        
        total_tgt_batch = tgt_batch + flat_hard_tgt_batch # (B + B x num_negatives)
    
        if self.phase == 'train':
            tgt_input = self.tokenizer(total_tgt_batch, return_tensors="pt",padding = "max_length",  truncation=True, add_special_tokens=True, max_length=70) # 58 for non-hardneg
        if self.phase == 'test':
            tgt_input = self.tokenizer(total_tgt_batch, return_tensors="pt",padding = "max_length",  truncation=True, add_special_tokens=True, max_length=44)

        src_input = {}
        src_input['features'] = feature_batch
        src_input['features_valid_mask'] = feature_mask_batch
        src_input['name_batch'] = name_batch

        return src_input, tgt_input

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'






