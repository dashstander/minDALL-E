# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import sys
import argparse
import polars as pl
import clip
import numpy as np
from PIL import Image
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dalle.models import Dalle
from dalle.utils.utils import set_seed, clip_score


def clean_text(text):
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r' ', '_', text)
    return text

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_candidates', type=int, default=16)
parser.add_argument('--prompt', type=str, default='A painting of a tree on the ocean')
parser.add_argument('--softmax-temperature', type=float, default=1.0)
parser.add_argument('--top-k', type=int, default=256)
parser.add_argument('--top-p', type=float, default=None, help='0.0 <= top-p <= 1.0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--group', type=int)

args = parser.parse_args()

# Setup
assert args.top_k <= 256, "It is recommended that top_k is set lower than 256."

set_seed(args.seed)
device = 'cuda:0'
model = Dalle.from_pretrained('minDALL-E/1.3B')  # This will automatically download the pretrained model.
model.to(device=device)

with open('prompts_specific.txt') as txtfile:
    data = [line.strip() for line in txtfile.readlines()]

#data = pl.read_parquet('index.parquet').filter(pl.col('group') == args.group)

for i, caption in enumerate(data.records()):
    # Sampling
    images = model.sampling(
        prompt=caption,
        top_k=args.top_k,
        top_p=args.top_p,
        softmax_temperature=args.softmax_temperature,
        num_candidates=args.num_candidates,
        device=device
    ).cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    # CLIP Re-ranking
    model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
    model_clip.to(device=device)
    rank = clip_score(
        prompt=args.prompt,
        images=images,
        model_clip=model_clip,
        preprocess_clip=preprocess_clip,
        device=device
    )

    # Save images
    images = images[rank]
    clean_caption = clean_text(caption)
    print(rank, images.shape)
    if not os.path.exists(f'./figures/{clean_caption}'):
        os.makedirs(f'./figures/{clean_caption}', exist_ok=True)
    for j in range(min(16, args.num_candidates)):
        if j > 0:
            break
        im = Image.fromarray((images[i]*255).astype(np.uint8))
        im.save(f'./figures/{clean_caption}/{i}.png')
    with open(f'figures/{clean_caption}/{i}.txt') as txtfile:
        txtfile.write(caption)
