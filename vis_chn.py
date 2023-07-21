import os, sys
import torch
import numpy as np

from models.ests import build_ests
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from util import box_ops
from PIL import Image
import datasets.transforms as T

import pickle
with open('chn_cls_list.txt', 'rb') as fp:
    CTLABELS = pickle.load(fp)

def _decode_recognition(rec):
    s = ''
    for c in rec:
        c = int(c)
        if c < 5461:
            s += str(chr(CTLABELS[c]))
        elif c == 5462:
            s += u''
    return s
    
def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    args.device = 'cuda'
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

model_config_path = "config/ESTS/ESTS_4scale.py" # change the path of the model config file
model_checkpoint_path = "checkpoint0090.pth" # change the path of the model checkpoint

args = SLConfig.fromfile(model_config_path) 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
model.cuda()
transform = T.Compose([
    T.RandomResize([1000],max_size=1100),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
)
image_dir = '../test/'
dir = os.listdir(image_dir)
for idx, i in enumerate(dir):
    image = Image.open(image_dir + i).convert('RGB')
    image, _ = transform(image,None)
    output = model(image[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]))[0]
    rec = [_decode_recognition(i) for i in output['rec']]
    thershold = 0.2 # set a thershold
    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold
    recs = []
    for i,r in zip(select_mask,rec):
        if i:
            recs.append(r)
    vslzr = COCOVisualizer()
    # box_label = ['text' for item in rec[select_mask]]
    pred_dict = {
        'boxes': boxes[select_mask],
        'size': torch.tensor([image.shape[1],image.shape[2]]),
        'box_label': recs,
        'image_id': idx,
        'beziers': output['beziers'][select_mask]
    }
    vslzr.visualize(image, pred_dict, savedir='vis_fin')