import argparse
import datetime
import random
import time
from pathlib import Path
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import cv2
from dataset import build_dataset
from engine import *
from models import build_model
import os
import warnings

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--threshold', default=0.5,type=float)

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")


    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')

    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--video_path', default='',
                        help='path where the images loaded')                   

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser


def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    video_path = args.video_path
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Could not open video source '{video_path}'")
        return

    sec = 0
    frameRate = 0.05  # captures image every 0.05 seconds
    count = 1
    flag = True
    c = 1
    out = None

    if args.output_dir:
        out_dir = os.path.dirname(args.output_dir)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    resample_mode = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS

    while flag:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if not hasFrames:
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            vidcap.release()
            break

        # Convert OpenCV BGR to RGB for PIL & normalization
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_raw = Image.fromarray(img_rgb)
        width, height = img_raw.size
        if width > 2000 or height > 2000:
            r = width / height
            width = 2000
            height = int(2000 / r)
        new_width = width // 128 * 128
        new_height = height // 128 * 128

        img_raw = img_raw.resize((new_width, new_height), resample_mode)
        if c == 1:
            size = (new_width, new_height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') if args.output_dir.lower().endswith('.mp4') else cv2.VideoWriter_fourcc(*'DIVX')
            out = cv2.VideoWriter(args.output_dir, fourcc, 20, size)
            c += 1

        # pre-processing
        img = transform(img_raw)
        samples = torch.Tensor(img).unsqueeze(0).to(device)

        # run inference
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]

        threshold = args.threshold
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()

        # draw predictions
        size = 2 if new_width < 1500 else 3
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        # overlay count
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_to_draw, f'Count: {len(points)}', (30, 40), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        if out is not None:
            out.write(img_to_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
