import cv2
import csv
import os
from pathlib import Path

import torch
# import clip
from CLIP import clip
from PIL import Image
import numpy as np

def get_KeyframeDict(csv_path='./data/star/Video_Keyframe_IDs.csv'):

    with open(csv_path) as f:
        csvreader = csv.reader(f)
        rows = []
        for row in csvreader:
            rows.append(row)
    
    KeyframeSets = {}
    KeyframeDict = {}
    for row in rows[1:]:

        list_id = row[2][1:-1].replace('\'', '').split(',')

        KeyframeDict[row[0]] = (row[1], list_id)

        if row[1] not in KeyframeSets.keys():
            KeyframeSets[row[1]] = set()
        for idx in list_id:
            KeyframeSets[row[1]].add(int(idx))

    return KeyframeDict, KeyframeSets

    return video_id, Keyframe_IDs

def extract_all_Keyframe():

    _, KeyframeSets = get_KeyframeDict()

    for video_id in KeyframeSets.keys():

        print(video_id)

        new_path = os.path.join('./data/star/Frames', video_id)
        Path(new_path).mkdir(parents=True, exist_ok=True)

        filename = video_id + '.mp4'

        vidcap = cv2.VideoCapture(os.path.join('./data/star/Charades_v1_480', filename))
        success, image = vidcap.read()
        count = 0
        while success:
            if count in KeyframeSets[video_id]:
                tmp = "{:06d}.png".format(count)
                cv2.imwrite(os.path.join(new_path, tmp), image)     # save frame as JPEG file      
            success,image = vidcap.read()
            count += 1

def sample_frames():

    _, KeyframeSets = get_KeyframeDict()

    id_count = 0

    for video_id in KeyframeSets.keys():

        if id_count % 100 == 0:

            print(id_count)

        # print(video_id)

        new_path = os.path.join('./data/star/Residual_Frames', video_id)
        Path(new_path).mkdir(parents=True, exist_ok=True)

        filename = video_id + '.mp4'

        frames = []

        vidcap = cv2.VideoCapture(os.path.join('./data/star/Charades_v1_480', filename))
        fps = vidcap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = round(frame_count/fps)

        success, image = vidcap.read()
        count = 0
        while success:
            # if count in KeyframeSets[video_id]:
                # tmp = "{:06d}.png".format(count)
                # cv2.imwrite(os.path.join(new_path, tmp), image)     # save frame as JPEG file     
            frames.append(image)
            success,image = vidcap.read()
            count += 1

        residual = []
        for i in range(len(frames)-1):
            frame1 = frames[i]
            frame2 = frames[i+1]
            ms_residual = np.sum((frame2-frame1)**2)
            residual.append(ms_residual)
        total_residual = np.sum(residual)

        count = 0
        sampled_ids = []
        accum_residual = 0
        while len(sampled_ids) < duration and count < len(residual):
            threshold = total_residual / (duration - len(sampled_ids))
            accum_residual += residual[count]
            if accum_residual >= threshold:
                sampled_ids.append(count)
                total_residual -= accum_residual
                accum_residual = 0
            count += 1

        # print(video_id, duration, len(sampled_ids))
        # if duration != len(sampled_ids):
        #     print('{}, frame count not aligned with duration! duration: {} and frame: {}'.format(video_id, duration, len(sampled_ids)))

        for sampled_id in sampled_ids:
            tmp = "{:06d}.png".format(sampled_id)
            cv2.imwrite(os.path.join(new_path, tmp), frames[sampled_id])

        id_count += 1

def get_all_clip_feature(image_folder, output_ckpt):

    _, KeyframeSets = get_KeyframeDict()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    Keyframe_features = {}
    count = 0
    
    for video_id in KeyframeSets.keys():

        Keyframe_features[video_id] = {}

        if count%100 == 0:
            print(count)

        folder_path = os.path.join(image_folder, video_id)
        frame_ids = sorted(os.listdir(folder_path))
        
        for idx in frame_ids:
            image_path = os.path.join(folder_path, idx)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)

            Keyframe_features[video_id][idx] = image_features

        count += 1
    
    torch.save(Keyframe_features, output_ckpt)

def get_patch_clip_feature(image_folder, output_ckpt):

    _, KeyframeSets = get_KeyframeDict()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    
    split_storage = 10
    video_count_for_single_storage = len(KeyframeSets.keys()) // 10

    for k in range(split_storage):

        print(k)

        features = {}

        start = k * video_count_for_single_storage
        if k < split_storage-1:
            end = (k+1) * video_count_for_single_storage
        else:
            end = len(KeyframeSets.keys())

        for i, video_id in enumerate(sorted(KeyframeSets.keys())):

            if i >= start and i < end:

                filename = video_id + '.mp4'
                frames = []
                vidcap = cv2.VideoCapture(os.path.join('./data/star/Charades_v1_480', filename))
                use_count = 0
                success, image = vidcap.read()
                while success:
                    if use_count == 0:
                        frames.append(image)
                        use_count = 3
                    else:
                        use_count -= 1
                    success, image = vidcap.read()
                
                tmp = []
                for frame in frames:
                    image = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = model.encode_image(image)

                    tmp.append(image_features.squeeze())
                features[video_id] = torch.stack(tmp, dim=0)
                    
        ext_len = len(output_ckpt.split('.')[-1])
        ext_len = -1 * (ext_len + 1)
        torch.save(features, '{}_part{}{}'.format(output_ckpt[:ext_len], k, output_ckpt[ext_len:]))

def get_fps():

    _, KeyframeSets = get_KeyframeDict()

    fps = {}

    for video_id in KeyframeSets.keys():

        filename = video_id + '.mp4'

        vidcap = cv2.VideoCapture(os.path.join('./data/star/Charades_v1_480', filename))
        fps[video_id] = vidcap.get(cv2.CAP_PROP_FPS)

    torch.save(fps, './data/star/fps.pth')

# extract_all_Keyframe()
# get_all_clip_feature(image_folder='./data/star/Frames', output_ckpt='./data/star/key_clipvitl14.pth')

# sample_frames()
# get_all_clip_feature(image_folder='./data/star/Residual_Frames', output_ckpt='./data/star/res_clipvitl14.pth')

# get_fps()

get_patch_clip_feature(image_folder='./data/star/Charades_v1_480', output_ckpt='./data/star/all_clipvitl14.pth')