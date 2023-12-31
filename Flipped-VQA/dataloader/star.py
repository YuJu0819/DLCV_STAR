import torch
from .base_dataset import BaseDataset
import json
import csv

def get_KeyframeDict(csv_path='./Video_Keyframe_IDs.csv'):

    with open(csv_path) as f:
        csvreader = csv.reader(f)
        rows = []
        for row in csvreader:
            rows.append(row)
    
    KeyframeSets = {}
    KeyframeDict = {}
    for row in rows[1:]:

        list_id = row[2][1:-1].replace('\'', '').replace(' ', '').split(',')

        KeyframeDict[row[0]] = (row[1], list_id)

        if row[1] not in KeyframeSets.keys():
            KeyframeSets[row[1]] = set()
        for idx in list_id:
            KeyframeSets[row[1]].add(int(idx))

    return KeyframeDict, KeyframeSets

class STAR(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = json.load(open(f'./data/star/STAR_{split}.json', 'r'))
        self.features = torch.load(f'./data/star/clipvitl14.pth')
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
        self.qtype_mapping = {'Interaction': 1, 'Sequence': 2, 'Prediction': 3, 'Feasibility': 4}
        self.num_options = 4
        print(f"Num {split} data: {len(self.data)}") 

        KeyframeDict, _ = get_KeyframeDict('./data/star/Video_Keyframe_IDs.csv')
        self.KeyframeDict = KeyframeDict

        self.use_key_features = False
        if self.use_key_features:
            self.key_features = torch.load(f'./data/star/key_clipvitl14.pth', map_location='cpu')

        self.use_residual_frames = False
        if self.use_residual_frames:
            self.res_features = torch.load(f'./data/star/res_clipvitl14.pth', map_location='cpu')
            self.video_fps = torch.load(f'./data/star/fps.pth', map_location='cpu')

        if self.use_key_features and self.use_residual_frames:
            print('Warning: \'use_key_features\' and \'use_residual_frames\' both set to \'True\', cannot both be \'True\', \'use_key_features=True\' will be the priority')

    def _get_text(self, idx):
        question = self.data[idx]["question"].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
            
        options = {x['choice_id']: x['choice'] for x in self.data[idx]['choices']}
        options = [options[i] for i in range(self.num_options)]
        if self.split != 'test':
            answer = options.index(self.data[idx]['answer'])
        else:
            answer = 0
        
        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        a_text = "Answer: The answer is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text, answer

    def _get_video(self, video_id, start, end):
        if video_id not in self.features:
            print(video_id)
            video = torch.zeros(1, self.features_dim)
        else:
            video = self.features[video_id][start: end +1, :].float() # ts
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], 0)
        else:
            video_len = self.max_feats

        return video, video_len

    def _get_keyframe(self, question_id, video_id):
        assert self.KeyframeDict[question_id][0] == video_id
        if video_id not in self.key_features:
            print(question_id)
            video = torch.zeros(1, self.features_dim)
        else:
            key_ids = self.KeyframeDict[question_id][1]
            video = []
            for i, key_id in enumerate(key_ids):
                video.append(self.key_features[video_id][key_id])
            video = torch.cat(video, dim=0).float()
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], 0)
        else:
            video_len = self.max_feats

        return video, video_len

    def _get_resframe(self, question_id, video_id, start, end):
        assert self.KeyframeDict[question_id][0] == video_id
        if video_id not in self.res_features:
            print(question_id)
            video = torch.zeros(1, self.features_dim)
        else:
            fps = self.video_fps[video_id]
            start_frame = round(fps * start)
            end_frame = round(fps * end)
            video = []
            for res_id in sorted(self.res_features[video_id].keys()):
                frame_id = int(res_id.split('.')[0])
                if frame_id >= start_frame and frame_id <= end_frame:
                    video.append(self.res_features[video_id][res_id])
            if len(video) == 0:
                for res_id in sorted(self.res_features[video_id].keys()):
                    video.append(self.res_features[video_id][res_id])
            video = torch.cat(video, dim=0).float()
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], 0)
        else:
            video_len = self.max_feats

        return video, video_len


    def __getitem__(self, idx):
        vid = self.data[idx]['video_id']
        qtype = self.qtype_mapping[self.data[idx]['question_id'].split('_')[0]]
        question_id = self.data[idx]['question_id']
        text, answer = self._get_text(idx)
        text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
        start, end = round(self.data[idx]['start']), round(self.data[idx]['end'])
        if self.use_key_features:
            video, video_len = self._get_keyframe(question_id, f'{vid}')
        elif self.use_residual_frames:
            video, video_len = self._get_resframe(question_id, f'{vid}', start, end)
        else:
            video, video_len = self._get_video(f'{vid}', start, end)
        return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                "video_index": video_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype,  "question_id": question_id}


    def __len__(self):
        return len(self.data)