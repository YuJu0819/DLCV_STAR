import os
import argparse
import datetime
import json
import time
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
# from engine import train_one_epoch, val_one_epoch
from llama import Tokenizer
from llama_vqa import LLaMA_VQA
from dataloader import load_data

import random


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_model_path', default='/content/sample_data/llama.pth', type=str, help='path of llama model')
    parser.add_argument('--model', default='llama7B_adapter', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--adapter_layer', type=int, default=32, metavar='LENGTH', help='the number of adapter layer')
    parser.add_argument('--adapter_len', type=int, default=10, metavar='LENGTH', help='the adapter length')
    parser.add_argument('--max_seq_len', type=int, default=512, metavar='LENGTH', help='the maximum sequence length')
    parser.add_argument('--max_feats', type=int, default=10, metavar='LENGTH', help='the maximum feature length')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', default='nextqa', type=str, help='dataset')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--vaq', action='store_true', help='vaq loss')
    parser.add_argument('--qav', action='store_true', help='qav loss')
    parser.add_argument('--bias', type=float, default=3., help='attention bias')
    parser.add_argument('--tau', type=float, default=100., help='tau')
    parser.add_argument('--sub', action='store_true', help='subtitles for VLEP and TVQA')

    return parser

def main(args):
    # no need to use distributed training
    # misc.init_distributed_mode(args)

    args.batch_size = 1

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)#cuda

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

    tokenizer = Tokenizer(model_path=f'./pretrained/tokenizer.model')

    data_loader_val = load_data(args, tokenizer, split='test')

    model = LLaMA_VQA(args)
    model.to(device)
    model_without_ddp = model

    # 加载预训练模型
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    model.eval()
    cnt = 0
    num = 0

    output_json = {
        "Interaction" : [],
        "Sequence" : [],
        "Prediction" : [],
        "Feasibility" : [],
    }

    for i, data in enumerate(data_loader_val):
        if i % 1000 == 0:
            print(i)
        # answer = data['answer'].cuda()
        # bsz = answer.shape[0]

        with torch.no_grad():
            logits = model(data, inference=True)
        
        count = (logits != 0).sum(-1)
        prediction = (logits.sum(-1) / count).argmin(-1)
        # print("prediction", prediction)
        # print("answer", answer)
        # print("====================================")
        # if answer == prediction:
        #     cnt += 1
        # num += 1
        # eval = (answer == prediction)
        # acc = eval.sum().item() / bsz

        qid = data['question_id'][0]
        qtype = data['qtype'][0]-1
        qtype_mapping = ['Interaction', 'Sequence', 'Prediction', 'Feasibility']

        p = {"question_id": qid, "answer": prediction.cpu().item()}
        output_json[qtype_mapping[qtype]].append(p)
        
    output_json["Interaction"] = sorted(output_json["Interaction"], key=lambda obj: int(obj["question_id"].split('_')[-1]))
    output_json["Sequence"] = sorted(output_json["Sequence"], key=lambda obj: int(obj["question_id"].split('_')[-1]))
    output_json["Prediction"] = sorted(output_json["Prediction"], key=lambda obj: int(obj["question_id"].split('_')[-1]))
    output_json["Feasibility"] = sorted(output_json["Feasibility"], key=lambda obj: int(obj["question_id"].split('_')[-1]))

    with open(f'{args.output_dir}/pred.json', 'w') as f:
        json.dump(output_json, f)
        
    print(cnt / (i+1))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
