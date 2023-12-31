import argparse
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from llama import Tokenizer
from llama_vqa import LLaMA_VQA
from dataloader import load_data
import csv
import random
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--llama_model_path', default='./pretrained/llama/7B/', type=str, help='path of llama model')
    parser.add_argument('--model', default='llama7B_adapter', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--adapter_layer', type=int, default=32, metavar='LENGTH', help='the number of adapter layer')
    parser.add_argument('--adapter_len', type=int, default=10, metavar='LENGTH', help='the adapter length')
    parser.add_argument('--max_seq_len', type=int, default=128, metavar='LENGTH', help='the maximum sequence length')
    parser.add_argument('--max_feats', type=int, default=10, metavar='LENGTH', help='the maximum feature length')
    # Dataset parameters
    parser.add_argument('--dataset', default='star', type=str, help='dataset')
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
    
    parser.add_argument('--vaq', action='store_true', help='vaq loss')
    parser.add_argument('--qav', action='store_true', help='qav loss')
    parser.add_argument('--bias', type=float, default=3., help='attention bias')
    parser.add_argument('--tau', type=float, default=100., help='tau')

    return parser

def main(args):

    device = torch.device(args.device)#cuda
    print(device)

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

    tokenizer = Tokenizer(model_path=f'{args.llama_model_path}/tokenizer.model')

    data_loader_val = load_data(args, tokenizer, split='test')

    model = LLaMA_VQA(args)
    model.to(device)
    model_without_ddp = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    model.eval()
    num = 0
    cnt = 0
    with open('result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['question_id', 'answer'])
        for data in tqdm(data_loader_val):

            # answer = data['answer'].cuda()
            with torch.no_grad():
                logits = model(data, inference=True)
            
            count = (logits != 0).sum(-1)
            prediction = (logits.sum(-1) / count).argmin(-1)
            writer.writerow([data['questionid'][0], prediction[0].detach().cpu().numpy()])
            # if answer == prediction:
            #     cnt += 1
            # num += 1

        # print("Acc:",cnt/num)
        

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
