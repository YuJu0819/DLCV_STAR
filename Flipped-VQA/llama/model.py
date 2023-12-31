# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import Embedding, Linear
import torch

@dataclass
class ModelArgs:
    dim: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    vocab_size: int = 32000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    adapter_len: int=10
    adapter_layer: int=24


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.max_feats = args.max_feats

        self.q_proj = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.k_proj = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.v_proj = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.o_proj = Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.gate1 = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))
        self.gate2 = torch.nn.Parameter(torch.ones(1, self.n_local_heads, 1, 1) * -args.bias)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_k = self.k_proj(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.v_proj(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            xk = torch.cat([adapter_k, xk], dim=1)
            xv = torch.cat([adapter_v, xv], dim=1)
            extra_mask = torch.zeros(1, 1, seqlen, adapter_len).to(mask)
            mask = torch.cat([extra_mask, mask], dim=-1)
        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        if adapter is not None:            
            adapter_scores = F.softmax(scores[..., :adapter_len].float(), dim=-1).type_as(xq) * self.gate1.tanh().half()
            if video_start is not None:
                vt_scores = scores[..., adapter_len:].clone()
                vt_scores[:, :, video_start + self.max_feats:, video_start:video_start + self.max_feats] = \
                    vt_scores[:, :, video_start + self.max_feats:, video_start:video_start + self.max_feats] + self.gate2.half()
                vt_scores = F.softmax(vt_scores.float(), dim=-1).type_as(xq)
            else:
                vt_scores = F.softmax(scores[..., adapter_len:], dim=-1)
            scores = torch.cat([adapter_scores, vt_scores], dim=-1)
        else:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = 5504

        self.gate_proj = Linear(dim, hidden_dim, bias=False)#gate
        self.down_proj = Linear(hidden_dim, dim, bias=False)#down
        self.up_proj = Linear(dim, hidden_dim, bias=False)#up

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.self_attn = Attention(args)
        self.mlp = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of)
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        h = x + self.self_attn.forward(self.input_layernorm(x), start_pos, freqs_cis, mask, adapter, video_start)
        out = h + self.mlp.forward(self.post_attention_layernorm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, args):
        super().__init__()
        params.max_feats = args.max_feats
        params.bias = args.bias
        self.args = args
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.max_feats = args.max_feats

        self.model = torch.nn.Module()
        self.model.embed_tokens = Embedding(params.vocab_size, params.dim)

        self.adapter_query = Embedding(params.adapter_len * params.adapter_layer, params.dim)
        self.visual_proj = Linear(768, params.dim, bias=False)
        self.temporal_emb = Embedding(self.max_feats, params.dim)
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        self.vqa_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.vaq_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.qav_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.inference_criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')

        self.model.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.model.layers.append(TransformerBlock(layer_id, params))

        self.model.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.lm_head = Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

        self.video_label = torch.arange(1, self.max_feats)
        self.tau = args.tau

    def forward(self, data, inference=False):
        video = data['video'].cuda()
        vqa_id, vaq_id, qav_id = data['text_id']['vqa'].cuda(), data['text_id']['vaq'].cuda(), data['text_id']['qav'].cuda()
        vqa_label, vaq_label, qav_label = data['label']['vqa'].cuda(), data['label']['vaq'].cuda(), data['label']['qav'].cuda()
        vqa_video_start, vaq_video_start, qav_video_index = data['video_start']['vqa'][0], data['video_start']['vaq'][0], data['video_index']['qav'].cuda()
        
        bsz, n_options, seqlen = vqa_id.shape
        vqa_id, vaq_id = vqa_id.reshape(-1, seqlen), vaq_id.reshape(-1, seqlen)
        vqa_label, vaq_label = vqa_label.reshape(-1, seqlen), vaq_label.reshape(-1, seqlen)
        vqa_label, vaq_label = vqa_label[:, 1:].flatten(), vaq_label[:, 1:].flatten()
        
        qav_id = qav_id.reshape(-1, seqlen)
        qav_label = qav_label.reshape(-1, seqlen)
        qav_video_mask = qav_label.ge(0)
        qav_label = qav_label[:, 1:].flatten()
        
        
        with torch.no_grad():
            vqa_h = self.model.embed_tokens(vqa_id)
            
            if self.args.vaq and not inference:
                vaq_h = self.model.embed_tokens(vaq_id)
            
            if self.args.qav and not inference:
                qav_h = self.model.embed_tokens(qav_id)
            
        freqs_cis = self.freqs_cis.to(vqa_h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=vqa_h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(vqa_h)
        start_pos = 0
        vaq_loss, qav_loss = torch.tensor([0]).cuda(), torch.tensor([0]).cuda()
        
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)
        _video_feature = self.visual_proj(video)
        if inference:
            _video_feature = _video_feature.unsqueeze(1).repeat(1, n_options, 1, 1).view(-1, _video_feature.shape[-2], _video_feature.shape[-1])
        video_feature = (_video_feature + self.temporal_emb.weight[None, :, :]).half()
        
        vqa_h = vqa_h.clone()
        vqa_h[:, vqa_video_start:vqa_video_start+self.max_feats] = video_feature

        
        if self.args.vaq and not inference:
            vaq_h = vaq_h.clone()
            vaq_h[:, vaq_video_start:vaq_video_start+self.max_feats] = video_feature
            
        if self.args.qav and not inference:
            qav_h = qav_h * ~qav_video_mask[..., None]
            qav_h.scatter_add_(1, qav_video_index[..., None].repeat(1, 1, self.params.dim), video_feature)
        
        for i, layer in enumerate(self.model.layers):
            vqa_h = layer(vqa_h, start_pos, freqs_cis, mask, adapter[i].half(), vqa_video_start)
            
            if self.args.vaq and not inference:
                vaq_h = layer(vaq_h, start_pos, freqs_cis, mask, adapter[i].half(), vaq_video_start)
            
            if self.args.qav and not inference:
                qav_h = layer(qav_h, start_pos, freqs_cis, mask, adapter[i].half(), None)
        
        
        vqa_h = self.model.norm(vqa_h)
        vqa_output = self.lm_head(vqa_h)
        #print(vqa_output.size())
        vqa_output = vqa_output[:, :-1, :].reshape(-1, self.vocab_size)
        #print(vqa_output.size())
        vqa_loss = self.vqa_criterion(vqa_output, vqa_label)
        #print(vqa_output.size(), vqa_label)
        
        if self.args.vaq and not inference:
            vaq_h = self.model.norm(vaq_h)
            vaq_output = self.lm_head(vaq_h)
            vaq_output = vaq_output[:, :-1, :].reshape(-1, self.vocab_size)
            vaq_loss = self.vaq_criterion(vaq_output, vaq_label)
            
        if self.args.qav and not inference:
            qav_h = self.model.norm(qav_h)
            qav_output = torch.bmm(qav_h[:, :-1].float(), _video_feature.transpose(1, 2).float()).reshape(-1, self.max_feats)
            qav_loss = self.qav_criterion(qav_output / self.tau, qav_label)

        if inference:
            logits = self.inference_criterion(vqa_output, vqa_label)
            logits = logits.reshape(bsz, n_options, -1)
            return logits
        else:
            return vqa_loss, vaq_loss, qav_loss