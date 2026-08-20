import numpy
import torch
import torch.nn as nn
import random
from transformers import BertTokenizer, BertConfig, BertModel,AutoTokenizer,AutoModel
from transformers import BertLayer
import numpy as np

from get_lossnew import Loss_Fn, EntityTypes, euclidean_dist
import torch.nn.functional as F
from abc import ABC, abstractmethod
import copy
import json

class Bert_pure(nn.Module):

    def __init__(self, args):
        super(Bert_pure, self).__init__()
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.filevocab)
        self.bert = AutoModel.from_pretrained(self.args.fileModel)
        self.augment = True

        if self.args.numFreeze > 0:
            self.freeze_layers(self.args.numFreeze)

    def freeze_layers(self, numFreeze):
        unfreeze_layers = []
        for i in range(numFreeze, 12):
            unfreeze_layers.append("layer."+str(i))

        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

    def forward(self, text, modd):
        input_ids=[]
        max_len=0
        if modd == "text":
            max_len = self.args.text_max_len
        else:
            max_len = self.args.label_max_len

        text_ids = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_len
        )
        text_ids = text_ids["input_ids"]

        bs = len(text)
        text_ids = torch.tensor(text_ids)

        text_len = text_ids.size(1)

        atten_mask_text = torch.ones(bs, text_len)
        atten_mask_text[text_ids == 0] = 0

        text_ids = text_ids.cuda()
        atten_mask_text = atten_mask_text.cuda()

        output_text = self.bert(
            text_ids,
            attention_mask = atten_mask_text,
            output_hidden_states=True
        )
        every_layer = output_text[2]

        layer_output = every_layer[11]

        return layer_output
    def get_last_layer(self):

        return self.bert.encoder.layer[11]

class MultiPromptLearner(nn.Module):
    def __init__(self, args):
        super(MultiPromptLearner, self).__init__()
        self.args = args
        self.prompt_len = self.args.prompt_len
        self.hidden_size = self.args.hidden_size
        self.key_hidden_size = self.args.key_hidden_size  #768
        self.pool_len = self.args.pool_len
        self.noise_std = 0.05

        prompt_key_init = self.args.key_init_method

        key_shape = (self.pool_len, self.key_hidden_size)
        if prompt_key_init == 'zero':
            self.prompt_key = nn.Parameter(torch.zeros(key_shape))
        elif prompt_key_init == 'uniform':
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            nn.init.uniform_(self.prompt_key, -1, 1)
        elif prompt_key_init == 'normal':
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            nn.init.normal_(self.prompt_key, std=0.05)
        ctx_vectors = torch.empty(self.pool_len, self.prompt_len, self.hidden_size, dtype=torch.float)
        nn.init.normal_(ctx_vectors, std=0.02)

        self.prompt = nn.Parameter(ctx_vectors)

    def forward(self, name_embs, mem):

        name_embs_cls = name_embs[:, :1, :].squeeze()
        noise = torch.randn_like(name_embs_cls) * self.noise_std
        name_embs_cls = name_embs_cls + noise
        prefix = name_embs[:, :1, :]
        suffix = name_embs[:, 1:, :]

        prompt_key = self.prompt_key
        name_embs_cls = name_embs_cls / name_embs_cls.norm(dim=-1, keepdim=True)
        prompt_key = prompt_key / prompt_key.norm(dim=-1, keepdim=True)

        weights = name_embs_cls @ prompt_key.T
        weights = F.softmax(weights, dim=1)

        prompt = self.prompt.permute(2, 0, 1)
        weights = weights.unsqueeze(dim=0)
        weighted_prompts = torch.matmul(weights, prompt)
        weighted_prompts = weighted_prompts.permute(1, 2, 0)
        mem_expand = mem.unsqueeze(0).expand(weighted_prompts.size(0), -1, -1)

        prompts = torch.cat(
                [
                    prefix,
                    mem_expand,
                    weighted_prompts,
                    suffix
                ],
                dim=1,
        )

        return prompts


class myModel(nn.Module):
    def __init__(self, args):
        super(myModel, self).__init__()
        self.args = args
        self.bert = Bert_pure(self.args)
        self.last_layer = self.bert.get_last_layer()
        self.meta_prompt = MultiPromptLearner(self.args)
        self.loss_fn = Loss_Fn(args)
        self.eps = 1e-6
        self.hdim = 768
        self.type_project = nn.Linear(768, 768)
        self.linear = nn.Linear(self.args.nway, self.args.nway)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
        self.input_size = 768
        self.hidden_size = self.input_size

        self.entity_types = EntityTypes(args.types_dict, args.nway)
        self.ebd_dim = 768
        self.temperature = 0.1

    def compute_class_stats(self, support_embeds, support_labels):
        class_means = []
        class_covs = []
        unique_labels = torch.unique(support_labels)

        for label in unique_labels:
            class_embed = support_embeds[support_labels == label]
            mean = class_embed.mean(dim=0)
            diff = class_embed - mean
            cov = (diff.unsqueeze(2) @ diff.unsqueeze(1)).mean(dim=0)
            cov += self.eps * torch.eye(self.ebd_dim,
                                        device=cov.device)
            class_means.append(mean)
            class_covs.append(cov)

        class_means = torch.stack(class_means, dim=0)
        class_covs = torch.stack(class_covs, dim=0)
        return class_means, class_covs

    def compute_logits(self, query_embeds, class_means, class_covs):
        num_queries = query_embeds.size(0)
        n_classes = class_means.size(0)
        logits = []
        for i in range(n_classes):
            mean = class_means[i]
            cov = class_covs[i]
            cov_inv = torch.inverse(cov)
            diff = query_embeds - mean.unsqueeze(0)
            left = torch.matmul(diff, cov_inv)
            mahalanobis_distance = (left * diff).sum(dim=1)
            logits.append(-mahalanobis_distance)
        logits = torch.stack(logits, dim=1)
        logits = logits / self.temperature
        return logits

    def forward(self, text, labels, labels_ids, flag, unique_list, modee, top_k=5):
        support_size = self.args.nway * self.args.kshot
        memony_list = []

        text_emb = self.bert(text,"text")
        label_emb = self.bert(unique_list,"label")
        text_emb_new = self.last_layer(text_emb)
        text_emb_new_cls = text_emb_new[0][:, :1, :].squeeze()

        support_emb_cls = text_emb_new_cls[:support_size]
        query_emb_cls = text_emb_new_cls[support_size:]

        for typ in range(self.args.nway):
            t, memory = self.entity_types.get_similar_memory(support_emb_cls[np.array(flag)[:support_size] == typ])
            top_k_indices = list(torch.argsort(t, descending=True).detach().cpu().numpy()[:top_k])
            top_k_memory = [memory[idx] for idx in top_k_indices if memory[idx] is not None]
            memory_concat = torch.cat(top_k_memory, dim=0) if len(top_k_memory) > 0 else None
            if memory_concat is not None:
                memory_concat = (torch.mean(memory_concat, dim=0, keepdim=True) +
                                 0.1 * torch.randn(self.args.kshot, self.hdim).to(text_emb.device))
            else:
                memory_concat = torch.zeros(self.args.kshot, self.hdim).to(text_emb.device)
            memony_list.append(memory_concat)

        memory_pro = torch.cat(memony_list, dim=0)
        prompts = self.meta_prompt(label_emb, memory_pro)

        label_emb_new = self.last_layer(prompts)
        label_emb_new_cls = label_emb_new[0][:, :1, :].squeeze()

        if modee == "test":
            support_labels_tensor = torch.tensor(flag[:support_size], device=text_emb.device)
            class_means, _ = self.compute_class_stats(support_emb_cls, support_labels_tensor)
            dot_sim = self.compute_logits(query_emb_cls, class_means, _)
            dot_sim_mlp = self.softmax(dot_sim)
            onehot_ = dot_sim_mlp
        else:
            support_labels_tensor = torch.tensor(flag[:support_size], device=text_emb.device)
            class_means, _ = self.compute_class_stats(support_emb_cls, support_labels_tensor)
            dot_sim = self.compute_logits(query_emb_cls, class_means, _)
            dot_sim_mlp = self.softmax(dot_sim)
            query_onehot = labels_ids[support_size:]
            query_onehot_tensor = torch.tensor(query_onehot, dtype=float).cuda()
            query_onehot_ = self.args.beta * query_onehot_tensor + dot_sim_mlp
            onehot_ = self.softmax(query_onehot_)

        loss_simcse = 0

        loss,p,r,f,acc,auc= self.loss_fn(support_emb_cls,query_emb_cls,label_emb_new_cls, onehot_,loss_simcse,labels_ids,flag,self.entity_types, unique_list, modee)

        return loss,p,r,f,acc,auc

