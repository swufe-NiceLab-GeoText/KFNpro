import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score,f1_score,recall_score, accuracy_score, roc_auc_score
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import math
import logging
import ot
import json

from sklearn.metrics.pairwise import cosine_similarity
logger = logging.getLogger(__file__)

class EntityTypes(object):

    def __init__(self,
                 types_dict,
                 nway: int = 5,
                 num_centroids: int = 15,
                 entity_embedding_size: int = 768,
                 min_num_supports: int = 10,
                 support_sample_std: float = 0.05,
                 uniform=True
                 ):
        self.types_dict = types_dict
        self.nway = nway
        self.mmpoollen = nway

        self.memory = {}
        self.memory_size = {}
        self.num_centroids = num_centroids
        self.min_centroids = 5
        self.min_num_supports = min_num_supports
        self.support_sample_std = support_sample_std
        self.embedding_size = entity_embedding_size
        self.uniform = uniform
        self.init_memory()

    def init_memory(self):
        for idx in range(self.mmpoollen):
            self.memory_size[idx] = 0
            self.memory[idx] = np.random.randn(self.num_centroids, self.embedding_size)
        logger.info("Init memory ...")

    def get_similar_memory(self, hiddens):

        device = hiddens.device
        D = hiddens.shape[-1]
        hiddens = hiddens.detach().cpu().numpy()
        num_hiddens = hiddens.shape[0]

        if num_hiddens < self.min_num_supports:
            aug_hiddens_list = [hiddens]
            for _ in range((self.min_num_supports - num_hiddens) // num_hiddens + 1):
                aug_hiddens_list.append(hiddens + \
                                        np.random.randn(*hiddens.shape) * self.support_sample_std)
            aug_hiddens_embed = np.concatenate(aug_hiddens_list, axis=0)[:self.min_num_supports]
        else:
            aug_hiddens_embed = hiddens

        ot_list = []
        transp_hiddens_embed = []
        for typ in range(self.mmpoollen):
            if typ in self.memory and self.memory_size[typ] >= self.min_centroids:
                ot_dist, coupling = self.calculate_ot_matrix(self.memory[typ][:self.memory_size[typ]],
                                                             aug_hiddens_embed, uniform=self.uniform)

                transp = coupling / np.sum(coupling, axis=1)[:, None]
                transp[~np.isfinite(transp)] = 0
                transp_Xs = np.dot(transp, aug_hiddens_embed)

                ot_list.append(ot_dist)
                transp_hiddens_embed.append(transp_Xs)
            else:
                ot_list.append(1e5)
                transp_hiddens_embed.append(None)

        t = 1.0 / np.array(ot_list)
        t = torch.tensor(t).to(device).type(torch.float32)
        memory = [torch.tensor(m).to(device).type(torch.float32) if m is not None else None for m in
                  transp_hiddens_embed]

        return t, memory

    def calculate_ot_matrix(self, X, Y, uniform=True):

        X_expand = np.expand_dims(X, axis=1)
        Y_expand = np.expand_dims(Y, axis=0)

        C = np.sum((X_expand - Y_expand) ** 2, axis=-1)
        C /= float(np.max(C))

        if uniform:
            a = np.ones(X.shape[0]) / X.shape[0]
            b = np.ones(Y.shape[0]) / Y.shape[0]
        else:
            a_dist = np.sum((X - np.mean(X, axis=0, keepdims=True)) ** 2, axis=-1)
            b_dist = np.sum((Y - np.mean(Y, axis=0, keepdims=True)) ** 2, axis=-1)
            a = F.softmax(a_dist / float(np.max(a_dist)))
            b = F.softmax(b_dist / float(np.max(b_dist)))

        T_reg = ot.sinkhorn(a, b, C, 1e-1)
        ot_dist = np.sum(C * T_reg)

        return ot_dist, T_reg

    def update_memory(self, entity_embed, entity_labels, entity_labels_name):

        for label in range(self.nway):
            embed = entity_embed[entity_labels == label].detach().cpu()
            ms = self.memory_size[label]
            if ms == 0:
                proto = embed.mean(dim=0, keepdim=True)
                for i in range(self.num_centroids):
                    noise = np.random.randn(self.embedding_size) * 0.1
                    self.memory[label][i] = proto.numpy() + noise
                self.memory_size[label] = self.num_centroids
            elif ms + embed.shape[0] <= self.num_centroids:
                self.memory[label][ms: ms + embed.shape[0]] = embed
                self.memory_size[label] = ms + embed.shape[0]
            else:
                current_memory = self.memory[label]
                sims = cosine_similarity(embed, current_memory)
                sim_scores = np.mean(sims, axis=1)
                sim_median = np.median(sim_scores)

                low_sim_indices = np.where(sim_scores < sim_median)[0]

                if len(low_sim_indices) > 0:
                    replace_embeds = embed[low_sim_indices]
                    replace_num = min(len(low_sim_indices), self.num_centroids)

                    replace_indices = np.random.choice(self.num_centroids, replace_num, replace=False)
                    self.memory[label][replace_indices] = replace_embeds[:replace_num]
                self.memory_size[label] = self.num_centroids

class InferenceNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(InferenceNet, self).__init__()

        self.input_dim = input_dim
        self.layer1 = nn.Linear(input_dim * 2, hidden_dim)
        self.output_mu = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.logstd_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, input1, input2, gamma, device):
        input1_mean = torch.mean(input1, dim=0, keepdim=True) if input1 is not None else torch.zeros(1,
                                                                                                     self.input_dim).to(
            device)
        input2_mean = torch.mean(input2, dim=0, keepdim=True) if input2 is not None else torch.zeros(1,
                                                                                                     self.input_dim).to(
            device)
        inputs = torch.cat([input1_mean, input2_mean], dim=-1)
        hiddens = self.dropout(F.relu(self.layer1(inputs)))
        mu = gamma * self.output_mu(hiddens) + (1 - gamma) * input1_mean
        logstd = -10 * torch.ones_like(mu)
        return mu, logstd
def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def count_eu_dist(x,y):
    distance = torch.norm(x,y,dim=1,p=2)
    return distance

def euclidean_distance(x1, x2):
    return torch.norm(x1 - x2, p=2, dim=-1)

def center_loss(emb_cls, w_proto, qshot):

    n_class = w_proto.shape[0]
    if emb_cls.shape[0] == n_class:
        emb_cls = emb_cls.repeat_interleave(qshot, dim=0)
    repeated_proto = w_proto.repeat_interleave(qshot, dim=0)
    assert emb_cls.shape == repeated_proto.shape, \
        f"Shape mismatch: support_emb_cls {emb_cls.shape}, repeated_proto {repeated_proto.shape}"

    dist = torch.pow(emb_cls - repeated_proto, 2).sum()
    dist = torch.sqrt(dist)
    return dist / (2.0 * emb_cls.size(0))

class Loss_Fn(torch.nn.Module):
    def __init__(self, args):
        super(Loss_Fn, self).__init__()
        self.args = args
        self.loss_ce = CrossEntropyLoss()
        self.hdim = 768
        self.gamma = 0.4
        self.lamda = 0.1
        self.loss_weight = nn.Parameter(torch.ones(5))
        self.w = nn.Parameter(torch.ones(2))
        self.inference_block = InferenceNet(input_dim=self.hdim, hidden_dim=self.hdim, output_dim=self.hdim, dropout_rate=0.1)

    def forward(self, support_emb_cls,query_emb_cls,prompts, query_lcm,loss_simcse,labels_ids, flag, entity_types, unique_list, modee="test", top_k=5):
        if modee=="test":
            query_size = self.args.nway * self.args.q_qshot
        else:
            query_size = self.args.nway * self.args.qshot
        support_size = self.args.nway * self.args.kshot

        mu_prior_list, logstd_prior_list = [], []
        mu_posterior_list, logstd_posterior_list = [], []
        memony_list = []

        proto_kl = None
        if modee == "train":

            for typ in range(self.args.nway):
                support_hiddens = support_emb_cls[np.array(flag)[:support_size] == typ]
                query_hiddens = query_emb_cls[np.array(flag)[support_size:] == typ]

                support_hiddens = None if len(support_hiddens) == 0 else support_hiddens
                query_hiddens = None if len(query_hiddens) == 0 else query_hiddens

                t, memory = entity_types.get_similar_memory(support_hiddens)
                top_k_indices = list(torch.argsort(t, descending=True).detach().cpu().numpy()[:top_k])
                top_k_memory = [memory[idx] for idx in top_k_indices if memory[idx] is not None]
                memory_concat = torch.cat(top_k_memory, dim=0) if len(top_k_memory) > 0 else None

                mu_prior, logstd_prior = self.inference_block(support_hiddens, memory_concat, self.gamma,
                                                              support_emb_cls.device)
                mu_posterior, logstd_posterior = self.inference_block(support_hiddens, query_hiddens, self.gamma,
                                                                      support_emb_cls.device)
                mu_prior_list.append(mu_prior)
                logstd_prior_list.append(logstd_prior)
                mu_posterior_list.append(mu_posterior)
                logstd_posterior_list.append(logstd_posterior)
            prototype_mu_prior = torch.cat(mu_prior_list, dim=0)
            prototype_logstd_prior = torch.cat(logstd_prior_list, dim=0)

            prototype_mu_posterior = torch.cat(mu_posterior_list, dim=0)
            prototype_logstd_posterior = torch.cat(logstd_posterior_list, dim=0)
            mem_prototypes = prototype_mu_posterior + torch.randn(self.args.nway, support_emb_cls.shape[-1]).to(
                prototype_mu_posterior.device) * torch.exp(prototype_logstd_posterior)

            kl_prototype = prototype_logstd_prior - prototype_logstd_posterior + 0.5 * (
                    -1 + (prototype_mu_posterior - prototype_mu_prior) ** 2 + torch.exp(
                2 * (prototype_logstd_posterior - prototype_logstd_prior)))  # P x D

            proto_kl = torch.mean(torch.sum(kl_prototype, dim=-1))
        else:
            for typ in range(self.args.nway):
                support_labels_tensor = torch.tensor(flag[:support_size]).to(support_emb_cls.device)

                support_hiddens = support_emb_cls[np.array(flag)[:support_size] == typ]

                support_hiddens = None if len(support_hiddens) == 0 else support_hiddens

                t, memory = entity_types.get_similar_memory(support_hiddens)
                top_k_indices = list(torch.argsort(t, descending=True).detach().cpu().numpy()[:top_k])
                top_k_memory = [memory[idx] for idx in top_k_indices if memory[idx] is not None]
                memory_concat = torch.cat(top_k_memory, dim=0) if len(top_k_memory) > 0 else None
                mu_prior, logstd_prior = self.inference_block(support_hiddens, memory_concat, self.gamma,
                                                              support_emb_cls.device)
                mu_prior_list.append(mu_prior)
                logstd_prior_list.append(logstd_prior)

            prototype_mu_prior = torch.cat(mu_prior_list, dim=0)
            mem_prototypes = prototype_mu_prior

        prototypes = support_emb_cls.view(self.args.nway, -1, support_emb_cls.shape[1])  # N X K X dim

        prototypes = torch.mean(prototypes, dim=1)

        combined_prototypes = 0.5 * mem_prototypes + 0.5 * prototypes
        w_proto = self.args.alpha * combined_prototypes + (1 - self.args.alpha) * prompts

        # Transductive refinement (test/eval only)
        w_proto_eval = w_proto
        if modee in ("test", "valid", "val", "eval") and query_emb_cls.size(0) > 0:
            w_proto_refined = w_proto.clone()
            for _ in range(3):
                refined_dists = euclidean_dist(query_emb_cls, w_proto_refined)
                assign = F.softmax(-refined_dists, dim=1)
                query_contrib = assign.T @ query_emb_cls
                weights = assign.sum(dim=0).unsqueeze(1) + 1e-8
                w_proto_refined = (w_proto + query_contrib) / (1.0 + weights)
            w_proto_eval = w_proto_refined

        qshot = self.args.q_qshot if modee == "test" else self.args.qshot
        center_loss_num = center_loss(query_emb_cls, w_proto, qshot)
        dists = euclidean_dist(query_emb_cls, w_proto)
        log_p_y = F.log_softmax(-dists, dim=1).cuda()
        kl_loss = F.kl_div(log_p_y.float(), query_lcm.float(), reduction='batchmean').float()

        loss_list = [center_loss_num, kl_loss]
        if proto_kl is not None:
            loss_list.append(proto_kl)
        final_loss = []
        for i in range(len(loss_list)):
            final_loss.append(loss_list[i] / (2 * self.loss_weight[i].pow(2)) + torch.log(self.loss_weight[i]))
        all_loss = torch.sum(torch.stack(final_loss))

        query_labels = labels_ids[support_size:]
        query_labels = torch.tensor(query_labels, dtype=float).cuda()
        euc_dists = euclidean_dist(query_emb_cls, w_proto_eval)
        # Min-distance to any support sample (helps 5-shot boundary)
        all_protos = support_emb_cls.view(self.args.nway, -1, support_emb_cls.shape[1])
        min_euc = euclidean_dist(query_emb_cls, all_protos.reshape(-1, all_protos.shape[-1]))
        min_dists = min_euc.view(query_emb_cls.size(0), self.args.nway, -1).min(dim=2)[0]
        # Cosine distance
        q_n = F.normalize(query_emb_cls, dim=-1)
        p_n = F.normalize(w_proto_eval, dim=-1)
        cos_dists = 1.0 - q_n @ p_n.T
        dists = (euc_dists + 0.5 * min_dists) / 1.5 + cos_dists
        log_p_y = F.log_softmax(-dists, dim=1).cuda()

        y_pred_idx = torch.argmax(log_p_y, dim=1)
        y_pred = F.one_hot(y_pred_idx, num_classes=log_p_y.size(1)).float()

        target_mode = 'macro'

        sq = query_labels.cpu().detach()
        yp = y_pred.cpu().detach()

        p = precision_score(sq, yp, average=target_mode)
        r = recall_score(sq, yp, average=target_mode)
        f = f1_score(sq, yp, average=target_mode)
        acc = accuracy_score(sq, yp)

        y_score = log_p_y
        y_score = y_score.cpu().detach()
        auc = roc_auc_score(sq, y_score)

        if modee == "train":
            entity_types.update_memory(
                # torch.cat([support_emb_cls, query_emb_cls], dim=0),
                # np.array(flag),
                # unique_list
                support_emb_cls,  # 只传 Support 特征
                np.array(flag[:support_size]),  # 只传 Support 标签
                unique_list
            )

        return all_loss, p,r,f,acc,auc

