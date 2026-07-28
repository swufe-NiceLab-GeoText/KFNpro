"""Loss functions for Robust Few-Shot Text Classification.

Contains:
- EntityTypes: Global memory module with OT-based retrieval
- InferenceNet: Deep variational inference network  
- Loss_Fn: Main loss with three robustness contributions:
  (C1) Semantic Consistency Loss — clean/augmented representation alignment
  (C2) Prototype Dispersion Loss — maximize inter-class prototype distances
  (C3) Variational Robustness Prior — adaptive prior variance based on support dispersion
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, roc_auc_score
import torch.nn as nn
import logging
import ot

from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ============================================================================
# Memory Module
# ============================================================================

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

        self.memory = {}
        self.memory_size = {}
        self.num_centroids = num_centroids
        self.min_centroids = 5
        self.num_centroids = max(num_centroids, 25)  # larger bank for 5-shot diversity
        self.min_num_supports = min_num_supports
        self.support_sample_std = support_sample_std
        self.embedding_size = entity_embedding_size
        self.uniform = uniform
        self.init_memory()

    def init_memory(self):
        for class_name in self.types_dict:
            self.memory_size[class_name] = 0
            self.memory[class_name] = np.random.randn(self.num_centroids, self.embedding_size)
        logger.info(f"Init memory for {len(self.types_dict)} global classes")

    def get_similar_memory(self, hiddens, class_name=None):
        if hiddens is None or (hasattr(hiddens, 'shape') and hiddens.shape[0] == 0):
            default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            num_slots = len(self.memory)
            t = torch.ones(num_slots, device=default_device, dtype=torch.float32) * 1e-5
            memory = [None for _ in range(num_slots)]
            return t, memory

        device = hiddens.device
        hiddens = hiddens.detach().cpu().numpy()
        num_hiddens = hiddens.shape[0]

        if num_hiddens < self.min_num_supports:
            aug_hiddens_list = [hiddens]
            for _ in range((self.min_num_supports - num_hiddens) // num_hiddens + 1):
                aug_hiddens_list.append(
                    hiddens + np.random.randn(*hiddens.shape) * self.support_sample_std)
            aug_hiddens_embed = np.concatenate(aug_hiddens_list, axis=0)[:self.min_num_supports]
        else:
            aug_hiddens_embed = hiddens

        if class_name is not None and class_name in self.memory:
            target_names = [class_name]
        else:
            target_names = list(self.memory.keys())

        ot_list = []
        transp_hiddens_embed = []
        for cn in target_names:
            if self.memory_size[cn] >= self.min_centroids:
                ot_dist, coupling = self.calculate_ot_matrix(
                    self.memory[cn][:self.memory_size[cn]],
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
        memory = [torch.tensor(m).to(device).type(torch.float32) if m is not None else None
                  for m in transp_hiddens_embed]

        return t, memory

    def calculate_ot_matrix(self, X, Y, uniform=True):
        X_expand = np.expand_dims(X, axis=1)
        Y_expand = np.expand_dims(Y, axis=0)
        C = np.sum((X_expand - Y_expand) ** 2, axis=-1)
        C /= float(np.max(C) + 1e-10)

        if uniform:
            a = np.ones(X.shape[0]) / X.shape[0]
            b = np.ones(Y.shape[0]) / Y.shape[0]
        else:
            a_dist = np.sum((X - np.mean(X, axis=0, keepdims=True)) ** 2, axis=-1)
            b_dist = np.sum((Y - np.mean(Y, axis=0, keepdims=True)) ** 2, axis=-1)
            a = np.exp(a_dist / float(np.max(a_dist) + 1e-10))
            a = a / a.sum()
            b = np.exp(b_dist / float(np.max(b_dist) + 1e-10))
            b = b / b.sum()

        T_reg = ot.sinkhorn(a, b, C, 1e-1)
        ot_dist = np.sum(C * T_reg)
        return ot_dist, T_reg

    def update_memory(self, entity_embed, entity_labels, unique_list):
        for local_label in range(self.nway):
            if local_label >= len(unique_list):
                continue
            class_name = unique_list[local_label]
            if class_name not in self.memory:
                continue
            embed = entity_embed[entity_labels == local_label].detach().cpu()
            if embed.shape[0] == 0:
                continue
            ms = self.memory_size[class_name]
            if ms == 0:
                # Multi-scale seeding: create centroids at varying distances.
                # Close centroids (small noise) = high precision.
                # Far centroids (large noise) = high recall / coverage.
                proto = embed.mean(dim=0, keepdim=True).numpy()
                scales = [0.02, 0.05, 0.1, 0.2, 0.4]
                slots_per_scale = self.num_centroids // len(scales)
                idx = 0
                for scale in scales:
                    for _ in range(slots_per_scale):
                        noise = np.random.randn(self.embedding_size) * scale
                        self.memory[class_name][idx] = proto + noise
                        idx += 1
                # Fill remaining slots with mid-scale
                for i in range(idx, self.num_centroids):
                    noise = np.random.randn(self.embedding_size) * 0.1
                    self.memory[class_name][i] = proto + noise
                self.memory_size[class_name] = self.num_centroids
            elif ms + embed.shape[0] <= self.num_centroids:
                self.memory[class_name][ms: ms + embed.shape[0]] = embed
                self.memory_size[class_name] = ms + embed.shape[0]
            else:
                current_memory = self.memory[class_name]
                sims = cosine_similarity(embed, current_memory)
                sim_scores = np.mean(sims, axis=1)
                sim_median = np.median(sim_scores)
                low_sim_indices = np.where(sim_scores < sim_median)[0]
                if len(low_sim_indices) > 0:
                    replace_embeds = embed[low_sim_indices]
                    replace_num = min(len(low_sim_indices), self.num_centroids)
                    replace_indices = np.random.choice(self.num_centroids, replace_num, replace=False)
                    self.memory[class_name][replace_indices] = replace_embeds[:replace_num]
                self.memory_size[class_name] = self.num_centroids


# ============================================================================
# Variational Inference Network (Deep, 2-layer + residual)
# ============================================================================

class InferenceNet(nn.Module):
    """Single-layer inference network (matches v1 behavior for few-shot stability)."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(InferenceNet, self).__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Linear(input_dim * 2, hidden_dim)
        self.output_mu = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input1, input2, gamma, device):
        input1_mean = (torch.mean(input1, dim=0, keepdim=True)
                       if input1 is not None
                       else torch.zeros(1, self.input_dim).to(device))
        input2_mean = (torch.mean(input2, dim=0, keepdim=True)
                       if input2 is not None
                       else torch.zeros(1, self.input_dim).to(device))

        inputs = torch.cat([input1_mean, input2_mean], dim=-1)
        hiddens = self.dropout(F.relu(self.layer1(inputs)))
        mu = gamma * self.output_mu(hiddens) + (1 - gamma) * input1_mean
        logstd = -10 * torch.ones_like(mu)  # fixed variance (v1 behavior)
        return mu, logstd


# ============================================================================
# Utility Functions
# ============================================================================

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise ValueError(f"Dimension mismatch: x has {d}, y has {y.size(1)}")
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def center_loss(emb_cls, w_proto, qshot):
    n_class = w_proto.shape[0]
    if emb_cls.shape[0] == n_class:
        emb_cls = emb_cls.repeat_interleave(qshot, dim=0)
    repeated_proto = w_proto.repeat_interleave(qshot, dim=0)
    assert emb_cls.shape == repeated_proto.shape, \
        f"Shape mismatch: emb_cls {emb_cls.shape}, repeated_proto {repeated_proto.shape}"
    dist = torch.pow(emb_cls - repeated_proto, 2).sum()
    dist = torch.sqrt(dist)
    return dist / (2.0 * emb_cls.size(0))


# ============================================================================
# Main Loss Function
# ============================================================================

class Loss_Fn(torch.nn.Module):
    """Loss function for few-shot prototypical learning.

    Losses:
        - Center Loss: pull queries toward class prototypes
        - KL Divergence: align prediction with soft labels
        - Proto KL: variational prior-posterior alignment (train only)
    """
    def __init__(self, args):
        super(Loss_Fn, self).__init__()
        self.args = args
        self.hdim = args.hidden_size
        self.gamma = 0.4
        self.lamda = 0.1

        # Learnable loss weights (auto-balance via uncertainty weighting)
        self.loss_weight = nn.Parameter(torch.ones(3))

        # Set Transformer: learns to aggregate K support samples into a prototype.
        # For K=1 → equivalent to identity. For K=5 → extracts complementary info.
        self.set_proto_query = nn.Parameter(torch.randn(1, 1, self.hdim) * 0.02)
        self.set_proto_attn = nn.MultiheadAttention(
            self.hdim, num_heads=4, batch_first=True)
        self.set_proto_norm = nn.LayerNorm(self.hdim)

        self.inference_block = InferenceNet(
            input_dim=self.hdim, hidden_dim=self.hdim,
            output_dim=self.hdim, dropout_rate=args.dropout)

    def forward(self, support_emb_cls, query_emb_cls, prompts, query_lcm,
                loss_simcse, labels_ids, flag, entity_types, unique_list,
                modee="test", mem_mix_logit=None, top_k=5):
        """
        Args:
            support_emb_cls: [S, dim] support embeddings
            query_emb_cls: [Q, dim] query embeddings  
            prompts: [N, dim] prompt-enhanced label representations
            query_lcm: [Q, N] soft label targets
            loss_simcse: unused (kept for compat)
            labels_ids: list of one-hot label vectors
            flag: list of class indices for all samples
            entity_types: EntityTypes memory module
            unique_list: list of class names in episode
            modee: "train" | "valid" | "test"
            mem_mix_logit: learnable prototype mixing weight
            top_k: top-k memory retrieval
            augmented_emb: [bs, dim] augmented text embeddings (for C1, train only)
        """
        if modee == "test":
            query_size = self.args.nway * self.args.q_qshot
        else:
            query_size = self.args.nway * self.args.qshot
        support_size = self.args.nway * self.args.kshot

        mu_prior_list, logstd_prior_list = [], []
        mu_posterior_list, logstd_posterior_list = [], []

        proto_kl = None

        # ====== Memory retrieval + variational inference ======
        for typ in range(self.args.nway):
            support_hiddens = support_emb_cls[np.array(flag)[:support_size] == typ]
            support_hiddens = None if len(support_hiddens) == 0 else support_hiddens

            class_name = unique_list[typ] if typ < len(unique_list) else None
            t, memory = entity_types.get_similar_memory(support_hiddens, class_name=class_name)
            top_k_indices = list(torch.argsort(t, descending=True).detach().cpu().numpy()[:top_k])
            top_k_memory = [memory[idx] for idx in top_k_indices if memory[idx] is not None]
            memory_concat = torch.cat(top_k_memory, dim=0) if len(top_k_memory) > 0 else None

            mu_prior, logstd_prior = self.inference_block(
                support_hiddens, memory_concat, self.gamma, support_emb_cls.device)

            mu_prior_list.append(mu_prior)
            logstd_prior_list.append(logstd_prior)

            if modee == "train":
                query_hiddens = query_emb_cls[np.array(flag)[support_size:] == typ]
                query_hiddens = None if len(query_hiddens) == 0 else query_hiddens
                mu_posterior, logstd_posterior = self.inference_block(
                    support_hiddens, query_hiddens, self.gamma, support_emb_cls.device)
                mu_posterior_list.append(mu_posterior)
                logstd_posterior_list.append(logstd_posterior)

        prototype_mu_prior = torch.cat(mu_prior_list, dim=0)
        prototype_logstd_prior = torch.cat(logstd_prior_list, dim=0)

        if modee == "train":
            prototype_mu_posterior = torch.cat(mu_posterior_list, dim=0)
            prototype_logstd_posterior = torch.cat(logstd_posterior_list, dim=0)

            mem_prototypes = (prototype_mu_posterior +
                              torch.randn(self.args.nway, support_emb_cls.shape[-1]).to(
                                  prototype_mu_posterior.device) * torch.exp(prototype_logstd_posterior))

            kl_prototype = (prototype_logstd_prior - prototype_logstd_posterior + 0.5 * (
                -1 + (prototype_mu_posterior - prototype_mu_prior) ** 2 +
                torch.exp(2 * (prototype_logstd_posterior - prototype_logstd_prior))))
            proto_kl = torch.mean(torch.sum(kl_prototype, dim=-1))
        else:
            mem_prototypes = prototype_mu_prior

        # ====== Compute prototypes (simple mean — statistically optimal) ======
        # Set Transformer prototype: learns to aggregate K samples per class.
        # For K=1: attention over single token → identity-like (safe).
        # For K=5: cross-attention extracts complementary info from all samples.
        proto_3d = support_emb_cls.view(self.args.nway, -1, support_emb_cls.shape[1])  # [N, K, H]
        prototypes = []
        for c in range(self.args.nway):
            x = proto_3d[c:c+1]  # [1, K, H]
            q = self.set_proto_query  # [1, 1, H]
            out, _ = self.set_proto_attn(q, x, x)
            prototypes.append(self.set_proto_norm(out.squeeze()))
        prototypes = torch.stack(prototypes, dim=0)  # [N, H]

        if mem_mix_logit is not None:
            mem_weight = torch.sigmoid(mem_mix_logit)
        else:
            mem_weight = 0.5
        combined_prototypes = mem_weight * mem_prototypes + (1 - mem_weight) * prototypes
        w_proto = self.args.alpha * combined_prototypes + (1 - self.args.alpha) * prompts

        # ====== Bidirectional prototype refinement (test/eval only) ======
        # Step 1: Prototype→Query attention — each prototype attends to all
        #   queries to gather relevant query context (reverse of transductive).
        # Step 2: Query→Prototype transductive refinement — soft query
        #   assignments iteratively update prototypes.
        w_proto_eval = w_proto
        if modee in ("test", "valid", "val", "eval") and query_emb_cls.size(0) > 0:
            # Step 1: Prototype→Query attention refinement
            with torch.no_grad():
                p_n = F.normalize(w_proto_eval, dim=-1)    # [N, H]
                q_n = F.normalize(query_emb_cls, dim=-1)   # [Q, H]
                p2q_attn = F.softmax(p_n @ q_n.T * 3.0, dim=1)  # [N, Q]
                q_context = p2q_attn @ query_emb_cls       # [N, H]
                w_proto_eval = w_proto_eval + 0.05 * q_context  # mild refinement

            # Step 2: Query→Prototype transductive refinement
            w_proto_refined = w_proto.clone()
            for _ in range(3):
                # Compute class stats using refined prototypes for Mahalanobis
                refined_dists = euclidean_dist(query_emb_cls, w_proto_refined)
                assign = F.softmax(-refined_dists, dim=1)  # [Q, N]
                query_contrib = assign.T @ query_emb_cls  # [N, H]
                weights = assign.sum(dim=0).unsqueeze(1) + 1e-8  # [N, 1]
                w_proto_refined = (w_proto + query_contrib) / (1.0 + weights)
            w_proto_eval = w_proto_refined

        # ====== Base losses (always use original w_proto for training signal) ======
        if modee == "test":
            qshot = self.args.q_qshot
        else:
            qshot = self.args.qshot

        center_loss_num = center_loss(query_emb_cls, w_proto, qshot)
        dists = euclidean_dist(query_emb_cls, w_proto)
        log_p_y = F.log_softmax(-dists, dim=1).cuda()
        kl_loss = F.kl_div(log_p_y.float(), query_lcm.float(), reduction='batchmean').float()

        # ====== Combine losses ======
        loss_list = [center_loss_num, kl_loss]
        if proto_kl is not None:
            loss_list.append(proto_kl)
        if modee == "train" and isinstance(loss_simcse, torch.Tensor) and loss_simcse > 0:
            loss_list.append(loss_simcse)

        final_loss = []
        for i in range(len(loss_list)):
            w_idx = min(i, self.loss_weight.size(0) - 1)
            final_loss.append(
                loss_list[i] / (2 * self.loss_weight[w_idx].pow(2)) +
                torch.log(self.loss_weight[w_idx]))
        all_loss = torch.sum(torch.stack(final_loss))

        # ====== Compute metrics (prototype + min-distance ensemble) ======
        query_labels = labels_ids[support_size:]
        query_labels = torch.tensor(query_labels, dtype=float).cuda()
        # Prototype distance (with transductive refinement)
        euc_dists = euclidean_dist(query_emb_cls, w_proto_eval)
        # Min-distance to support samples (leverages multi-sample coverage)
        all_protos = support_emb_cls.view(self.args.nway, -1, support_emb_cls.shape[1])
        min_euc = euclidean_dist(query_emb_cls, all_protos.reshape(-1, all_protos.shape[-1]))
        min_dists = min_euc.view(query_emb_cls.size(0), self.args.nway, -1).min(dim=2)[0]
        # Blend: prototype distance + min-distance (weight 0.3 for min)
        dists = euc_dists + 0.5 * min_dists
        # Cosine distance
        q_n = F.normalize(query_emb_cls, dim=-1)
        p_n = F.normalize(w_proto_eval, dim=-1)
        cos_dists = 1.0 - q_n @ p_n.T
        dists = (dists / dists.mean()) + (cos_dists / cos_dists.mean())
        log_p_y = F.log_softmax(-dists, dim=1).cuda()

        # Standard argmax-based one-hot prediction
        y_pred_idx = torch.argmax(log_p_y, dim=1)
        y_pred = F.one_hot(y_pred_idx, num_classes=log_p_y.size(1)).float()

        target_mode = 'macro'
        sq = query_labels.cpu().detach()
        yp = y_pred.cpu().detach()

        p = precision_score(sq, yp, average=target_mode)
        r = recall_score(sq, yp, average=target_mode)
        f = f1_score(sq, yp, average=target_mode)
        acc = accuracy_score(sq, yp)

        y_score = log_p_y.cpu().detach()
        auc = roc_auc_score(sq, y_score)

        if modee == "train":
            entity_types.update_memory(
                support_emb_cls,
                np.array(flag[:support_size]),
                unique_list
            )

        return all_loss, p, r, f, acc, auc
