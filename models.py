import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np

from get_loss import Loss_Fn, EntityTypes, euclidean_dist
import torch.nn.functional as F


# ============================================================================
# Multi-Granularity Pooling for LLM (handles 32+ layers)
# ============================================================================

class MultiGranularityPooling(nn.Module):
    """Extract and fuse representations at multiple granularities from deep LLMs."""
    def __init__(self, num_layers, hidden_size, num_groups=4):
        super().__init__()
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.layers_per_group = num_layers // num_groups

        self.group_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_groups)
        ])
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * num_groups, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_groups),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, all_hidden_states, attention_mask=None):
        group_cls = []
        for g in range(self.num_groups):
            start = 1 + g * self.layers_per_group
            end = start + self.layers_per_group
            group_layers = all_hidden_states[start:end]
            group_avg = torch.stack(group_layers, dim=0).mean(dim=0)
            group_avg = self.group_norms[g](group_avg)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (group_avg * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            else:
                pooled = group_avg[:, 0, :]
            group_cls.append(pooled)

        concat = torch.cat(group_cls, dim=-1)
        gate = F.softmax(self.gate_net(concat), dim=-1)
        stacked = torch.stack(group_cls, dim=1)
        fused = (stacked * gate.unsqueeze(-1)).sum(dim=1)
        fused = self.output_norm(fused)
        return fused


# ============================================================================
# Lightweight Layer Fusion (for shallow models like BERT/DeBERTa)
# ============================================================================

class LayerAttentionFusion(nn.Module):
    """Learnable weighted fusion of hidden layers (for encoder models)."""
    def __init__(self, num_layers=12, hidden_size=768):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, all_hidden_states):
        stacked = torch.stack(all_hidden_states[1:], dim=0)
        weights = F.softmax(self.layer_weights, dim=0)
        weights = weights.view(-1, 1, 1, 1)
        fused = (stacked * weights).sum(dim=0)
        return self.layer_norm(fused)


# ============================================================================
# Support-Query Cross Attention
# ============================================================================

class SupportQueryCrossAttention(nn.Module):
    """Query attends to support for task-adaptive representation."""
    def __init__(self, hidden_size=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, query_emb, support_emb):
        q = query_emb.unsqueeze(0)
        kv = support_emb.unsqueeze(0)
        attn_out, _ = self.cross_attn(q, kv, kv)
        query_emb = self.norm1(query_emb + attn_out.squeeze(0))
        query_emb = self.norm2(query_emb + self.ffn(query_emb))
        return query_emb


# ============================================================================
# Unified Text Encoder
# ============================================================================

class TextEncoder(nn.Module):
    """Unified text encoder supporting BERT, DeBERTa, LLM + LoRA."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(args.filevocab)
        config = AutoConfig.from_pretrained(args.fileModel)
        self.num_layers = config.num_hidden_layers
        self.is_decoder = getattr(config, 'is_decoder', False)

        model_type = getattr(config, 'model_type', '').lower()
        self.is_decoder = self.is_decoder or model_type in [
            'llama', 'qwen2', 'qwen', 'mistral', 'phi', 'gpt2', 'gpt_neo',
            'opt', 'falcon', 'bloom', 'gemma', 'internlm2'
        ]

        if self.is_decoder:
            self.model = AutoModel.from_pretrained(
                args.fileModel,
                torch_dtype=torch.float16 if getattr(args, 'fp16', False) else torch.float32
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.model = AutoModel.from_pretrained(args.fileModel)

        if getattr(args, 'use_lora', False):
            for param in self.model.parameters():
                param.requires_grad = False
            self._apply_lora(args)
        elif getattr(args, 'numFreeze', 0) > 0:
            self._freeze_layers(args.numFreeze)

        # Pooling
        if self.is_decoder:
            num_groups = max(min(4, self.num_layers // 4), 2)
            self.pooling = MultiGranularityPooling(
                num_layers=self.num_layers, hidden_size=config.hidden_size, num_groups=num_groups)
            self.pooling_type = 'multi_granularity'
        else:
            self.pooling = LayerAttentionFusion(
                num_layers=self.num_layers, hidden_size=config.hidden_size)
            self.pooling_type = 'layer_fusion'

        # Projection if model hidden size differs
        model_hidden_size = config.hidden_size
        if model_hidden_size != self.hidden_size:
            self.projection = nn.Linear(model_hidden_size, self.hidden_size)
        else:
            self.projection = None

    def _apply_lora(self, args):
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError("Please install peft: pip install peft")

        lora_r = getattr(args, 'lora_r', 16)
        lora_alpha = getattr(args, 'lora_alpha', 32)
        lora_dropout = getattr(args, 'lora_dropout', 0.1)

        model_type = getattr(self.model.config, 'model_type', '').lower()
        if model_type in ['llama', 'mistral', 'qwen2', 'gemma', 'internlm2']:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif model_type in ['qwen']:
            target_modules = ["c_attn", "c_proj"]
        else:
            target_modules = ["query", "value", "key"]

        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules, bias="none")
        self.model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"LoRA enabled: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    def _freeze_layers(self, numFreeze):
        unfreeze_layers = [f"layer.{i}" for i in range(numFreeze, self.num_layers)]
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

    def forward(self, text, modd):
        """Encode text.

        For encoder models (BERT/DeBERTa): returns (hidden [bs,seq,dim], attn_mask [bs,seq])
        For decoder models (LLM): returns pooled [bs, dim]
        """
        if modd == "text":
            max_len = self.args.text_max_len
        else:
            max_len = self.args.label_max_len

        encoded = self.tokenizer(
            text, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].cuda()
        attention_mask = encoded["attention_mask"].cuda()

        outputs = self.model(
            input_ids, attention_mask=attention_mask,
            output_hidden_states=True
        )
        all_hidden_states = outputs.hidden_states

        if self.pooling_type == 'multi_granularity':
            # LLM: return pooled vector directly
            pooled = self.pooling(all_hidden_states, attention_mask)
            if self.projection is not None:
                pooled = self.projection(pooled)
            return pooled
        else:
            # Encoder (BERT/DeBERTa): return (hidden, mask) for last_layer usage
            last_hidden = all_hidden_states[-1]  # [bs, seq, dim]
            if self.projection is not None:
                last_hidden = self.projection(last_hidden)
            return last_hidden, attention_mask

    def get_last_layer(self):
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            return self.model.encoder.layer[-1]
        elif hasattr(self.model, 'layers'):
            return self.model.layers[-1]
        elif hasattr(self.model, 'h'):
            return self.model.h[-1]
        else:
            return nn.Identity()


# ============================================================================
# Prompt Learner (restored to v1 behavior for encoder models)
# ============================================================================

class MultiPromptLearner(nn.Module):
    def __init__(self, args):
        super(MultiPromptLearner, self).__init__()
        self.args = args
        self.prompt_len = self.args.prompt_len
        self.hidden_size = self.args.hidden_size
        self.key_hidden_size = self.args.key_hidden_size
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
        """
        For sequence input (encoder models): name_embs is [N, seq, dim]
            Returns prompts as [N, new_seq, dim] (concatenated sequence)
        For pooled input (LLM): name_embs is [N, dim]
            Returns prompts as [N, dim] (fused vector)
        """
        if name_embs.dim() == 3:
            # Encoder mode: mem is [Nway, kshot, H], each class uses only its own memory
            name_embs_cls = name_embs[:, 0, :]  # [N, H], safe when N=1
            if self.training:
                noise = torch.randn_like(name_embs_cls) * self.noise_std
                name_embs_cls = name_embs_cls + noise
            prefix = name_embs[:, :1, :]
            suffix = name_embs[:, 1:, :]

            prompt_key = self.prompt_key
            name_embs_cls = name_embs_cls / (name_embs_cls.norm(dim=-1, keepdim=True) + 1e-8)
            prompt_key = prompt_key / (prompt_key.norm(dim=-1, keepdim=True) + 1e-8)

            weights = name_embs_cls @ prompt_key.T         # [N, pool_len]
            weights = F.softmax(weights, dim=1)
            weighted_prompts = torch.einsum('bp,phd->bhd', weights, self.prompt)  # [N, prompt_len, H]

            # mem is already [N, kshot, H] — each class's own memory → direct concat
            prompts = torch.cat([prefix, mem, weighted_prompts, suffix], dim=1)
            return prompts
        else:
            # LLM mode: pooled vector processing
            label_normed = name_embs / (name_embs.norm(dim=-1, keepdim=True) + 1e-8)
            noise = torch.randn_like(label_normed) * self.noise_std
            label_normed = label_normed + noise

            prompt_key = self.prompt_key
            prompt_key = prompt_key / (prompt_key.norm(dim=-1, keepdim=True) + 1e-8)

            weights = label_normed @ prompt_key.T
            weights = F.softmax(weights, dim=1)

            prompt_mean = self.prompt.mean(dim=1)
            weighted_prompt = weights @ prompt_mean

            nway = name_embs.size(0)
            kshot = mem.size(0) // nway
            mem_reshaped = mem.view(nway, kshot, -1).mean(dim=1)

            combined = name_embs + weighted_prompt + mem_reshaped
            return combined


# ============================================================================
# Main Model
# ============================================================================

class myModel(nn.Module):
    def __init__(self, args):
        super(myModel, self).__init__()
        self.args = args
        self.hdim = args.hidden_size

        self.encoder = TextEncoder(self.args)
        self.is_llm = self.encoder.is_decoder

        # Only use last_layer for encoder models (BERT/DeBERTa)
        if not self.is_llm:
            self.last_layer = self.encoder.get_last_layer()

        self.meta_prompt = MultiPromptLearner(self.args)
        self.loss_fn = Loss_Fn(args)
        self.eps = 1e-6
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(args.dropout)

        self.entity_types = EntityTypes(args.types_dict, args.nway,
                                        entity_embedding_size=self.hdim)
        self.ebd_dim = self.hdim
        self.temperature = 0.1
        self.mem_mix_logit = nn.Parameter(torch.zeros(1))
        self.calib_scale = nn.Parameter(torch.tensor(0.1))  # cross-modal blend ratio

        self.cross_attention = SupportQueryCrossAttention(
            hidden_size=self.hdim, num_heads=8, dropout=args.dropout)

    def cross_modal_calibrate(self, support_emb, label_emb):
        """Bidirectional cross-modal calibration for few-shot.
        - Support ← Label: enrich support with label semantics via soft attention.
          If a support sample aligns poorly with its own label (noisy 1-shot),
          cross-label attention provides corrective context.
        - Label ← Support: enrich labels with episode-specific support info.
        Returns (calibrated_support, calibrated_label)."""
        s_norm = F.normalize(support_emb, dim=-1)      # [S, H]
        l_norm = F.normalize(label_emb, dim=-1)         # [N, H]
        scale = torch.sigmoid(self.calib_scale) * 5.0   # adaptive temperature

        # Support ← Label: each support sample attends to label embeddings
        s2l_sim = s_norm @ l_norm.T
        s2l_attn = F.softmax(s2l_sim * scale, dim=1)
        label_context = s2l_attn @ label_emb
        calib_support = support_emb + 0.1 * label_context

        # Label ← Support: each label attends to support embeddings
        l2s_sim = l_norm @ s_norm.T
        l2s_attn = F.softmax(l2s_sim * scale, dim=1)
        support_context = l2s_attn @ support_emb
        calib_label = label_emb + 0.1 * support_context

        return calib_support, calib_label

    def build_extended_attention_mask(self, attention_mask):
        """Convert [B, L] attention mask to [B, 1, 1, L] extended format
           required by BERT self-attention layers."""
        return self.encoder.model.get_extended_attention_mask(
            attention_mask, attention_mask.shape, attention_mask.device)

    def build_diverse_memory_tokens(self, support_hiddens, class_name=None, top_k=3, modee="train"):
        """Retrieve the single best memory token per class via OT similarity."""
        device = support_hiddens.device
        dtype = support_hiddens.dtype

        if support_hiddens is None or support_hiddens.size(0) == 0:
            return torch.zeros(self.args.kshot, self.hdim, device=device, dtype=dtype)

        t, memory = self.entity_types.get_similar_memory(support_hiddens, class_name=class_name)

        # Find first valid memory bank
        best_token = None
        for idx in torch.argsort(t, descending=True).tolist():
            if memory[idx] is not None:
                best_token = memory[idx]
                break

        if best_token is None:
            return torch.zeros(self.args.kshot, self.hdim, device=device, dtype=dtype)

        # Select closest centroid to support center
        support_center = F.normalize(support_hiddens.mean(dim=0, keepdim=True), dim=-1)
        mem_norm = F.normalize(best_token, dim=-1)
        scores = support_center @ mem_norm.T  # [1, M]
        best_idx = scores.argmax(dim=-1)
        selected = best_token[best_idx]  # [1, H]

        # Pad to kshot copies (required by downstream prompt construction)
        if self.args.kshot > 1:
            selected = selected.repeat(self.args.kshot, 1)

        if modee == "train":
            selected = selected + 1e-4 * torch.randn_like(selected)

        return selected

    def compute_class_stats(self, support_embeds, support_labels):
        """Compute class means and a shared covariance with shrinkage."""
        unique_labels = torch.unique(support_labels)
        class_means = []
        for label in unique_labels:
            class_embed = support_embeds[support_labels == label]
            class_means.append(class_embed.mean(dim=0))
        class_means = torch.stack(class_means, dim=0)

        all_residuals = []
        for i, label in enumerate(unique_labels):
            class_embed = support_embeds[support_labels == label]
            all_residuals.append(class_embed - class_means[i])
        pooled_residuals = torch.cat(all_residuals, dim=0)
        shared_cov = (pooled_residuals.unsqueeze(2) @ pooled_residuals.unsqueeze(1)).mean(dim=0)
        total_k = pooled_residuals.size(0)
        shrink = total_k / (total_k + 3.0)
        shared_cov = shrink * shared_cov + (1 - shrink) * torch.eye(self.ebd_dim, device=shared_cov.device)
        shared_cov += self.eps * torch.eye(self.ebd_dim, device=shared_cov.device)
        class_covs = shared_cov.unsqueeze(0).expand(class_means.size(0), -1, -1)
        return class_means, class_covs

    def compute_logits(self, query_embeds, class_means, class_covs):
        """Compute Mahalanobis-distance logits (matches v1 behavior)."""
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

    def forward(self, text, labels, labels_ids, flag, unique_list, modee, top_k=3):
        support_size = self.args.nway * self.args.kshot

        # Encode text and labels
        enc_text = self.encoder(text, "text")
        enc_label = self.encoder(unique_list, "label")

        # Branch based on model type
        if not self.is_llm:
            # ====== ENCODER MODEL (BERT/DeBERTa) ======
            text_emb, text_mask = enc_text
            label_emb, label_mask = enc_label

            # Text: pass through last_layer with attention mask
            text_ext_mask = self.build_extended_attention_mask(text_mask)
            text_emb_new = self.last_layer(text_emb, attention_mask=text_ext_mask)
            text_emb_new_cls = text_emb_new[0][:, 0, :]  # [B, H]

            support_emb_cls = text_emb_new_cls[:support_size]
            query_emb_cls = text_emb_new_cls[support_size:]

            # Diverse memory tokens: [Nway, kshot, H] per class
            flag_tensor = torch.as_tensor(flag, device=support_emb_cls.device, dtype=torch.long)
            class_memory_tokens = []
            for typ in range(self.args.nway):
                sup_mask = flag_tensor[:support_size] == typ
                support_hiddens = support_emb_cls[sup_mask]
                class_name = unique_list[typ] if typ < len(unique_list) else None
                class_mem = self.build_diverse_memory_tokens(
                    support_hiddens=support_hiddens, class_name=class_name,
                    top_k=top_k, modee=modee)
                class_memory_tokens.append(class_mem)
            memory_pro = torch.stack(class_memory_tokens, dim=0)  # [Nway, kshot, H]

            # Prompt: label with memory → last_layer → CLS
            prompts = self.meta_prompt(label_emb, memory_pro)  # [Nway, new_seq, H]

            # Build prompt attention mask
            insert_len = memory_pro.size(1) + self.args.prompt_len
            insert_mask = torch.ones(label_mask.size(0), insert_len,
                                     device=label_mask.device, dtype=label_mask.dtype)
            prompt_mask = torch.cat([label_mask[:, :1], insert_mask, label_mask[:, 1:]], dim=1)
            prompt_ext_mask = self.build_extended_attention_mask(prompt_mask)

            label_emb_new = self.last_layer(prompts, attention_mask=prompt_ext_mask)
            label_emb_enhanced = label_emb_new[0][:, 0, :]  # [N, dim]

        else:
            # ====== LLM MODEL: Pooled vector pipeline ======
            text_emb = enc_text  # [bs, dim] already pooled
            label_emb = enc_label  # [N, dim]

            support_emb_cls = text_emb[:support_size]
            query_emb_cls = text_emb[support_size:]

            query_emb_cls = self.cross_attention(query_emb_cls, support_emb_cls)

            memory_list = []
            for typ in range(self.args.nway):
                class_name = unique_list[typ] if typ < len(unique_list) else None
                t, memory = self.entity_types.get_similar_memory(
                    support_emb_cls[np.array(flag)[:support_size] == typ],
                    class_name=class_name)
                top_k_indices = list(torch.argsort(t, descending=True).detach().cpu().numpy()[:top_k])
                top_k_memory = [memory[idx] for idx in top_k_indices if memory[idx] is not None]
                memory_concat = torch.cat(top_k_memory, dim=0) if len(top_k_memory) > 0 else None
                if memory_concat is not None:
                    memory_concat = (torch.mean(memory_concat, dim=0, keepdim=True) +
                                     torch.randn(self.args.kshot, self.hdim).to(text_emb.device) *
                                     torch.exp(-10 * torch.ones(self.args.kshot, self.hdim)).to(text_emb.device))
                else:
                    memory_concat = torch.zeros(self.args.kshot, self.hdim).to(text_emb.device)
                memory_list.append(memory_concat)

            memory_pro = torch.cat(memory_list, dim=0)
            label_emb_enhanced = self.meta_prompt(label_emb, memory_pro)  # [N, dim]

        # Cross-modal calibration: enrich support with label semantics,
        # and labels with episode-specific support context.
        calibrated_support, calibrated_label = self.cross_modal_calibrate(
            support_emb_cls, label_emb_enhanced)

        # Compute class stats and Mahalanobis logits
        support_labels_tensor = torch.tensor(flag[:support_size], device=support_emb_cls.device)
        class_means, class_covs = self.compute_class_stats(calibrated_support, support_labels_tensor)
        dot_sim = self.compute_logits(query_emb_cls, class_means, class_covs)

        dot_sim_mlp = self.softmax(dot_sim)

        if modee == "test":
            onehot_ = dot_sim_mlp
        else:
            query_onehot = labels_ids[support_size:]
            query_onehot_tensor = torch.tensor(query_onehot, dtype=float).cuda()
            query_onehot_ = self.args.beta * query_onehot_tensor + dot_sim_mlp
            onehot_ = self.softmax(query_onehot_)

        # Instance-level contrastive loss
        loss_simcse = 0
        if modee == "train" and getattr(self.args, 'use_contrastive', True):
            all_emb = torch.cat([calibrated_support, query_emb_cls], dim=0)
            all_emb_n = F.normalize(all_emb, dim=-1)
            sim = all_emb_n @ all_emb_n.T
            all_labels = torch.tensor(flag, device=all_emb.device)
            pos_mask = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)
            pos_mask.fill_diagonal_(0)
            if pos_mask.sum() > 0:
                tau = 0.15
                exp_sim = torch.exp(sim / tau)
                pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
                all_sum = exp_sim.sum(dim=1) - exp_sim.diag()
                loss_simcse = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8)).mean()

        loss, p, r, f, acc, auc = self.loss_fn(
            calibrated_support, query_emb_cls, calibrated_label, onehot_,
            loss_simcse, labels_ids, flag, self.entity_types, unique_list,
            modee, self.mem_mix_logit)

        return loss, p, r, f, acc, auc
