
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.nn import functional as F

# -------------------------------------------------------
# 1. Dataset
# -------------------------------------------------------
class DHDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts  = df["Comment"].astype(str).tolist()
        self.satire = df["Satire Level New"].astype(int).tolist()
        self.humor  = df["Humor Attribute New"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["satire"] = torch.tensor(self.satire[idx], dtype=torch.long)
        item["humor"]  = torch.tensor(self.humor[idx], dtype=torch.long)
        return item

# -------------------------------------------------------
# 2. Model with dropout + mean+CLS pooling
# -------------------------------------------------------
class MultiTaskModel(nn.Module):
    def __init__(self, base_model="roberta-base", satire_classes=3, humor_classes=5, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        pooled_dim = hidden * 2  # CLS + mean
        self.dropout = nn.Dropout(dropout)

        # Heads
        self.satire_head = nn.Linear(pooled_dim, satire_classes - 1)  # CORAL
        self.humor_head  = nn.Linear(pooled_dim, humor_classes)

        # Learnable scalars
        self.synth_weight_satire = nn.Parameter(torch.tensor(0.2))
        self.synth_weight_humor  = nn.Parameter(torch.tensor(0.2))
        self.alpha_sat = nn.Parameter(torch.tensor(0.2))
        self.alpha_hum = nn.Parameter(torch.tensor(0.2))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0]
        mean_pool = out.last_hidden_state.mean(dim=1)
        pooled = torch.cat([cls_token, mean_pool], dim=1)
        pooled = self.dropout(pooled)
        sat_logits = self.satire_head(pooled)
        hum_logits = self.humor_head(pooled)
        return sat_logits, hum_logits, pooled


# -------------------------------------------------------
# 3. Loss functions
# -------------------------------------------------------
def coral_loss(logits, targets, num_classes=3):
    device = logits.device
    k_minus_1 = logits.shape[1]
    targets = targets.unsqueeze(1).expand(-1, k_minus_1)
    t = (targets > torch.arange(k_minus_1, device=device).unsqueeze(0)).float()
    return -(t * F.logsigmoid(logits) + (1 - t) * F.logsigmoid(-logits)).sum(1).mean()

def label_smooth_ce(logits, targets, num_classes=5, eps=0.1):
    one_hot = torch.zeros_like(logits).scatter(1, targets.unsqueeze(1), 1)
    one_hot = one_hot * (1 - eps) + eps / num_classes
    return -(one_hot * F.log_softmax(logits, dim=1)).sum(1).mean()

def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """
    Supervised Contrastive Loss with variable positives.
    """
    device = embeddings.device
    batch_size = embeddings.size(0)
    embeddings = F.normalize(embeddings, dim=1)

    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    mask.fill_diagonal_(0)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()

    exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size, device=device))
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    loss = -mean_log_prob_pos.mean()
    return loss

# -------------------------------------------------------
# 4. Evaluation
# -------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    preds_sat, preds_hum, gts_sat, gts_hum = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            sat_logits, hum_logits, _ = model(ids, mask)
            sat_pred = torch.sum(torch.sigmoid(sat_logits) > 0.5, dim=1)
            hum_pred = hum_logits.argmax(1)
            preds_sat.extend(sat_pred.cpu().numpy())
            preds_hum.extend(hum_pred.cpu().numpy())
            gts_sat.extend(batch["satire"].numpy())
            gts_hum.extend(batch["humor"].numpy())
    r_sat = classification_report(gts_sat, preds_sat, digits=4, zero_division=0)
    r_hum = classification_report(gts_hum, preds_hum, digits=4, zero_division=0)
    return preds_sat, preds_hum, r_sat, r_hum

# -------------------------------------------------------
# 5. Main training loop
# -------------------------------------------------------
def main():
    train_df = pd.read_excel("DHD-Train.xlsx")
    val_df   = pd.read_excel("DHD-Val.xlsx")
    test_df  = pd.read_excel("DHD-Test.xlsx")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_ds  = DHDataset(train_df, tokenizer)
    val_ds    = DHDataset(val_df, tokenizer)
    test_ds   = DHDataset(test_df, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)  # increased batch size
    val_loader   = DataLoader(val_ds, batch_size=32)
    test_loader  = DataLoader(test_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MultiTaskModel().to(device)

    # Optimizer groups
    main_params = [p for n, p in model.named_parameters() if "synth_weight" not in n and "alpha" not in n]
    synth_params = [model.synth_weight_satire, model.synth_weight_humor]
    alpha_params = [model.alpha_sat, model.alpha_hum]

    optimizer = torch.optim.AdamW([
        {"params": main_params,  "lr": 2e-5, "weight_decay": 0.01},
        {"params": synth_params, "lr": 1e-3, "weight_decay": 0.0},
        {"params": alpha_params, "lr": 1e-3, "weight_decay": 0.0},
    ])

    total_steps = len(train_loader) * 5
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Dynamic weighting
    lambda_satire = nn.Parameter(torch.tensor(1.0)).to(device)
    lambda_humor  = nn.Parameter(torch.tensor(1.0)).to(device)

    with open("Results.txt", "a") as results_file:
        results_file.write("\nTraining with improved pooling, dropout, dynamic loss weighting\n")
        for epoch in range(1, 6):
            model.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch}")
            for batch in loop:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                sat_labels = batch["satire"].to(device)
                hum_labels = batch["humor"].to(device)

                sat_logits, hum_logits, embeddings = model(ids, mask)

                # Classification losses
                loss_sat = coral_loss(sat_logits, sat_labels)
                loss_hum = label_smooth_ce(hum_logits, hum_labels)

                # Entropy regularization
                sat_entropy = -(torch.sigmoid(sat_logits) * torch.log(torch.sigmoid(sat_logits) + 1e-12)).sum(1).mean()
                hum_entropy = -(F.softmax(hum_logits, dim=1) * F.log_softmax(hum_logits, dim=1)).sum(1).mean()

                # Contrastive loss
                contrastive_loss_sat = supervised_contrastive_loss(embeddings, sat_labels)
                contrastive_loss_hum = supervised_contrastive_loss(embeddings, hum_labels)

                # Total losses
                total_loss_sat = loss_sat - (model.synth_weight_satire.clamp(min=0.0) * sat_entropy) + (model.alpha_sat.clamp(min=0.0) * contrastive_loss_sat)
                total_loss_hum = loss_hum - (model.synth_weight_humor.clamp(min=0.0) * hum_entropy) + (model.alpha_hum.clamp(min=0.0) * contrastive_loss_hum)

                loss = lambda_satire * total_loss_sat + lambda_humor * total_loss_hum

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loop.set_postfix(
                    loss=loss.item(),
                    w_sat=model.synth_weight_satire.item(),
                    w_hum=model.synth_weight_humor.item(),
                    a_sat=model.alpha_sat.item(),
                    a_hum=model.alpha_hum.item()
                )

            # Evaluate on validation
            preds_val_sat, preds_val_hum, rep_val_sat, rep_val_hum = evaluate(model, val_loader, device)
            results_file.write(f"\nEpoch {epoch} Validation\n=== Satire Level ===\n{rep_val_sat}\n")
            results_file.write(f"=== Humor Attribute ===\n{rep_val_hum}\n")
            results_file.flush()

            # Save test predictions
            preds_sat, preds_hum, rep_sat, rep_hum = evaluate(model, test_loader, device)
            out_df = test_df.copy()
            out_df["Pred_Satire"] = preds_sat
            out_df["Pred_Humor"]  = preds_hum
            out_df.to_excel(f"Preds_epoch{epoch}.xlsx", index=False)

if __name__ == "__main__":
    main()



# import os
# import pandas as pd
# from tqdm import tqdm
# from sklearn.metrics import classification_report
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
# from torch.nn import functional as F
# import numpy as np

# # -------------------------------------------------------
# # 1. Dataset
# # -------------------------------------------------------
# class DHDataset(Dataset):
#     def __init__(self, df, tokenizer, max_len=128):
#         self.texts  = df["Comment"].astype(str).tolist()
#         self.satire = df["Satire Level New"].astype(int).tolist()
#         self.humor  = df["Humor Attribute New"].astype(int).tolist()
#         self.tokenizer = tokenizer
#         self.max_len   = max_len

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         enc = self.tokenizer(
#             self.texts[idx],
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_len,
#             return_tensors="pt",
#         )
#         item = {k: v.squeeze(0) for k, v in enc.items()}
#         item["satire"] = torch.tensor(self.satire[idx], dtype=torch.long)
#         item["humor"]  = torch.tensor(self.humor[idx], dtype=torch.long)
#         return item

# # -------------------------------------------------------
# # 2. Model with two learnable scalars
# # -------------------------------------------------------
# class MultiTaskModel(nn.Module):
#     def __init__(self, base_model="bert-base-uncased",
#                  satire_classes=3, humor_classes=5):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(base_model)
#         hidden = self.encoder.config.hidden_size
#         self.satire_head = nn.Linear(hidden, satire_classes - 1)  # CORAL
#         self.humor_head  = nn.Linear(hidden, humor_classes)
#         # Independent learnable regularizer weights
#         self.synth_weight_satire = nn.Parameter(torch.tensor(0.2))
#         self.synth_weight_humor  = nn.Parameter(torch.tensor(0.2))
        
#         # New learnable weights for contrastive loss
#         self.alpha_sat = nn.Parameter(torch.tensor(0.2))
#         self.alpha_hum = nn.Parameter(torch.tensor(0.2))

#     def forward(self, input_ids, attention_mask):
#         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         pooled = out.last_hidden_state[:, 0]
#         sat_logits = self.satire_head(pooled)
#         hum_logits = self.humor_head(pooled)
#         return sat_logits, hum_logits, pooled

# # -------------------------------------------------------
# # 3. Loss functions
# # -------------------------------------------------------
# def coral_loss(logits, targets, num_classes=3):
#     device = logits.device
#     k_minus_1 = logits.shape[1]
#     targets = targets.unsqueeze(1).expand(-1, k_minus_1)
#     t = (targets > torch.arange(k_minus_1, device=device).unsqueeze(0)).float()
#     return -(t * F.logsigmoid(logits) + (1 - t) * F.logsigmoid(-logits)).sum(1).mean()

# def label_smooth_ce(logits, targets, num_classes=5, eps=0.1):
#     one_hot = torch.zeros_like(logits).scatter(1, targets.unsqueeze(1), 1)
#     one_hot = one_hot * (1 - eps) + eps / num_classes
#     return -(one_hot * F.log_softmax(logits, dim=1)).sum(1).mean()

# def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
#     """
#     Supervised Contrastive Loss (InfoNCE) implementation
#     """
#     device = embeddings.device
#     batch_size = embeddings.size(0)
#     embeddings = F.normalize(embeddings, dim=1)

#     # Compute similarity matrix
#     sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

#     labels = labels.view(-1, 1)
#     mask = torch.eq(labels, labels.T).float().to(device)
#     mask.fill_diagonal_(0)  # exclude self-comparison

#     # For numerical stability
#     logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
#     logits = sim_matrix - logits_max.detach()

#     exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size, device=device))
#     log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

#     # Only keep positive samples
#     mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

#     loss = -mean_log_prob_pos.mean()
#     return loss


# # -------------------------------------------------------
# # 4. Evaluation
# # -------------------------------------------------------
# def evaluate(model, loader, device):
#     model.eval()
#     preds_sat, preds_hum, gts_sat, gts_hum = [], [], [], []
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Eval", leave=False):
#             ids = batch["input_ids"].to(device)
#             mask = batch["attention_mask"].to(device)
#             sat_logits, hum_logits, _ = model(ids, mask)
#             sat_pred = torch.sum(torch.sigmoid(sat_logits) > 0.5, dim=1)
#             hum_pred = hum_logits.argmax(1)
#             preds_sat.extend(sat_pred.cpu().numpy())
#             preds_hum.extend(hum_pred.cpu().numpy())
#             gts_sat.extend(batch["satire"].numpy())
#             gts_hum.extend(batch["humor"].numpy())
#     r_sat = classification_report(gts_sat, preds_sat, digits=4, zero_division=0)
#     r_hum = classification_report(gts_hum, preds_hum, digits=4, zero_division=0)
#     return preds_sat, preds_hum, r_sat, r_hum

# # -------------------------------------------------------
# # 5. Main training loop (updated)
# # -------------------------------------------------------
# def main():
#     train_df = pd.read_excel("DHD-Train.xlsx")
#     val_df   = pd.read_excel("DHD-Val.xlsx")
#     test_df  = pd.read_excel("DHD-Test.xlsx")

#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     train_ds  = DHDataset(train_df, tokenizer)
#     val_ds    = DHDataset(val_df, tokenizer)
#     test_ds   = DHDataset(test_df, tokenizer)

#     train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
#     test_loader  = DataLoader(test_ds, batch_size=32)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model  = MultiTaskModel().to(device)

#     # ----- Optimizer: two groups -----
#     main_params = [p for n, p in model.named_parameters() if "synth_weight" not in n and "alpha" not in n]
#     synth_params = [model.synth_weight_satire, model.synth_weight_humor]
#     alpha_params = [model.alpha_sat, model.alpha_hum]
    
#     optimizer = torch.optim.AdamW(
#         [
#             {"params": main_params,  "lr": 2e-5, "weight_decay": 0.01},
#             {"params": synth_params, "lr": 1e-3, "weight_decay": 0.0},
#             {"params": alpha_params, "lr": 1e-3, "weight_decay": 0.0}, # Separate LR for contrastive scalars
#         ]
#     )

#     total_steps = len(train_loader) * 5
#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer, num_warmup_steps=int(0.1 * total_steps),
#         num_training_steps=total_steps
#     )

#     lambda_humor = 1.0

#     with open("Results.txt", "a") as results_file:
#         results_file.write("\nTraining with Supervised Contrastive Learning and independent scalar updates\n")
#         for epoch in range(1, 6):
#             model.train()
#             loop = tqdm(train_loader, desc=f"Epoch {epoch}")
#             for batch in loop:
#                 ids = batch["input_ids"].to(device)
#                 mask = batch["attention_mask"].to(device)
#                 sat_labels = batch["satire"].to(device)
#                 hum_labels = batch["humor"].to(device)

#                 sat_logits, hum_logits, embeddings = model(ids, mask)

#                 # 1. Classification Losses
#                 loss_sat = coral_loss(sat_logits, sat_labels)
#                 loss_hum = label_smooth_ce(hum_logits, hum_labels)

#                 # 2. Regularization Losses (Entropy)
#                 sat_entropy = -(torch.sigmoid(sat_logits) * torch.log(torch.sigmoid(sat_logits) + 1e-12)).sum(1).mean()
#                 hum_entropy = -(F.softmax(hum_logits, dim=1) * F.log_softmax(hum_logits, dim=1)).sum(1).mean()
                
#                 # 3. Contrastive Losses
#                 contrastive_loss_sat = supervised_contrastive_loss(embeddings, sat_labels)
#                 contrastive_loss_hum = supervised_contrastive_loss(embeddings, hum_labels)

#                 # Combine all losses independently
#                 total_loss_sat = loss_sat - (model.synth_weight_satire.clamp(min=0.0) * sat_entropy) + (model.alpha_sat.clamp(min=0.0) * contrastive_loss_sat)
#                 total_loss_hum = loss_hum - (model.synth_weight_humor.clamp(min=0.0) * hum_entropy) + (model.alpha_hum.clamp(min=0.0) * contrastive_loss_hum)

#                 loss = total_loss_sat + lambda_humor * total_loss_hum

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 scheduler.step()

#                 loop.set_postfix(
#                     loss=loss.item(),
#                     w_sat=model.synth_weight_satire.item(),
#                     w_hum=model.synth_weight_humor.item(),
#                     a_sat=model.alpha_sat.item(),
#                     a_hum=model.alpha_hum.item()
#                 )

#             # ---- Evaluate on test each epoch ----
#             preds_sat, preds_hum, rep_sat, rep_hum = evaluate(model, test_loader, device)
#             results_file.write(f"\nEpoch {epoch}\n=== Satire Level ===\n{rep_sat}\n")
#             results_file.write(f"=== Humor Attribute ===\n{rep_hum}\n")
#             results_file.flush()

#             out_df = test_df.copy()
#             out_df["Pred_Satire"] = preds_sat
#             out_df["Pred_Humor"]  = preds_hum
#             out_df.to_excel(f"Preds_epoch{epoch}.xlsx", index=False)

# if __name__ == "__main__":
#     main()

# import os
# import pandas as pd
# from tqdm import tqdm
# from sklearn.metrics import classification_report
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
# from torch.nn import functional as F

# # -------------------------------------------------------
# # 1. Dataset
# # -------------------------------------------------------
# class DHDataset(Dataset):
#     def __init__(self, df, tokenizer, max_len=128):
#         self.texts  = df["Comment"].astype(str).tolist()
#         self.satire = df["Satire Level New"].astype(int).tolist()
#         self.humor  = df["Humor Attribute New"].astype(int).tolist()
#         self.tokenizer = tokenizer
#         self.max_len   = max_len

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         enc = self.tokenizer(
#             self.texts[idx],
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_len,
#             return_tensors="pt",
#         )
#         item = {k: v.squeeze(0) for k, v in enc.items()}
#         item["satire"] = torch.tensor(self.satire[idx], dtype=torch.long)
#         item["humor"]  = torch.tensor(self.humor[idx], dtype=torch.long)
#         return item

# # -------------------------------------------------------
# # 2. Model with two learnable scalars
# # -------------------------------------------------------
# class MultiTaskModel(nn.Module):
#     def __init__(self, base_model="bert-base-uncased",
#                  satire_classes=3, humor_classes=5):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(base_model)
#         hidden = self.encoder.config.hidden_size
#         self.satire_head = nn.Linear(hidden, satire_classes - 1)  # CORAL
#         self.humor_head  = nn.Linear(hidden, humor_classes)
#         # Independent learnable regularizer weights
#         self.synth_weight_satire = nn.Parameter(torch.tensor(0.2))
#         self.synth_weight_humor  = nn.Parameter(torch.tensor(0.2))

#     def forward(self, input_ids, attention_mask):
#         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         pooled = out.last_hidden_state[:, 0]
#         return self.satire_head(pooled), self.humor_head(pooled)

# # -------------------------------------------------------
# # 3. Loss functions
# # -------------------------------------------------------
# def coral_loss(logits, targets, num_classes=3):
#     device = logits.device
#     k_minus_1 = logits.shape[1]
#     targets = targets.unsqueeze(1).expand(-1, k_minus_1)
#     t = (targets > torch.arange(k_minus_1, device=device).unsqueeze(0)).float()
#     return -(t * F.logsigmoid(logits) + (1 - t) * F.logsigmoid(-logits)).sum(1).mean()

# def label_smooth_ce(logits, targets, num_classes=5, eps=0.1):
#     one_hot = torch.zeros_like(logits).scatter(1, targets.unsqueeze(1), 1)
#     one_hot = one_hot * (1 - eps) + eps / num_classes
#     return -(one_hot * F.log_softmax(logits, dim=1)).sum(1).mean()

# # -------------------------------------------------------
# # 4. Evaluation
# # -------------------------------------------------------
# def evaluate(model, loader, device):
#     model.eval()
#     preds_sat, preds_hum, gts_sat, gts_hum = [], [], [], []
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Eval", leave=False):
#             ids = batch["input_ids"].to(device)
#             mask = batch["attention_mask"].to(device)
#             sat_head, hum_head = model(ids, mask)
#             sat_pred = torch.sum(torch.sigmoid(sat_head) > 0.5, dim=1)
#             hum_pred = hum_head.argmax(1)
#             preds_sat.extend(sat_pred.cpu().numpy())
#             preds_hum.extend(hum_pred.cpu().numpy())
#             gts_sat.extend(batch["satire"].numpy())
#             gts_hum.extend(batch["humor"].numpy())
#     r_sat = classification_report(gts_sat, preds_sat, digits=4, zero_division=0)
#     r_hum = classification_report(gts_hum, preds_hum, digits=4, zero_division=0)
#     return preds_sat, preds_hum, r_sat, r_hum

# # -------------------------------------------------------
# # 5. Main training loop (updated)
# # -------------------------------------------------------
# def main():
#     train_df = pd.read_excel("DHD-Train.xlsx")
#     val_df   = pd.read_excel("DHD-Val.xlsx")
#     test_df  = pd.read_excel("DHD-Test.xlsx")

#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     train_ds  = DHDataset(train_df, tokenizer)
#     val_ds    = DHDataset(val_df, tokenizer)
#     test_ds   = DHDataset(test_df, tokenizer)

#     train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
#     test_loader  = DataLoader(test_ds, batch_size=32)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model  = MultiTaskModel().to(device)

#     # ----- Optimizer: two groups -----
#     main_params = [p for n, p in model.named_parameters()
#                    if "synth_weight" not in n]
#     scalar_params = [model.synth_weight_satire, model.synth_weight_humor]

#     optimizer = torch.optim.AdamW(
#         [
#             {"params": main_params,   "lr": 2e-5, "weight_decay": 0.01},
#             {"params": scalar_params, "lr": 1e-3, "weight_decay": 0.0},  # higher LR
#         ]
#     )

#     total_steps = len(train_loader) * 5
#     # Scheduler only for main group
#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer, num_warmup_steps=int(0.1 * total_steps),
#         num_training_steps=total_steps
#     )

#     lambda_humor = 1.0

#     with open("Results.txt", "a") as results_file:
#         results_file.write("\nTraining with higher LR for learnable scalars\n")
#         for epoch in range(1, 6):
#             model.train()
#             loop = tqdm(train_loader, desc=f"Epoch {epoch}")
#             for batch in loop:
#                 ids = batch["input_ids"].to(device)
#                 mask = batch["attention_mask"].to(device)
#                 sat_labels = batch["satire"].to(device)
#                 hum_labels = batch["humor"].to(device)

#                 sat_logits, hum_logits = model(ids, mask)

#                 loss_sat = coral_loss(sat_logits, sat_labels)
#                 loss_hum = label_smooth_ce(hum_logits, hum_labels)

#                 # Entropy regularization
#                 sat_entropy = -(torch.sigmoid(sat_logits) *
#                                  torch.log(torch.sigmoid(sat_logits) + 1e-12)).sum(1).mean()
#                 hum_entropy = -(F.softmax(hum_logits, dim=1) *
#                                  F.log_softmax(hum_logits, dim=1)).sum(1).mean()

#                 # Decouple the regularization by applying it to each task's loss
#                 # This ensures independent learning of the two scalars
#                 loss_sat_with_reg = loss_sat - (model.synth_weight_satire.clamp(min=0.0) * sat_entropy)
#                 loss_hum_with_reg = loss_hum - (model.synth_weight_humor.clamp(min=0.0) * hum_entropy)

#                 # The total loss is now the sum of the regularized losses
#                 loss = loss_sat_with_reg + lambda_humor * loss_hum_with_reg

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 scheduler.step()

#                 loop.set_postfix(
#                     loss=loss.item(),
#                     w_sat=model.synth_weight_satire.item(),
#                     w_hum=model.synth_weight_humor.item()
#                 )

#             # ---- Evaluate on test each epoch ----
#             preds_sat, preds_hum, rep_sat, rep_hum = evaluate(model, test_loader, device)
#             results_file.write(f"\nEpoch {epoch}\n=== Satire Level ===\n{rep_sat}\n")
#             results_file.write(f"=== Humor Attribute ===\n{rep_hum}\n")
#             results_file.flush()

#             out_df = test_df.copy()
#             out_df["Pred_Satire"] = preds_sat
#             out_df["Pred_Humor"]  = preds_hum
#             out_df.to_excel(f"Preds_epoch{epoch}.xlsx", index=False)

# if __name__ == "__main__":
#     main()


# import os
# import pandas as pd
# from tqdm import tqdm
# from sklearn.metrics import classification_report
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
# from torch.nn import functional as F

# # -------------------------------------------------------
# # 1. Dataset
# # -------------------------------------------------------
# class DHDataset(Dataset):
#     def __init__(self, df, tokenizer, max_len=128):
#         self.texts  = df["Comment"].astype(str).tolist()
#         self.satire = df["Satire Level New"].astype(int).tolist()
#         self.humor  = df["Humor Attribute New"].astype(int).tolist()
#         self.tokenizer = tokenizer
#         self.max_len   = max_len

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         enc = self.tokenizer(
#             self.texts[idx],
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_len,
#             return_tensors="pt",
#         )
#         item = {k: v.squeeze(0) for k, v in enc.items()}
#         item["satire"] = torch.tensor(self.satire[idx], dtype=torch.long)
#         item["humor"]  = torch.tensor(self.humor[idx], dtype=torch.long)
#         return item

# # -------------------------------------------------------
# # 2. Model with two learnable scalars
# # -------------------------------------------------------
# class MultiTaskModel(nn.Module):
#     def __init__(self, base_model="bert-base-uncased",
#                  satire_classes=3, humor_classes=5):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(base_model)
#         hidden = self.encoder.config.hidden_size
#         self.satire_head = nn.Linear(hidden, satire_classes - 1)  # CORAL
#         self.humor_head  = nn.Linear(hidden, humor_classes)
#         # Independent learnable regularizer weights
#         self.synth_weight_satire = nn.Parameter(torch.tensor(0.2))
#         self.synth_weight_humor  = nn.Parameter(torch.tensor(0.2))

#     def forward(self, input_ids, attention_mask):
#         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         pooled = out.last_hidden_state[:, 0]
#         return self.satire_head(pooled), self.humor_head(pooled)

# # -------------------------------------------------------
# # 3. Loss functions
# # -------------------------------------------------------
# def coral_loss(logits, targets, num_classes=3):
#     device = logits.device
#     k_minus_1 = logits.shape[1]
#     targets = targets.unsqueeze(1).expand(-1, k_minus_1)
#     t = (targets > torch.arange(k_minus_1, device=device).unsqueeze(0)).float()
#     return -(t * F.logsigmoid(logits) + (1 - t) * F.logsigmoid(-logits)).sum(1).mean()

# def label_smooth_ce(logits, targets, num_classes=5, eps=0.1):
#     one_hot = torch.zeros_like(logits).scatter(1, targets.unsqueeze(1), 1)
#     one_hot = one_hot * (1 - eps) + eps / num_classes
#     return -(one_hot * F.log_softmax(logits, dim=1)).sum(1).mean()

# # -------------------------------------------------------
# # 4. Evaluation
# # -------------------------------------------------------
# def evaluate(model, loader, device):
#     model.eval()
#     preds_sat, preds_hum, gts_sat, gts_hum = [], [], [], []
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Eval", leave=False):
#             ids = batch["input_ids"].to(device)
#             mask = batch["attention_mask"].to(device)
#             sat_head, hum_head = model(ids, mask)
#             sat_pred = torch.sum(torch.sigmoid(sat_head) > 0.5, dim=1)
#             hum_pred = hum_head.argmax(1)
#             preds_sat.extend(sat_pred.cpu().numpy())
#             preds_hum.extend(hum_pred.cpu().numpy())
#             gts_sat.extend(batch["satire"].numpy())
#             gts_hum.extend(batch["humor"].numpy())
#     r_sat = classification_report(gts_sat, preds_sat, digits=4, zero_division=0)
#     r_hum = classification_report(gts_hum, preds_hum, digits=4, zero_division=0)
#     return preds_sat, preds_hum, r_sat, r_hum

# # -------------------------------------------------------
# # 5. Main training loop
# # -------------------------------------------------------
# def main():
#     train_df = pd.read_excel("DHD-Train.xlsx")
#     val_df   = pd.read_excel("DHD-Val.xlsx")
#     test_df  = pd.read_excel("DHD-Test.xlsx")

#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     train_ds  = DHDataset(train_df, tokenizer)
#     val_ds    = DHDataset(val_df, tokenizer)
#     test_ds   = DHDataset(test_df, tokenizer)

#     train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
#     test_loader  = DataLoader(test_ds, batch_size=32)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model  = MultiTaskModel().to(device)

#     # ----- Optimizer: two groups -----
#     main_params = [p for n, p in model.named_parameters()
#                    if "synth_weight" not in n]
#     scalar_params = [model.synth_weight_satire, model.synth_weight_humor]

#     optimizer = torch.optim.AdamW(
#         [
#             {"params": main_params,   "lr": 2e-5, "weight_decay": 0.01},
#             {"params": scalar_params, "lr": 1e-3, "weight_decay": 0.0},  # higher LR
#         ]
#     )

#     total_steps = len(train_loader) * 5
#     # Scheduler only for main group
#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer, num_warmup_steps=int(0.1 * total_steps),
#         num_training_steps=total_steps
#     )

#     lambda_humor = 1.0

#     with open("Results.txt", "a") as results_file:
#         results_file.write("\nTraining with higher LR for learnable scalars\n")
#         for epoch in range(1, 6):
#             model.train()
#             loop = tqdm(train_loader, desc=f"Epoch {epoch}")
#             for batch in loop:
#                 ids = batch["input_ids"].to(device)
#                 mask = batch["attention_mask"].to(device)
#                 sat_labels = batch["satire"].to(device)
#                 hum_labels = batch["humor"].to(device)

#                 sat_logits, hum_logits = model(ids, mask)

#                 loss_sat = coral_loss(sat_logits, sat_labels)
#                 loss_hum = label_smooth_ce(hum_logits, hum_labels)

#                 # Entropy regularization
#                 sat_entropy = -(torch.sigmoid(sat_logits) *
#                                 torch.log(torch.sigmoid(sat_logits) + 1e-12)).sum(1).mean()
#                 hum_entropy = -(F.softmax(hum_logits, dim=1) *
#                                 F.log_softmax(hum_logits, dim=1)).sum(1).mean()

#                 reg = -(model.synth_weight_satire.clamp(min=0.0) * sat_entropy +
#                         model.synth_weight_humor.clamp(min=0.0) * hum_entropy)

#                 loss = loss_sat + lambda_humor * loss_hum + reg

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 scheduler.step()

#                 loop.set_postfix(
#                     loss=loss.item(),
#                     w_sat=model.synth_weight_satire.item(),
#                     w_hum=model.synth_weight_humor.item()
#                 )

#             # ---- Evaluate on test each epoch ----
#             preds_sat, preds_hum, rep_sat, rep_hum = evaluate(model, test_loader, device)
#             results_file.write(f"\nEpoch {epoch}\n=== Satire Level ===\n{rep_sat}\n")
#             results_file.write(f"=== Humor Attribute ===\n{rep_hum}\n")
#             results_file.flush()

#             out_df = test_df.copy()
#             out_df["Pred_Satire"] = preds_sat
#             out_df["Pred_Humor"]  = preds_hum
#             out_df.to_excel(f"Preds_epoch{epoch}.xlsx", index=False)

# if __name__ == "__main__":
#     main()



# import os
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import classification_report
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, AutoModel
# from torch.optim import AdamW
# from torch.nn import functional as F

# # -------------------------------------------------------
# # 1. Data
# # -------------------------------------------------------
# class DHDataset(Dataset):
#     def __init__(self, df, tokenizer, max_len=128):
#         self.texts = df["Comment"].astype(str).tolist()
#         self.satire = df["Satire Level New"].astype(int).tolist()
#         self.humor = df["Humor Attribute New"].astype(int).tolist()
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         enc = self.tokenizer(
#             self.texts[idx],
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_len,
#             return_tensors="pt",
#         )
#         item = {k: v.squeeze(0) for k, v in enc.items()}
#         item["satire"] = torch.tensor(self.satire[idx], dtype=torch.long)
#         item["humor"] = torch.tensor(self.humor[idx], dtype=torch.long)
#         # synthetic score placeholder: replace with real scores if available
#         item["synth"] = torch.rand(1).item()
#         return item

# # -------------------------------------------------------
# # 2. Model: shared encoder + two heads
# # -------------------------------------------------------
# class MultiTaskModel(nn.Module):
#     def __init__(self, base_model="google-bert/bert-base-uncased", satire_classes=3, humor_classes=5):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(base_model)
#         hidden = self.encoder.config.hidden_size
#         # CORAL: K-1 logits for satire (here 2 thresholds)
#         self.satire_head = nn.Linear(hidden, satire_classes - 1)
#         self.humor_head = nn.Linear(hidden, humor_classes)

#     def forward(self, input_ids, attention_mask):
#         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         pooled = out.last_hidden_state[:, 0]  # [CLS]
#         return self.satire_head(pooled), self.humor_head(pooled)

# # -------------------------------------------------------
# # 3. Loss functions
# # -------------------------------------------------------
# def coral_loss(logits, targets, num_classes=3):
#     # logits: (B, K-1)
#     # targets: (B,) with {0..K-1}  (assuming labels start at 0)
#     device = logits.device
#     batch_size, k_minus_1 = logits.shape
#     targets = targets.unsqueeze(1).expand(-1, k_minus_1)
#     # t_{i,k} = 1 if y_i > k else 0
#     t = (targets > torch.arange(k_minus_1, device=device).unsqueeze(0)).float()
#     log_p = F.logsigmoid(logits)
#     log_not_p = F.logsigmoid(-logits)
#     loss = -(t * log_p + (1 - t) * log_not_p).sum(dim=1).mean()
#     return loss

# def label_smooth_ce(logits, targets, num_classes=5, eps=0.1):
#     one_hot = torch.zeros_like(logits).scatter(1, targets.unsqueeze(1), 1)
#     one_hot = one_hot * (1 - eps) + eps / num_classes
#     log_probs = F.log_softmax(logits, dim=1)
#     return -(one_hot * log_probs).sum(dim=1).mean()

# # -------------------------------------------------------
# # 4. Training / evaluation
# # -------------------------------------------------------
# def evaluate(model, loader, device):
#     model.eval()
#     preds_sat, preds_hum, gts_sat, gts_hum = [], [], [], []
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Eval", leave=False):
#             ids = batch["input_ids"].to(device)
#             mask = batch["attention_mask"].to(device)
#             sat_head, hum_head = model(ids, mask)
#             # CORAL prediction: count thresholds passed
#             sat_pred = torch.sum(torch.sigmoid(sat_head) > 0.5, dim=1)
#             hum_pred = hum_head.argmax(1)
#             preds_sat.extend(sat_pred.cpu().numpy())
#             preds_hum.extend(hum_pred.cpu().numpy())
#             gts_sat.extend(batch["satire"].numpy())
#             gts_hum.extend(batch["humor"].numpy())

#     report_sat = classification_report(
#         gts_sat, preds_sat, digits=4, zero_division=0
#     )
#     report_hum = classification_report(
#         gts_hum, preds_hum, digits=4, zero_division=0
#     )
#     return preds_sat, preds_hum, report_sat, report_hum

# def main():
#     # Paths
#     train_path = "DHD-Train.xlsx"
#     val_path   = "DHD-Val.xlsx"
#     test_path  = "DHD-Test.xlsx"  # make sure this exists

#     tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

#     train_df = pd.read_excel(train_path)
#     val_df   = pd.read_excel(val_path)
#     test_df  = pd.read_excel(test_path)

#     train_ds = DHDataset(train_df, tokenizer)
#     val_ds   = DHDataset(val_df, tokenizer)
#     test_ds  = DHDataset(test_df, tokenizer)

#     train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
#     val_loader   = DataLoader(val_ds, batch_size=32)
#     test_loader  = DataLoader(test_ds, batch_size=32)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MultiTaskModel().to(device)
#     optimizer = AdamW(model.parameters(), lr=2e-5)

#     lambda_humor = 1.0
#     beta_synth   = 0.3
#     gamma_cons   = 0.2  # augmentation consistency skipped for brevity

#     results_file = open("Results.txt", "w")

#     for epoch in range(1, 6):
#         model.train()
#         loop = tqdm(train_loader, desc=f"Epoch {epoch}")
#         for batch in loop:
#             ids = batch["input_ids"].to(device)
#             mask = batch["attention_mask"].to(device)
#             sat_labels = batch["satire"].to(device)
#             hum_labels = batch["humor"].to(device)
#             synth_score = batch["synth"].to(device)

#             sat_logits, hum_logits = model(ids, mask)

#             loss_sat = coral_loss(sat_logits, sat_labels)
#             loss_hum = label_smooth_ce(hum_logits, hum_labels)
#             # synthetic-aware entropy reg (maximize entropy => minimize -entropy)
#             probs = F.softmax(hum_logits, dim=1)
#             entropy = -(probs * probs.log()).sum(1)
#             synth_reg = (synth_score * -entropy).mean()

#             loss = loss_sat + lambda_humor * loss_hum + beta_synth * synth_reg
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             loop.set_postfix(loss=loss.item())

#         # --- Evaluation on test after each epoch ---
#         preds_sat, preds_hum, rep_sat, rep_hum = evaluate(model, test_loader, device)
#         results_file.write(f"\nEpoch {epoch}\n")
#         results_file.write("=== Satire Level ===\n")
#         results_file.write(rep_sat + "\n")
#         results_file.write("=== Humor Attribute ===\n")
#         results_file.write(rep_hum + "\n")
#         results_file.flush()

#         # Save predictions to Excel for this epoch
#         out_df = test_df.copy()
#         out_df["Pred_Satire"] = preds_sat
#         out_df["Pred_Humor"]  = preds_hum
#         out_df.to_excel(f"Preds_epoch{epoch}.xlsx", index=False)

#     results_file.close()

# if __name__ == "__main__":
#     main()
