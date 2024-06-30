import torch
import torch.nn as nn
import torch.nn.functional as F
from options import prepare_train_args
PARSER = prepare_train_args()
TEMPERATURE=PARSER.temperature

def nt_xent_loss(out_1, out_2, temperature=0.8):
    batch_size = out_1.shape[0]
    out = torch.cat([out_1, out_2], dim=0)
    sim_matrix = nn.functional.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=2)
    sim_matrix = sim_matrix / temperature
    sim_labels = torch.arange(batch_size).to(out.device)
    sim_labels = torch.cat([sim_labels, sim_labels], dim=0)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(sim_matrix, sim_labels)
    return loss


def info_nce_loss( features,temperature=0.9):
    labels = torch.cat(
        [torch.arange(PARSER.batch_size) for i in range(PARSER.n_views)],
        dim=0,
    )
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(PARSER.device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     PARSER.n_views * PARSER.batch_size, PARSER.n_views * PARSER.batch_size)
    # assert similarity_matrix.shape == labels.shape
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(PARSER.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(
        similarity_matrix.shape[0], -1
    )
    # assert similarity_matrix.shape == labels.shape
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(
        similarity_matrix.shape[0], -1
    )
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(PARSER.device)
    logits = logits / temperature
    loss = nn.CrossEntropyLoss(logits,labels)
    return loss