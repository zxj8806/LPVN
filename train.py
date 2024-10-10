import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time

from input_data import load_data
from preprocessing import *
import args
import model

# Train on CPU (hide GPU) due to memory constraints
#os.environ['CUDA_VISIBLE_DEVICES'] = ""

print("torch.cuda.is_available()", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

adj, features = load_data(args.dataset)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)

adj_orig.eliminate_zeros()
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create Model
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)


adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                            torch.FloatTensor(adj_norm[1]), 
                            torch.Size(adj_norm[2])).to(device)
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                            torch.FloatTensor(adj_label[1]), 
                            torch.Size(adj_label[2])).to(device)
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                            torch.FloatTensor(features[1]), 
                            torch.Size(features[2])).to(device)

weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0)) 
weight_tensor[weight_mask] = pos_weight

# init model and optimizer
model = getattr(model,args.model)(adj_norm)
optimizer = Adam(model.parameters(), lr=args.learning_rate)


def get_scores(edges_pos, edges_neg, adj_rec):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:

        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def contrastive_loss(A_pred, adj_label, margin=1.0):
    A_pred = torch.sigmoid(A_pred.view(-1))
    adj_label = adj_label.to_dense().view(-1)

    positive_pairs = A_pred[adj_label == 1]
    negative_pairs = A_pred[adj_label == 0]

    positive_loss = torch.mean(1 - positive_pairs)

    negative_loss = torch.mean(negative_pairs)

    loss = positive_loss + margin * negative_loss

    return loss

def focal_loss(A_pred, adj_label, gamma=2, alpha=0.25):
    A_pred = torch.sigmoid(A_pred.view(-1))
    adj_label = adj_label.to_dense().view(-1)
    
    BCE_loss = F.binary_cross_entropy(A_pred, adj_label, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    
    return torch.mean(F_loss)

def weighted_bce_loss(A_pred, adj_label, norm=1):
    A_pred = torch.sigmoid(A_pred.view(-1))
    adj_label = adj_label.to_dense().view(-1)

    bce_loss = F.binary_cross_entropy(A_pred, adj_label, reduction='none')

    errors = torch.abs(A_pred - adj_label)

    weights = 1 + errors

    weighted_bce = weights * bce_loss

    loss = norm * torch.mean(weighted_bce)

    return loss

def infoNCE_loss(z, adj_labels, temperature=0.1):
    pos_mask = adj_labels.to_dense() == 1
    neg_mask = adj_labels.to_dense() == 0
    
    sim_matrix = torch.matmul(z, z.T) / temperature
    sim_matrix = torch.exp(sim_matrix)

    pos_sum = torch.sum(sim_matrix * pos_mask, dim=1)
    all_sum = torch.sum(sim_matrix, dim=1)
    
    loss = -torch.log(pos_sum / all_sum).mean()
    return loss

def focal_loss(inputs, targets, alpha=0.5, gamma=2):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    targets = targets.float()
    at = alpha * targets + (1 - alpha) * (1 - targets)
    pt = torch.exp(-BCE_loss)
    F_loss = at * (1 - pt)**gamma * BCE_loss
    return F_loss.mean()

def contrastive_bce_loss(A_pred, adj_label, weight_tensor, alpha=0.5):
    bce_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
    
    pos_pairs = torch.where(adj_label.to_dense() > 0)
    neg_pairs = torch.where(adj_label.to_dense() == 0)

    pos_sim = torch.mean(A_pred[pos_pairs[0], pos_pairs[1]])
    neg_sim = torch.mean(A_pred[neg_pairs[0], neg_pairs[1]])

    contrast_loss = -torch.log(pos_sim + 1e-8) + torch.log(1 - neg_sim + 1e-8)  # 避免数值稳定问题

    combined_loss = (1 - alpha) * bce_loss + alpha * contrast_loss

    return combined_loss

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def enhanced_loss(output, target, margin=0.1, lambda_margin=0.5):
    prob = sigmoid(output)
    
    bce_loss = F.binary_cross_entropy(prob, target, reduction='mean')
    
    positive_mask = target == 1
    negative_mask = target == 0
    
    positive_loss = F.relu((0.5 + margin) - prob[positive_mask])
    
    negative_loss = F.relu(prob[negative_mask] - (0.5 - margin))
    
    margin_loss = torch.mean(positive_loss + negative_loss)

    combined_loss = (1 - lambda_margin) * bce_loss + lambda_margin * margin_loss
    
    return combined_loss

def weighted_contrastive_loss(A_pred, adj_label, weight_tensor):
    A_pred_flat = A_pred.view(-1)
    adj_label_flat = adj_label.to_dense().view(-1)
    weight_tensor_flat = weight_tensor

    bce_loss = F.binary_cross_entropy(A_pred_flat, adj_label_flat, reduction='none')
    
    weighted_loss = bce_loss * weight_tensor_flat

    return weighted_loss.mean()




def contrastive_enhanced_loss(A_pred, adj_label, weight_tensor, pos_threshold=0.7, neg_threshold=0.3, scale=0.5):
    A_pred_flat = A_pred.view(-1)
    adj_label_flat = adj_label.to_dense().view(-1)
    weight_tensor_flat = weight_tensor

    bce_loss = F.binary_cross_entropy(A_pred_flat, adj_label_flat, reduction='none')

    positive_samples = adj_label_flat == 1
    negative_samples = adj_label_flat == 0

    pos_penalty = F.relu(pos_threshold - A_pred_flat) ** 2
    neg_penalty = F.relu(A_pred_flat - neg_threshold) ** 2

    loss = torch.where(positive_samples, weight_tensor_flat * (bce_loss + scale * pos_penalty),
                       weight_tensor_flat * (bce_loss + scale * neg_penalty))

    return loss.mean()

weight_tensor = weight_tensor.to(device)
model = model.to(device)
features = features.to(device)
# train model
for epoch in range(args.num_epoch):
    t = time.time()

    A_pred = model(features)
    optimizer.zero_grad()
                          
    loss = contrastive_enhanced_loss(A_pred, adj_label, weight_tensor)
    
    #if args.model == 'LPVN':
    #    kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)**2).sum(1).mean()
    #    loss -= kl_divergence

    loss.backward()
    optimizer.step()

    train_acc = get_acc(A_pred,adj_label)
    A_pred = A_pred.cpu()

    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
          "val_ap=", "{:.5f}".format(val_ap),
          "time=", "{:.5f}".format(time.time() - t))


test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))