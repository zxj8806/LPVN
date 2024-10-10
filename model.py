import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import args

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		x = inputs
		x = torch.mm(x,self.weight)
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

		
class LPVN(nn.Module):
	def __init__(self, adj):
		super(LPVN, self).__init__()
		self.num = args.n
		# Define separate GCN layers for a and b for each node
		self.gcns_a1 = nn.ModuleList([GraphConvSparse(args.input_dim, args.hidden1_dim, adj) for _ in range(self.num)])
		self.gcns_a2 = nn.ModuleList([GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj) for _ in range(self.num)])
		self.gcns_b1 = nn.ModuleList([GraphConvSparse(args.input_dim, args.hidden1_dim, adj) for _ in range(self.num)])
		self.gcns_b2 = nn.ModuleList([GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj) for _ in range(self.num)])

	def forward(self, X):
		a_vectors = []
		b_vectors = []
		for i in range(self.num):
			a_i = self.gcns_a2[i](self.gcns_a1[i](X))
			b_i = self.gcns_b2[i](self.gcns_b1[i](X))
			# Calculate the mean of each output (mean over all nodes)
			#mean_a_i = a_i.mean(dim=0, keepdim=True)  # Mean across the node dimension
			#mean_b_i = b_i.mean(dim=0, keepdim=True)

			# Subtract the mean from each output to normalize
			a_i = a_i - a_i.mean(dim=0, keepdim=True)
			b_i = b_i - b_i.mean(dim=0, keepdim=True)

			a_vectors.append(a_i)
			b_vectors.append(b_i)

		# Compute A_pred
		A_pred = torch.zeros(X.shape[0], X.shape[0], device=X.device)

		for i in range(self.num):
			#print("torch.mm(a_vectors[i], b_vectors[i].t()).shape", torch.mm(a_vectors[i], b_vectors[i].t()).shape)

			#print("torch.mm(b_vectors[i], a_vectors[i].t().shape", torch.mm(b_vectors[i], a_vectors[i].t()).shape)
			#print("torch.ger(a_vectors[i], b_vectors[i]).shape", torch.ger(a_vectors[i], b_vectors[i]).shape)

			A_pred += torch.mm(a_vectors[i], b_vectors[i].t()) + torch.mm(b_vectors[i], a_vectors[i].t())

		return torch.sigmoid(A_pred)