import torch
import torch.nn as nn
import torch.nn.functional as F
from gate import Gate, GateMul
import math
import os.path as osp
import time
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.datasets import Entities
from rGAT import rgatconv
from torch_geometric.utils import coalesce, cumsum, to_torch_csc_tensor, to_scipy_sparse_matrix
def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type, use_residual=False, args=None):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type
        self.use_residual = use_residual
        self.weight = nn.Parameter(torch.FloatTensor(self.in_dim, self.in_dim))
        if use_residual:
            self.linear_h0 = nn.Linear(args.embed_dim, self.in_dim)
            nn.init.xavier_uniform_(self.linear_h0.weight)

        self.reset_parameters()

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        self.layer_normalize = nn.LayerNorm(self.out_dim)

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            if self.use_residual:
                self.linear_h = nn.Linear(self.in_dim * 2, self.in_dim)
                nn.init.xavier_uniform_(self.linear_h.weight)
                self.linear = nn.Linear(self.in_dim, self.out_dim)
                nn.init.xavier_uniform_(self.linear.weight)
            else:
                self.linear = nn.Linear(
                    self.in_dim * 2, self.out_dim)
                nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(
                self.in_dim, self.out_dim)
            self.linear2 = nn.Linear(
                self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        elif self.aggregator_type == 'gin':
            hidden_dim = args.mlp_hidden_dim
            self.num_layers = args.n_mlp_layers
            self.weight = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))

            self.linear_h0 = nn.Linear(args.embed_dim, hidden_dim)
            nn.init.xavier_uniform_(self.linear_h0.weight)

            if self.num_layers == 1:
                # Linear model
                self.linear = nn.Linear(self.in_dim, self.out_dim)
            else:
                # Multi-layer model
                self.inp_linear = torch.nn.Linear(self.in_dim, hidden_dim)
                self.linears = torch.nn.ModuleList()
                self.mlp_layer_norms = torch.nn.ModuleList()

                for layer in range(self.num_layers - 1):
                    self.linears.append(nn.Linear(hidden_dim, hidden_dim))

                self.out_linear = nn.Linear(hidden_dim, self.out_dim)

                for layer in range(self.num_layers - 1):
                    self.mlp_layer_norms.append(nn.LayerNorm(hidden_dim))

        else:
            raise NotImplementedError

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_dim)
        self.weight.data.uniform_(-stdv, stdv)

    def residual_connection(self, hi, h0, lamda, alpha, l):

        if self.use_residual:
            h0 = self.linear_h0(h0)
            residual = (1 - alpha) * hi + alpha * h0
            beta = math.log(lamda / l + 1)
            identity_mapping = (1 - beta) + beta * self.weight
            return torch.mm(residual, identity_mapping)
        else:
            return hi

    
    def forward(self, ego_embeddings, A_in, all_layers, lamda, alpha, l):
        """
        ego_embeddings:  (n_heads + n_tails, in_dim)
        A_in:            (n_heads + n_tails, n_heads + n_tails), torch.sparse.FloatTensor
        """
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            hi = ego_embeddings + side_embeddings
            embeddings = self.residual_connection(hi, all_layers[0], lamda, alpha, l)
            embeddings = self.activation(self.linear(embeddings))
        elif self.aggregator_type == 'gin':
            hi = ego_embeddings + side_embeddings
            h = self.inp_linear(ego_embeddings)
            layer_embeds = [h]
            if self.num_layers == 1:
                # Linear model
                hi = self.linear(hi)
                layer_embeds.append(hi)
            else:
                # If MLP
                h = self.inp_linear(hi)
                for layer in range(self.num_layers - 1):
                    h = self.mlp_layer_norms[layer](self.activation(self.linears[layer](h)))
                    layer_embeds.append(h)

            X = torch.sum(torch.stack(layer_embeds), dim=0)
            X = self.residual_connection(X, all_layers[0], lamda, alpha, l)

            embeddings = self.activation(self.out_linear(X))

            if len(all_layers) > 1:
                layer_embeds = [self.layer_normalize(embeddings)]

                for index, layer in enumerate(all_layers):
                    if index != 0:
                        layer_embeds.append(layer)

                embeddings = torch.sum(torch.stack(layer_embeds), dim=0)

        # (n_heads + n_tails, out_dim)
        embeddings = self.message_dropout(self.layer_normalize(embeddings))

        return embeddings

class RGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations):
        super().__init__()
        self.conv1 = rgatconv(in_channels, hidden_channels,num_relations +1 )
        self.conv2 = rgatconv(hidden_channels, hidden_channels, num_relations +1 )
       
        #self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type):
       
        x = self.conv1(x, edge_index, edge_type).relu()
        x , (adj, alpha) = self.conv2(x, edge_index, edge_type, return_attention_weights =True)
        x = F.relu(x)
        #x = self.lin(x)
        # print(f'fking x: {x.size()}')
        # print(f'fking edge_index: {adj.size()}')
        # print(f'fking edge_type: {alpha.size()}')
        
        # raise SystemExit()
    
        return x, (adj, alpha) # F.log_softmax(x, dim=-1)
class LiteralKG(nn.Module):

    def __init__(self, args, n_entities, n_relations, A_in=None,  numerical_literals=None, text_literals=None):

        super(LiteralKG, self).__init__()
        self.use_pretrain = args.use_pretrain
        self.args = args



        self.device = args.device
      
        
        # self.edge_index = edge_index.to(self.device)
        # self.edge_type = edge_type.to(self.device)

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.scale_gat_dim = args.scale_gat_dim

        # Use residual connection
        self.use_residual = args.use_residual
        self.alpha = args.alpha
        self.lamda = args.lamda

        self.aggregation_type = args.aggregation_type
        self.n_layers = args.n_conv_layers
        # self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.conv_dim_list = [args.embed_dim] + [args.conv_dim] * self.n_layers

        self.total_conv_dim = sum([self.conv_dim_list[i] for i in range(self.n_layers + 1)])

        # self.mess_dropout = eval(args.mess_dropout)
        self.mess_dropout = [args.mess_dropout] * self.n_layers

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.prediction_l2loss_lambda = args.fine_tuning_l2loss_lambda

        self.pre_training_neg_rate = args.pre_training_neg_rate
        self.fine_tuning_neg_rate = args.fine_tuning_neg_rate

        # Num. Literal
        # num_ent x n_num_lit
        # self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()
        self.n_num_lit = args.num_lit_dim
        # Txt. Literal
        # num_ent x n_txt_lit
        # self.text_literals = Variable(torch.from_numpy(text_literals)).cuda()
        self.n_txt_lit = args.txt_lit_dim

        self.entity_embed = nn.Embedding(self.n_entities, self.embed_dim)

        print(f"self.n_relations, self.relation_dim: {self.n_relations}, {self.relation_dim} ")
        self.relation_embed = nn.Embedding(self.n_relations+ 1, self.relation_dim)
 
        if self.scale_gat_dim is not None:
            self.linear_gat = nn.Linear(self.total_conv_dim, self.scale_gat_dim)
            self.gat_activation = nn.LeakyReLU()
            nn.init.xavier_uniform_(self.linear_gat.weight)

        nn.init.xavier_uniform_(self.entity_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)

        self.aggregator_layers = nn.ModuleList()

        self.numerical_literals_embed = numerical_literals
        self.text_literals_embed = text_literals

        # LiteralE's g
        if self.args.use_num_lit and self.args.use_txt_lit:
            self.emb_mul_lit = GateMul(self.embed_dim, self.n_num_lit, self.n_txt_lit)
        elif self.args.use_num_lit:
            self.emb_num_lit = Gate(self.embed_dim, self.n_num_lit)
        elif self.args.use_txt_lit:
            self.emb_txt_lit = Gate(self.embed_dim, self.n_txt_lit)

        for k in range(self.n_layers):
            self.aggregator_layers.append(
                Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k],
                           self.aggregation_type, self.use_residual, args))
 
        
        self.A_in = nn.Parameter( torch.sparse.FloatTensor(self.n_entities, self.n_entities))
         
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False

        self.milestone_score = args.milestone_score

        self.fc1 = nn.Linear(self.scale_gat_dim * 2, 32)
        self.norm1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.norm2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1)
        
    def gate_embeddings(self):
        ent_emb = self.entity_embed.weight  # torch.Size([101187, 300])
        # if self.args.use_num_lit and self.args.use_txt_lit:
        #     self.numerical_literals_embed = self.numerical_literals_embed.to(self.device)
        #     self.text_literals_embed = self.text_literals_embed.to(self.device)
        #     return self.emb_mul_lit(ent_emb, self.numerical_literals_embed, self.text_literals_embed)
        
        if self.args.use_num_lit: # torch.Size([101187, 6])
            self.numerical_literals_embed = self.numerical_literals_embed.to(self.device)
            return self.emb_num_lit(ent_emb, self.numerical_literals_embed)
        # elif self.args.use_txt_lit:
        #     self.text_literals_embed = self.text_literals_embed.to(self.device)
        #     return self.emb_txt_lit(ent_emb, self.text_literals_embed)

        return ent_emb

    def gat_embeddings(self):

        ent_lit_mul_r = self.gate_embeddings() # initial node embeddings
        #print(f"debug 1  ent_lit_mul_r: {ent_lit_mul_r.shape}")
        #raise SystemExit()
        #print(f"model_bce.py: initial embeddings: {ent_lit_mul_r.size()}")  # [101187, 300])                torch.Size([595172, 300])
        #print(f"model_bce.py: self.A_in: {self.A_in.size()}")               # [101187, 101187]              torch.Size([595172, 595172])

        all_embed = [ent_lit_mul_r]
        #gat_embed, (adj, alpha) = self.rgat(ent_lit_mul_r,self.edge_index, self.edge_type)
        # self.adj = adj
        # self.alpha = alpha
        # return gat_embed
    
        
        #print(f"gat_embed: {gat_embed.size()}") 
        for idx, layer in enumerate(self.aggregator_layers):
            ent_lit_mul_r = layer(ent_lit_mul_r, self.A_in, all_embed, self.lamda, self.alpha, idx + 1)
            norm_embed = F.normalize(ent_lit_mul_r, p=2, dim=1)
            all_embed.append(norm_embed)
        if self.scale_gat_dim is not None:
            gat_embed = self.linear_gat(torch.cat(all_embed, dim=1))
            gat_embed = self.gat_activation(gat_embed)
            return gat_embed
        else:
            # (n_heads + n_tails, concat_dim)
            return torch.cat(all_embed, dim=1)

    def calculate_prediction_loss(self, head_ids, tail_pos_ids, tail_neg_ids):
        """
        head_ids:       (prediction_batch_size)
        tail_pos_ids:   (prediction_batch_size)
        tail_neg_ids:   (prediction_batch_size)
        """
        gat_embed = self.gat_embeddings()           # (n_heads + n_tails, concat_dim)

        print(f"debug gat_embed: {gat_embed.shape}")
  
    
        head_embed = gat_embed[head_ids]            # (batch_size, concat_dim)
        tail_pos_embed = gat_embed[tail_pos_ids]    # (batch_size, concat_dim)
        tail_neg_embed = gat_embed[tail_neg_ids]    # (batch_size, concat_dim)

        # head_embed = self.gat_embeddings(head_ids)  # (batch_size, concat_dim)
        # tail_pos_embed = self.gat_embeddings(tail_pos_ids)  # (batch_size, concat_dim)
        # tail_neg_embed =self.gat_embeddings(tail_neg_ids)  # (batch_size, concat_dim)

        pos_score = torch.sum(head_embed * tail_pos_embed, dim=1)  # (batch_size)
        # print("Compare the positive score and negative score")
        # print(pos_score)

        neg_score = torch.sum(head_embed * tail_neg_embed,dim=1)  # (batch_size)
        # print(neg_score)

        # prediction_loss = F.softplus(neg_score - pos_score)
        prediction_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        prediction_loss = torch.mean(prediction_loss)

        l2_loss = _L2_loss_mean(
            head_embed) + _L2_loss_mean(tail_pos_embed) + _L2_loss_mean(tail_neg_embed)
        loss = prediction_loss + self.prediction_l2loss_lambda * l2_loss
        return loss

    def calc_triplet_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)            # (kg_batch_size, relation_dim)
        # h_embed = self.entity_embed(h)            # (kg_batch_size, embed_dim)
        # pos_t_embed = self.entity_embed(pos_t)    # (kg_batch_size, embed_dim)
        # neg_t_embed = self.entity_embed(neg_t)    # (kg_batch_size, embed_dim)

        # GAT embeddings
        gat_embed = self.gat_embeddings()  # (n_heads + n_tails, concat_dim)
        # print(f'calc_triplet_loss gat_embed: {gat_embed.size()} ') #torch.Size([595172, 300]) 
        # print(f'calc_triplet_loss r_embed: {r_embed.size()} ') # torch.Size([348, 300])
        head_embed = gat_embed[h]           # (batch_size, concat_dim)   # [348, 300]
        tail_pos_embed = gat_embed[pos_t]   # (batch_size, concat_dim)
        tail_neg_embed = gat_embed[neg_t]   # (batch_size, concat_dim)

        # print(f'calc_triplet_loss head_embed: {head_embed.size()} ')            # [348, 300]
        # print(f'calc_triplet_loss tail_pos_embed: {tail_pos_embed.size()} ')    # [348, 300]
        # print(f'calc_triplet_loss tail_neg_embed: {tail_neg_embed.size()} ')    # [348, 300]
        # print(f'calc_triplet_loss h: {h.size()} ')
        # print(f'calc_triplet_loss h: {pos_t.size()} ')
        # print(f'calc_triplet_loss h: {neg_t.size()} ')

        # Trans E
        # Equation (1)
        pos_score = torch.sum(
            torch.pow(head_embed + r_embed - tail_pos_embed, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(head_embed + r_embed - tail_neg_embed, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        # triplet_loss = F.softplus(pos_score - neg_score)
        triplet_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)

        triplet_loss = torch.mean(triplet_loss)

        l2_loss = _L2_loss_mean(head_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(tail_pos_embed) + _L2_loss_mean(
            tail_neg_embed)
        loss = triplet_loss + self.kg_l2loss_lambda * l2_loss

        return loss

    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]

        h_embed = self.entity_embed.weight[h_list]
        t_embed = self.entity_embed.weight[t_list]

        v_list = torch.sum(t_embed * torch.tanh(h_embed + r_embed), dim=1)
        return v_list


    #relations: list(data.laplacian_dict.keys()) [r]: {h,t}

    def convert_coo2tensor(self, adj,alpha):
        values = alpha.cpu()
        indices = adj.cpu()

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        return torch.sparse.FloatTensor(i, v)
    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device
        rows = []
        cols = []
        values = []
        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]
            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)
        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

        # rGAT
        # self.alpha = self.alpha.view(-1)
        # A_in = self.convert_coo2tensor(self.adj, self.alpha )
        # #A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        # self.A_in.data = A_in.to(device)

    def calc_score(self, head_ids, tail_ids):
        all_embed = self.gat_embeddings()  # (n_heads + n_tails, concat_dim)
        head_embed = all_embed[head_ids]  # (n_heads, concat_dim)
        tail_embed = all_embed[tail_ids]  # (n_items, concat_dim)

        # head_embed = self.gat_embeddings(head_ids)  # (n_heads, concat_dim)
        # tail_embed = self.gat_embeddings(tail_ids)  # (n_items, concat_dim)

        prediction_score = torch.matmul(
            head_embed, tail_embed.transpose(0, 1))  # (n_heads, n_items)

        # print(prediction_score)

        return prediction_score

    def train_MLP(self, head_ids, tail_ids):
        gat_embed = self.gat_embeddings()  # (n_heads + n_tails, concat_dim)

        head_embed = gat_embed[head_ids]  # (batch_size, concat_dim)
        tail_embed = gat_embed[tail_ids]  # (batch_size, concat_dim)

        x = torch.cat([head_embed, tail_embed], dim=1)  # (batch_size)

        x = self.norm1(torch.relu(self.fc1(x)))
        x = self.norm2(torch.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))

        return x

    def forward(self, *input, device, mode):
        self.device = device
        if mode == 'fine_tuning':
            return self.calculate_prediction_loss(*input)
        if mode == 'pre_training':
            return self.calc_triplet_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'mlp':
            return self.train_MLP(*input)

