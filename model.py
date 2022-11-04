import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError

class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.dataset = dataset
        self.device = config['device']
        self.t = config['t']
        self.sim_type = config['sim_type']
        self.bpr_sim_type = config['bpr_sim_type']
        self.test_sim_type = config['test_sim_type']
        self.total_epoch = config['epochs']
        self.epsilon = config['epsilon']
        self.epoch = 0
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        print("using Normal distribution N(0, 0.1) initialization for PureMF")
    
    def computer(self):
        """
        propagate methods for lightGCN
        if we need to propagate in inference, we need this one.
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        g_droped = self.prop_Graph    
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        '''
        This is used for the inference stage only
        '''
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        if self.test_sim_type  == 'cos':
            users_emb = users_emb.unsqueeze(-1)
            items_emb = items_emb.repeat(users_emb.shape[0], 1, 1)
            items_emb = torch.transpose(items_emb, dim0=1, dim1=2)
            rating = torch.cosine_similarity(users_emb, items_emb, dim=1)
        else:
            scores = torch.matmul(users_emb, items_emb.t())
            rating = self.f(scores)
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users = self.embedding_user.weight
        all_items = self.embedding_item.weight
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        # users_emb_ego = self.embedding_user(users)
        # pos_emb_ego = self.embedding_item(pos_items)
        # neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb, pos_emb, neg_emb
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        if self.bpr_sim_type  == 'ip':
            pos_scores = torch.mul(users_emb, pos_emb)
            pos_scores = torch.sum(pos_scores, dim=1)
            neg_scores = torch.mul(users_emb, neg_emb)
            neg_scores = torch.sum(neg_scores, dim=1)
        elif self.bpr_sim_type  == 'cos':
            pos_scores = torch.cosine_similarity(users_emb, pos_emb)
            neg_scores = torch.cosine_similarity(users_emb, neg_emb)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss

    def softmax_loss(self, users, items, weights=False):
        eps = 1e-8
        gamma = torch.log(torch.FloatTensor([1 + self.epoch / self.epsilon])).to(self.device)
        users_emb = self.embedding_user(users.long())
        items_emb = self.embedding_item(items.long())
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) / float(len(users)))
        
        if self.sim_type  == 'cos':
            users_emb = users_emb.unsqueeze(-1)
            items_emb = torch.transpose(items_emb, dim0=1, dim1=2)
            scores = torch.cosine_similarity(users_emb, items_emb, dim=1)
        elif self.sim_type == 'non_sym':
            items_emb = items_emb.div(items_emb.norm(2, dim=2, keepdim=True))
            users_emb = users_emb.unsqueeze(-1)
            scores = torch.bmm(items_emb, users_emb)
            scores = scores.squeeze(-1)
        else:
            users_emb = users_emb.unsqueeze(-1)
            scores = torch.bmm(items_emb, users_emb)
            scores = scores.squeeze(-1)

        pos_logits = torch.exp(scores[:, 0] / self.t)
        neg_logits = torch.exp(scores[:, 1:] / self.t)
        Ng = neg_logits.sum(dim=-1)
        prob = pos_logits / Ng
        if weights:
            w = prob.detach().pow(gamma).reshape(-1, 1)
            loss = (torch.mul(w, -torch.log(prob))).mean()
        else:
            loss = (-torch.log(prob)).mean()
        return loss, reg_loss
    


