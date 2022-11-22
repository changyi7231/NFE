import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, model_name, num_entity, num_relation, dimension, gamma):
        super(Model, self).__init__()
        self.model_name = model_name
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.dimension = dimension
        self.gamma = gamma

        if model_name == 'NFE-2':
            self.entity = nn.Embedding(num_entity, 3*dimension)
            self.relation = nn.Embedding(num_relation, 2*dimension)
        else:
            self.entity = nn.Embedding(num_entity, 2*dimension)
            self.relation = nn.Embedding(num_relation, 2*dimension)

        bound = 0.1
        nn.init.uniform_(self.entity.weight, -bound, bound)
        nn.init.uniform_(self.relation.weight, -bound, bound)

    def forward(self, heads, relations):
        if self.model_name == 'NFE-1':
            h = self.entity(heads)
            r = self.relation(relations)

            h_mu, h_sigma = torch.chunk(h, 2, dim=-1)
            r_mu, r_sigma = torch.chunk(r, 2, dim=-1)
            t_mu, t_sigma = torch.chunk(self.entity.weight, 2, dim=-1)

            mu1 = r_sigma * h_mu + r_mu
            mu2 = t_mu
            sigma1 = (r_sigma * h_sigma).abs()
            sigma2 = t_sigma.abs()

            score = -2*torch.matmul(mu1, mu2.t()) + mu1.pow(2).sum(-1).unsqueeze(-1) + mu2.pow(2).sum(-1).unsqueeze(0)
            score = score + (-2*torch.matmul(sigma1, sigma2.t()) + sigma1.pow(2).sum(-1).unsqueeze(-1) + sigma2.pow(2).sum(-1).unsqueeze(0))
            return self.gamma-score

        elif self.model_name == 'NFE-2':
            h = self.entity(heads)
            r = self.relation(relations)

            h_mu, h_sigma1, h_sigma2 = torch.chunk(h, 3, dim=-1)
            r_mu, r_sigma = torch.chunk(r, 2, dim=-1)
            t_mu, t_sigma1, t_sigma2 = torch.chunk(self.entity.weight, 3, dim=-1)

            mu1 = r_sigma * h_mu + r_mu
            mu2 = t_mu
            sigma1 = (r_sigma * h_sigma1).abs()
            sigma2 = (r_sigma * h_sigma2).abs()
            sigma3 = t_sigma1.abs()
            sigma4 = t_sigma2.abs()
            # constant = (3 / 4) ** 0.5
            constant = (2/3.14159265359) ** 0.5
            score = -2 * torch.matmul(mu1, mu2.t()) + mu1.pow(2).sum(-1).unsqueeze(-1) + mu2.pow(2).sum(-1).unsqueeze(0)
            score = score + (-torch.matmul(sigma1, sigma3.t()) + sigma1.pow(2).sum(-1).unsqueeze(-1) / 2 + sigma3.pow(2).sum(-1).unsqueeze(0) / 2)
            score = score + (-torch.matmul(sigma2, sigma4.t()) + sigma2.pow(2).sum(-1).unsqueeze(-1) / 2 + sigma4.pow(2).sum(-1).unsqueeze(0) / 2)
            score = score + constant * ((mu1 * (sigma2 - sigma1)).sum(-1).unsqueeze(-1) + torch.matmul(mu1, (sigma3 - sigma4).t()) - torch.matmul(sigma2 - sigma1, mu2.t()) - (mu2 * (sigma3 - sigma4)).sum(-1).unsqueeze(0))
            return self.gamma-score

        elif self.model_name == 'NFE-3':
            h = self.entity(heads)
            r = self.relation(relations)

            h_mu, h_sigma = torch.chunk(h, 2, dim=-1)
            r_mu, r_sigma = torch.chunk(r, 2, dim=-1)
            t_mu, t_sigma = torch.chunk(self.entity.weight, 2, dim=-1)

            mu1 = r_sigma * h_mu + r_mu
            mu2 = t_mu
            sigma1 = F.softplus(r_sigma) * F.softplus(h_sigma)
            sigma2 = F.softplus(t_sigma)

            score = torch.matmul(sigma1.pow(2), sigma2.pow(-2).t()) / 2 - torch.sum(sigma1.log(), dim=-1).unsqueeze(-1) + torch.sum(sigma2.log(), dim=-1).unsqueeze(0) - self.dimension / 2
            score = score + (-torch.matmul(mu1, (mu2*sigma2.pow(-2)).t()) + torch.matmul(mu1.pow(2), sigma2.pow(-2).t()) / 2 + (mu2/sigma2).pow(2).sum(-1).unsqueeze(0)/2)
            return self.gamma-score

        elif self.model_name == 'NFE-w/o-uncertainty':
            h = self.entity(heads)
            r = self.relation(relations)

            h_mu, h_sigma = torch.chunk(h, 2, dim=-1)
            r_mu, r_sigma = torch.chunk(r, 2, dim=-1)
            t_mu, t_sigma = torch.chunk(self.entity.weight, 2, dim=-1)

            mu1 = r_sigma * h_mu + r_mu
            mu2 = t_mu

            score = -2*torch.matmul(mu1, mu2.t()) + mu1.pow(2).sum(-1).unsqueeze(-1) + mu2.pow(2).sum(-1).unsqueeze(0)
            return self.gamma-score

        elif self.model_name == 'NFE-sigma':
            h = self.entity(heads)
            r = self.relation(relations)

            h_mu, h_sigma = torch.chunk(h, 2, dim=-1)
            r_mu, r_sigma = torch.chunk(r, 2, dim=-1)
            t_mu, t_sigma = torch.chunk(self.entity.weight, 2, dim=-1)

            sigma1 = (r_sigma * h_sigma).abs()
            sigma2 = t_sigma.abs()

            score = -2 * torch.matmul(sigma1, sigma2.t()) + sigma1.pow(2).sum(-1).unsqueeze(-1) + sigma2.pow(2).sum(-1).unsqueeze(0)
            return self.gamma - score

        elif self.model_name == 'TransE':
            h = self.entity(heads)
            r = self.relation(relations)

            score = -2 * torch.matmul(h + r, self.entity.weight.t()) + (h + r).pow(2).sum(-1).unsqueeze(-1) + self.entity.weight.pow(2).sum(-1).unsqueeze(0)
            return self.gamma - score

        elif self.model_name == 'DistMult':
            h = self.entity(heads)
            r = self.relation(relations)

            score = torch.matmul(h * r, self.entity.weight.t())
            return score

        else:
            raise ValueError('wrong model')
