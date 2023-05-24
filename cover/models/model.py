import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import (MulticlassAUROC, MulticlassAUPRC, MulticlassRecall, 
                               MulticlassPrecision, MulticlassConfusionMatrix, 
                               MulticlassAccuracy, MulticlassF1Score)

from util import set_random_seed
set_random_seed(42)

class FeedForward(nn.Module):
    def __init__(self,dim,hidden,dropout,input_shape=None,bias=True):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=bias)
        self.w2 = nn.Linear(hidden, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.norm = nn.BatchNorm1d(dim if input_shape is None else input_shape[-1])
        self.do = nn.AlphaDropout(dropout)

    def forward(self, x):
        return self.w2(F.selu(self.w1(self.do(self.norm(x)))) * self.w3(x))

class FC(nn.Module):
    def __init__(self,dim_in,dim_out,dropout,bias=True):
        super().__init__()
        self.w = nn.Linear(dim_in, dim_out, bias=bias)
        self.do = nn.AlphaDropout(dropout)
        self.norm = nn.BatchNorm1d(dim_in)

    def forward(self, x):
        return self.do(F.selu(self.w(self.norm(x))))

class EntityEmbed(nn.Module):
    def __init__(self,input_shape,ee_dim,bias=False):
        super().__init__()
        self.w = nn.ModuleList([
            nn.Linear(1, ee_dim, bias=bias) for i in range(input_shape[-1])])
        self.norm = nn.BatchNorm1d(input_shape[-1])

    def forward(self, x):
        return self.norm(torch.stack([self.w[i](col) for i, col in enumerate(
                torch.tensor_split(x, x.shape[1], dim=1))], dim=1))

class Transformer(nn.Module):
    def __init__(self,input_shape,ee_dim,num_heads,dropout,bias):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=ee_dim, num_heads=num_heads, 
                                         dropout=dropout, bias=bias, batch_first=True)
        self.norm = nn.BatchNorm1d(input_shape[-1])
        self.ff = FeedForward(ee_dim,ee_dim//2,dropout=0.0,bias=False,input_shape=input_shape)
    def forward(self, x):
        x = self.norm(x)
        x = self.mha(x, x, x, need_weights=False)[0] + x
        return F.selu(self.ff(x)) + x

class CovertypeModel(nn.Module):
    def __init__(self, input_shape, output_shape, dropout, ee_dim, num_heads, name, seed):
        print(f'name:{name},input_shape:{input_shape}:output_shape{output_shape}' )
        self.seed = seed
        set_random_seed(self.seed)
        super(CovertypeModel, self).__init__()
        self.output_shape = output_shape[0]
        self.binary = False # compatibility with other experiments
        self.metrics = self.get_metrics(self.output_shape)
        self.name = name
        self.norm = nn.BatchNorm1d(input_shape)
        self.ee = EntityEmbed(input_shape, ee_dim, bias=True)
        self.tf1 = Transformer(input_shape, ee_dim, num_heads, bias=False, dropout=dropout)
        self.tf2 = Transformer(input_shape, ee_dim, num_heads, bias=False, dropout=dropout)
        self.tf3 = Transformer(input_shape, ee_dim, num_heads, bias=False, dropout=dropout)
        self.tf4 = Transformer(input_shape, ee_dim, num_heads, bias=False, dropout=dropout)
        self.tf5 = Transformer(input_shape, ee_dim, num_heads, bias=False, dropout=dropout)
        self.tf6 = Transformer(input_shape, ee_dim, num_heads, bias=False, dropout=dropout)
        shape = input_shape[-1]*ee_dim
        self.layer1 = FC(shape, shape*4, dropout=dropout)
        self.layer2 = FeedForward(shape*4, shape, dropout=dropout)
        self.layer3 = FC(shape*4, shape*2, dropout=dropout)
        self.layer4 = FeedForward(shape*2, shape//2, dropout=dropout)
        self.layer5 = FC(shape*2, shape, dropout=dropout)
        self.norm_out = nn.BatchNorm1d(shape)
        self.fc_out = nn.Linear(shape, self.output_shape)

    def forward(self, x):
        x = self.norm(x)
        x = self.ee(x)
        x = self.tf1(x)
        x = self.tf2(x)
        x = self.tf3(x)
        x = self.tf4(x)
        x = self.tf5(x)
        x = self.tf6(x)
        x = torch.flatten(x, start_dim=1)
        x = self.layer1(x)
        x = self.layer2(x)+x
        x = self.layer3(x)
        x = self.layer4(x)+x
        x = self.layer5(x)
        x = self.norm_out(x)
        x = self.fc_out(x)
        return x

    @staticmethod
    def get_metrics(output_shape):
        # output_shape may be larger than num_classes due to clients missing classes
        metrics = [MulticlassF1Score(num_classes=output_shape, average='weighted'),
                   MulticlassAUROC(num_classes=output_shape),
                   MulticlassAUPRC(num_classes=output_shape),
                   MulticlassRecall(num_classes=output_shape, average='weighted'),
                   MulticlassPrecision(num_classes=output_shape, average='weighted'),
                   MulticlassConfusionMatrix(num_classes=output_shape),
                   MulticlassAccuracy(num_classes=output_shape, average='macro')]
        # slice gives us just the class name
        metrics = {str(m.__module__)[33:]:m for m in metrics}
        return metrics

class FullyFederatedCovertypeModel(CovertypeModel):
    def __init__(self, *args, **kwargs):
        super(FullyFederatedCovertypeModel, self).__init__(*args, **kwargs)
        self.global_layers = list(self.children())[1:]

class PartiallyPersonalizedCovertypeModel(CovertypeModel):
    def __init__(self, *args, **kwargs):
        super(PartiallyPersonalizedCovertypeModel, self).__init__(*args, **kwargs)
        self.global_layers = [
                              self.layer2, 
                              self.layer3,
                              self.layer4,
                              ]

def load_covertype_models(cover_trains, seed, personalize, ee_dim=16, num_heads=8, dropout=0.0, device='cpu'):
    if personalize:
        Model = PartiallyPersonalizedCovertypeModel
    else:
        Model = FullyFederatedCovertypeModel
    print(f'--personalize set to {personalize}; loading {Model.__name__} ...')
    set_random_seed(seed)
    models = dict()
    for split in cover_trains.keys():
        model = Model(input_shape=cover_trains[split].dataset.get_input_shape(),
                      output_shape=cover_trains[split].dataset.get_output_shape(),
                      dropout=dropout, 
                      ee_dim=ee_dim, 
                      num_heads=num_heads,
                      name=f'cover_{split}',
                      seed=seed)
        models[split] = model.to(device)

    return models