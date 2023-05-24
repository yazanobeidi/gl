import torch
from functools import reduce

class FederatedTrainer(object):
    def __init__(self):
        pass

    @staticmethod
    def _get_weights(models):
        with torch.no_grad():
            res = dict()
            for split, model in models.items():
                params = list()
                for gp in model.global_layers:
                    for p in gp.parameters():
                        if p.data.dim()==1:
                            params.append(p.data.unsqueeze(0))
                        else:
                            params.append(p.data)
                res[split] = torch.nested.nested_tensor(params)
            return res

    @staticmethod
    def _fed_avg(weights):
        with torch.no_grad():
            avg = reduce(torch.add, weights)/len(weights)
            return [p.squeeze(0) for p in avg]

    @staticmethod
    def _set_weights(models, new_weights):
        with torch.no_grad():
            for split, model in models.items():
                ix = 0
                for gp in model.global_layers:
                    for name, p in gp.named_parameters():
                        try:
                            p.data.copy_(new_weights[ix])
                        except:
                            print('\n\n\n\n\n\nERROR COPYING',
                             p.shape, new_weights[ix].shape, name, p, gp)
                            raise
                        ix+=1