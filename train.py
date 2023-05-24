import torch
from tqdm import tqdm
from torcheval.metrics.toolkit import reset_metrics
from contextlib import contextmanager
import os
from importlib import import_module


from util import set_random_seed, get_tensorboard, multiline_tqdm, import_model_loader
set_random_seed(42)
from federated import FederatedTrainer

class Trainer(FederatedTrainer):
    def __init__(self, run_description, batches, ee_dim, seed, 
                 federate, drop_last, device, log_metadata, 
                 force_consistent_feature_space,
                 force_consistent_target_space, personalize, 
                 basepath, model_filename, dataset_loader, 
                 learning_rate, weight_decay, name):
        self.seed = seed
        set_random_seed(self.seed)
        self.federate = federate
        self.batches = batches
        self.drop_last = drop_last
        if not self.drop_last:
            print(f'drop_last set to {self.drop_last}; running with model.eval()')
        self.trains, self.tests, self.vals = dataset_loader(self.batches,
                                                            drop_last=self.drop_last,
                                                            seed=self.seed,
                                                            device=device,
                        force_consistent_feature_space=force_consistent_feature_space,
                        force_consistent_target_space=force_consistent_target_space)
        self.model_loader = import_model_loader(model_filename, name)
        self.models = self.model_loader(self.trains, ee_dim=ee_dim, seed=self.seed,
                                        personalize=personalize, device=device)
        self.names = {ix:name for ix, name in enumerate(self.models.keys())}
        self.loss_fn = [torch.nn.BCEWithLogitsLoss() if m.binary else torch.nn.CrossEntropyLoss()
                        for m in self.models.values()]
        self.optimizer = [torch.optim.AdamW(m.parameters(), lr=learning_rate, 
                                 weight_decay=weight_decay) for m in self.models.values()]
        self.metrics = [m.metrics for m in self.models.values()]
        self.tb = get_tensorboard(run_description, name, basepath)
        self.name, self.run_description = name, run_description
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        if log_metadata:
            self.add_graph_and_embed_to_tb()
            self.show_devices()
        self.pbar_update =  {'base': 'Epoch {epoch}: ',
                             'msg': '{metric}: train {train:.3f} valid {valid:.3f} '}

    def test_or_validation(self, test_or_val_set):
        val_losses = list()
        for ix, (split, dataloader) in enumerate(test_or_val_set.items()):
            running_vloss = 0.
            for i, (inputs, labels) in enumerate(dataloader):
                outputs = self.models[split](inputs)
                try:
                    if self.models[split].binary:
                        # torch binary cross entropy expects float() targets with same dim as model output (with batch dim)
                        # while, torch multi class cross entropy expects long() targets and flattened
                        # This discrepency seems like a bug. We resolve it by manually casting.
                        outputs = outputs.squeeze().view(labels.shape)
                        loss = self.loss_fn[ix](outputs, labels.float())
                    else:
                        loss = self.loss_fn[ix](outputs, labels)
                except:
                    print('error on', split, outputs, labels, outputs.dim(), self.trains[split].dataset.y.unique(), self.loss_fn[ix])
                    raise
                running_vloss += loss
                for metric in self.metrics[ix].values():
                    metric.update(outputs, labels)
            val_losses.append(running_vloss/(i+1))
        metrics = self.collect_metrics(val_losses)
        return metrics
    
    @contextmanager
    def training(self):
        for m in self.models.values():
            m.train(True)
        self.eval_if_needed()
        try:
            yield
        finally:
            for m in self.models.values():
                m.train(False)

    def train(self, epochs):
        with multiline_tqdm(total=epochs, leave=False) as pbar:
            for epoch in range(1, epochs+1):
                with self.training():
                    train_metrics = self.train_one_epoch()
                with torch.no_grad():
                    val_metrics = self.test_or_validation(self.vals)
                self.update_tb(epoch, train_metrics, val_metrics)
                msg = self.get_pbar_update(epoch, train_metrics, val_metrics)
                pbar.set_description(msg)
                pbar.update(1)
                if self.federate:
                    agg_weights = self._get_weights(self.models)
                    agg_weights = self._fed_avg(agg_weights.values())
                    self._set_weights(self.models, agg_weights)
        print(f'\nFinal epoch metrics:\n{msg}\n')

    def train_one_epoch(self):
        losses = [0.]*len(self.models)
        for i, data in enumerate(zip(*list(self.trains.values()))):
            for ix in torch.randperm(len(data)):
                name = self.names[int(ix)]
                model = self.models[name]
                inputs, labels = data[ix][0], data[ix][1]
                try:
                    outputs = model(inputs)
                except:
                    print('error on', name, )#inputs.shape, inputs, labels.shape, labels)
                    raise
                try:
                    if model.binary:
                        # torch binary cross entropy expects float() targets with same dim as model output (with batch dim)
                        # while, torch multi class cross entropy expects long() targets and flattened
                        # This discrepency seems like a bug. We resolve it by manually casting.
                        outputs = outputs.squeeze()#.view(labels.shape)
                        loss = self.loss_fn[ix](outputs, labels.float())
                    else:
                        loss = self.loss_fn[ix](outputs, labels)
                except:
                    print('error on', name, outputs, labels, outputs.dim(), self.trains[name].dataset.y.unique(), self.loss_fn[ix])
                    raise
                self.optimizer[ix].zero_grad()
                loss.backward()
                self.optimizer[ix].step()
                losses[ix] += loss.item()
                for metric in self.metrics[ix].values():
                    metric.update(outputs, labels)

        losses = [g/float(self.batches) for g in losses]
        metrics = self.collect_metrics(losses)
        return metrics

    def update_tb(self, epoch, train_metrics, val_metrics):
        #we add confusion matrix as an Image
        #https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard/42857070#42857070
        for split, train_metric in train_metrics.items():
            for train, val in zip(train_metric.items(), val_metrics[split].items()):
                name, val_metric = val
                _, train_metric = train
                if 'conf' in name:
                    train_metric = train_metric.view((1,)+train_metric.shape+(1,))
                    val_metric = val_metric.view((1,)+val_metric.shape+(1,))
                    self.tb.add_image(f'{split} {name}',train_metric,epoch+1, dataformats='NHWC')
                    self.tb.add_image(f'{split} {name}',val_metric,epoch+1, dataformats='NHWC')
                    continue
                try:
                    self.tb.add_scalars(f'{split} {name}',
                                            {'Training': train_metric, 
                                             'Validation': val_metric},
                                             epoch+1)
                except:
                    print(f'failed on {split} {name}')
                    raise

    def collect_metrics(self, losses=None):
        collected = dict()
        for ix, model in self.names.items():
            collected[model] = {k:v.compute() for k,v in self.metrics[ix].items()}
            if losses:
                collected[model]['loss'] = losses[ix]
            for m in self.metrics[ix].values():
                m.reset()
        return collected

    def get_pbar_update(self, epoch, train_metrics, val_metrics):
        msg = self.pbar_update['base'].format(epoch=epoch)
        for split, train_metric in train_metrics.items():
            for train, val in zip(train_metric.items(), val_metrics[split].items()):
                name, val_metric = val
                _, train_metric = train
                try:
                    if 'conf' in name:
                        continue
                    msg += f'{split}:' + self.pbar_update['msg'].format(metric=name, 
                                train=train_metric, valid=val_metric)
                except:
                    print('error on', split, name, train_metric, val_metric)
                    raise
        return msg

    def add_graph_and_embed_to_tb(self, debug=True):
        tb_log_dir = self.tb.log_dir
        #tensorboard overwrite all but the latest passed add_graph().
        #this is so we can register multiple graphs
        # see https://github.com/tensorflow/tensorflow/issues/9512
        # and https://github.com/pytorch/pytorch/issues/32651
        self.tb.close()
        for ix, split in enumerate(self.trains.keys()):
            try:
                _, data = next(iter(enumerate(self.trains[split])))
                self.tb.log_dir = tb_log_dir+split+'_graph'
                self.tb.add_graph(self.models[split], data[0], verbose=False)
                if debug: print(f'added graph for {split} to {self.tb.log_dir}')
                self.tb.close()
                self.tb.log_dir = tb_log_dir
            except:
                print(f'failed to add graph to tb for {split}')
                raise
        for ix, split in enumerate(self.trains.keys()):
            try:
                metadata = {'labels': self.trains[split].dataset.y,
                            'client': self.trains[split].dataset.n_samples*[ix]}
                self.tb.add_embedding(self.trains[split].dataset.X, 
                                      metadata=list(zip(*list(metadata.values()))),
                                      metadata_header=list(metadata.keys()),
                                      tag=f'{split}_embeddings', global_step='embeddings')
                if debug: print(f'added embedding for {split} to {self.tb.log_dir}')
            except:
                print(f'failed to add embedding to tb for {split}')
                raise

    def show_devices(self):
        for k,v in self.models.items():
            print(f'{k}model:{next(v.parameters()).device}')
        for k,v in self.trains.items():
            print(f'{k}Xtrain:{v.dataset.X.device}',
                  f'{k}ytrain:{v.dataset.y.device}')
        for k,v in self.vals.items():
            print(f'{k}Xval:{v.dataset.X.device}',
                  f'{k}yval:{v.dataset.y.device}')

    def force_move_devices(self, device):
        self.models = {k:v.to(device) for k,v in self.models.items()}
        for s in [self.trains, self.vals]:
            for k,v in s.items():
                d = v.dataset
                d.X = d.X.to(device)
                d.y = d.y.to(device)

    def eval_if_needed(self):
        if self.drop_last:
            [m.eval() for m in self.models.values()]

    def test(self):
        with torch.no_grad():
            test_metrics = self.test_or_validation(self.tests)
        msg = 'Test results:\n'
        for ix, (split, test_metric) in enumerate(test_metrics.items()):
            msg += f'\n{split}:\n'
            for name, metric in test_metric.items():
                try:
                    if 'conf' in name:
                        maxlen = len(str(int(torch.max(metric).item())))
                        classes = list(range(0, self.models[split].output_shape+1))
                        classes = torch.tensor(classes, requires_grad=False)
                        metric = torch.vstack((classes[1:].unsqueeze(0), metric))
                        metric = torch.hstack((classes.unsqueeze(1), metric))
                        metric = '\n'.join([' '.join([f'{int(q)}'.rjust(maxlen) for q in m])\
                                            for m in metric.tolist()])
                        msg += f'{name}\n{metric}\n'
                    else:
                        msg += f'{name} {metric:.5f}\n'
                except:
                    print(split, name, metric)
                    raise

        print(msg)
        write_dir = os.path.join(os.path.dirname(os.path.dirname(self.tb.log_dir)),'test_results.txt')
        print(f'saving test results to {write_dir}')
        path_to_model_def = self.model_loader.__globals__["__file__"]
        with open(write_dir, 'w+') as f:
            f.write(f'Experiment: {self.name}: {self.run_description}\n\n'
                    f'learning rate: {self.learning_rate}, weight_decay:{self.weight_decay}\n\n'
                    f'Trained on model: {path_to_model_def}\n\n{msg}')

        # cast to numpy numeric
        for split, metrics in test_metrics.items():
            for name, metric in metrics.items():
                test_metrics[split][name] = metric.tolist()

        return test_metrics
