from abc import ABC, abstractmethod
import torch
import time


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Model(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit_iterator_one_epoch(self, iterator):
        pass

    @abstractmethod
    def fit_batch(self, iterator):
        pass

    @abstractmethod
    def evaluate_iterator(self, iterator):
        pass

    def update_from_model(self, model):
        """
        update parameters using gradients from another model
        :param model: Model() object, gradients should be precomputed;
        """
        for param_idx, param in enumerate(self.net.parameters()):
            param.grad = list(model.net.parameters())[param_idx].grad.data.clone()

        self.optimizer.step()
        self.lr_scheduler.step()

    def fit_batches(self, iterator, n_steps):
        global_loss = 0
        global_metrics = [0] * len(self.metrics)  

        for step in range(n_steps):
            batch_loss, batch_metrics, batch_gradients = self.fit_batch(iterator)
            global_loss += batch_loss
            for i, metric in enumerate(batch_metrics):
                global_metrics[i] += metric

        
        return global_loss / n_steps, [metric / n_steps for metric in global_metrics], batch_gradients


    def fit_iterator(self, train_iterator, val_iterator=None, n_epochs=1, path=None, verbose=0):
        best_valid_loss = float('inf')

        for epoch in range(n_epochs):

            start_time = time.time()

            train_loss, train_metrics = self.fit_iterator_one_epoch(train_iterator)
            if val_iterator:
                valid_loss, valid_metrics = self.evaluate_iterator(val_iterator)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if val_iterator:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    if path:
                        torch.save(self.net, path)

            if verbose:
                print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train RMSE: {train_metrics[0]:.3f} | Train MAE: {train_metrics[1]:.3f}')
                if val_iterator:
                    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. RMSE: {valid_metrics[0]:.3f} | Val. MAE: {valid_metrics[1]:.3f}')


    def get_param_tensor(self):
        param_list = []

        for param in self.net.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)
