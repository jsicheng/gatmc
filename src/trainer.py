import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, dataset, data, calc_rmse, optimizer):
        self.model = model
        self.dataset = dataset
        self.data = data
        self.calc_rmse = calc_rmse
        self.optimizer = optimizer

    def training(self, epochs):
        self.epochs = epochs
        for epoch in range(self.epochs):
            loss, train_rmse = self.train_one(epoch)
            test_rmse = self.test()
            self.summary(epoch, loss, train_rmse, test_rmse)
        print('END TRAINING')

    def train_one(self, epoch):
        self.model.train()
        out = self.model(self.data.x, self.data.edge_index,
                         self.data.edge_type, self.data.edge_norm, self.data)
        loss = F.cross_entropy(out[self.data.train_idx], self.data.train_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        rmse = self.calc_rmse(out[self.data.train_idx], self.data.train_gt)
        return loss.item(), rmse.item()

    def test(self):
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index, 
                         self.data.edge_type, self.data.edge_norm)
        rmse = self.calc_rmse(out[self.data.test_idx], self.data.test_gt)
        return rmse.item()

    def summary(self, epoch, loss, train_rmse=None, test_rmse=None):
        if test_rmse is None:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} ]'.format(
                epoch, self.epochs, loss))
        else:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} | RMSE: {:.6f} | Test RMSE: {:.6f} ]'.format(
                epoch, self.epochs, loss, train_rmse, test_rmse))
