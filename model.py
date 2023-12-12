import pandas as pd
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm.auto import tqdm

def read_data(P,Q,J):
    dates_split = list(pd.read_csv("dates_split.csv",index_col=0).T.values[0])
    kdf = pd.read_csv("K_bert.csv",index_col=0).iloc[:,:P]
    vdf = pd.read_csv("V_bert.csv",index_col=0).iloc[:,:Q]
    ydf = pd.read_csv("Y_bert.csv",index_col=0).iloc[:,:J]
    K = torch.tensor(kdf.values, dtype=torch.float32)
    V = torch.tensor(vdf.values, dtype=torch.float32)
    Y = torch.tensor(ydf.values, dtype=torch.float32)
    return dates_split,K,V,Y
    
class LSTMModel(nn.Module):
    def __init__(self, K, V, Y, U, H, batch_first=True):
        super(LSTMModel, self).__init__()
        self.K = K; self.V = V; self.Y = Y; self.U = U; self.H = H
        self.T, self.P = K.shape; self.T, self.Q = V.shape; self.T, self.J = Y.shape
        
        self.S = nn.Parameter(torch.zeros(self.P, self.J))
        self.gru = nn.LSTM(self.P, self.H)
        self.linear = nn.Linear(self.H, 1)
        
    def forward(self, idx):
        X = torch.zeros([len(idx),self.T,self.Q])
        for b,j in enumerate(idx):
            X[b] = self.calculate_market_status(j)
        gru_out, _ = self.gru(X)
        predictions = self.linear(gru_out)
        return predictions
    
    def calculate_market_status(self, j):
        M = torch.zeros(self.T, self.Q)
        W = self.K@self.S[:,j]
        for t in range(1,len(self.dates)):
            alpha = F.softmax(W[self.dates[t-1]:self.dates[t]],dim=0)
            for u,a in enumerate(alpha):
                M[t-1,:] += a * self.V[self.dates[t-1]:self.dates[t],:][u,:]
        return M

class GRUModel(nn.Module):
    def __init__(self, K, V, Y, dates_split, hidden_size = 1, batch_first=True):
        super(GRUModel, self).__init__()
        # input data
        self.K = K; self.V = V; self.Y = Y; self.dates = dates_split
        # hyper-parameters
        self.hidden_size = hidden_size
        # shape of data
        self.P = K.shape[1]; self.Q = V.shape[1]; self.T, self.J = Y.shape
        # stock embeddings
        self.S = nn.Parameter(torch.zeros(self.P, self.J))
        # structure of the model
        self.gru = nn.GRU(self.Q, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, 1)
        
    def forward(self, idx):
        # consturct input (market status) for selected indices
        X = torch.zeros([len(idx),self.T,self.Q])
        for b,j in enumerate(idx):
            X[b] = self.calculate_market_status(j)
        # compute the output for the input
        gru_out, _ = self.gru(X)
        predictions = self.linear(gru_out)
        return predictions
    
    def calculate_market_status(self, j):
        M = torch.zeros(self.T, self.Q)
        W = self.K@self.S[:,j]
        for t in range(1,len(self.dates)):
            idx1 = self.dates[t-1]
            idx2 = self.dates[t]
            alpha = F.softmax(W[idx1:idx2],dim=0)
            for u,a in enumerate(alpha):
                M[t-1,:] += a * self.V[self.dates[t-1]:self.dates[t],:][u,:]
        return M
    
def construct_model(model_type, K, V, Y, dates_split, hidden_size, learning_rate,loss_type):
    
    # define a loss function
    if loss_type == "L1":
        loss_function = nn.L1Loss()
    if loss_type == "MSE":
        loss_function = nn.MSELoss()
    
    # define a model
    if model_type == "LSTM":
        model = LSTMModel(K, V, Y)
    if model_type == "GRU":
        model = GRUModel(K, V, Y, dates_split, hidden_size)
    
    # define an optimizer
    optimizer = optim.Adam([p for n, p in model.named_parameters() if n != 'S'] , lr=learning_rate)
    
    return model,optimizer,loss_function

def train(model_type,K,V,Y,dates_split,num_epochs,batch_size,hidden_size,learning_rate,loss_type):
    
    # construct a model
    model,optimizer,loss_function = construct_model(model_type, K, V, Y, dates_split, hidden_size, learning_rate, loss_type)
    
    # train the model
    progress_bar = tqdm(range(num_epochs)); loss_history = []; batched_idx = set()
    
    for epoch in range(num_epochs):
        
        # reset gradients
        optimizer.zero_grad()
        model.S.grad = torch.zeros(model.P, model.J)
        
        # batch data
        batch_pop = set(range(model.J)).difference(batched_idx)
        batch_idx = random.sample(list(batch_pop), batch_size)
        batched_idx = batched_idx.union(batch_idx)
        if len(batched_idx)==model.J:
            batched_idx = set()

        # compute loss
        output = model(batch_idx).squeeze().reshape(-1)
        labels = Y[:,batch_idx].transpose(0, 1).reshape(-1)
        loss = loss_function(output,labels)
        
        # backward propagation and update parameters
        loss.backward()
        with torch.no_grad():
            model.S -= (10**6) * learning_rate * model.S.grad
        optimizer.step()
        
        # save and print the result for this iteration
        progress_bar.update(1)
        loss_history.append(loss.item())
        print(f'Epoch {epoch}, Loss: {round(loss.item(),3)}',end=", ")
        print("S",end=":"); print([round(i,3) for i in model.S[0,:6].tolist()])
    
    # return trained model and history of loss function
    return model,loss_history

def main(params):
    dates_split,K,V,Y = read_data(params.P, params.Q, params.J)
    model,loss_history = train(params.model_type, K, V, Y, dates_split,
                               params.num_epochs, params.batch_size, params.hidden_size,
                               params.learning_rate, params.loss_type)
    result = model.S.detach().numpy()
    
    print('=========')
    print('S')
    for i in range(params.J):
        for j in range(params.P):
            print(result[j,i],end=",")
        print()
    
    print('=========')
    print('loss history')
    for i in range(params.num_epochs):
        print(loss_history[i],end=",")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Embedding Model")

    parser.add_argument("--P", type=int, default=5)
    parser.add_argument("--Q", type=int, default=5)
    parser.add_argument("--J", type=int, default=459)
    parser.add_argument("--model_type", type=str, default="GRU")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--loss_type", type=str, default="L1")

    params, unknown = parser.parse_known_args()
    main(params)