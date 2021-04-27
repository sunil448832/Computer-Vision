from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    pool = nn.MaxPool2d(kernel_size=2, stride=2) 

    #Encoder
    self.encoder=nn.Sequential(nn.Conv2d(3, 8, kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True),pool,
                               nn.Conv2d(8, 16, kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True),pool,
                               nn.Conv2d(16, 32, kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True),pool,
                               )
    
    #Decoder
    self.decoder=nn.Sequential(nn.ConvTranspose2d(32, 16,kernel_size=2, stride=2),nn.ReLU(inplace=True),
                               nn.ConvTranspose2d(16, 8,kernel_size=2, stride=2),nn.ReLU(inplace=True),
                               nn.ConvTranspose2d(8, 3,kernel_size=2, stride=2),nn.Sigmoid()
                               )
    self.criterion=nn.MSELoss()

  def forward(self, x):
    x=self.encoder(x)
    x=self.decoder(x)   
    return x
    
  # Auto Encoder loss function  
  def loss(self,x,y):
    x=x.view(x.size(0),-1).float()
    y=y.view(y.size(0),-1).float()
    return self.criterion(x,y)


def train(model,train_data,optimizer,batch_size):
    
  epoch_loss = 0
  len=0
  model.train()
  for input,target in tqdm(DataLoader(train_data,batch_size=batch_size)):
    optimizer.zero_grad()
    input,target= input.to(device),target.to(device)
    predictions = model(input)
    loss = model.loss(predictions,target)

    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    len+=1
      
  return epoch_loss / len

# function to evaluate the model on validation dataset returns loss and accuracy
def eval(model,eval_data,batch_size):
    
  epoch_loss = 0
  len=0
  model.eval()
  with torch.no_grad():
    for input,target in tqdm(DataLoader(eval_data,batch_size=batch_size)):
      input,target= input.to(device),target.to(device)
      predictions = model(input)
      loss = model.loss(predictions,target)
      epoch_loss += loss.item()
      len+=1 
      
  return epoch_loss / len

def fit(model,train_data,val_data,epoch,lr=.001):
  optimizer = optim.Adam(model.parameters(),lr=.01)
  for epoch in range(epoch):
    train_loss=train(model,train_data,optimizer,64)
    val_loss=eval(model,val_data,64)
    print("epoch: %d  train_loss: %.4f  val_loss: %.4f"%(epoch,train_loss,val_loss))
