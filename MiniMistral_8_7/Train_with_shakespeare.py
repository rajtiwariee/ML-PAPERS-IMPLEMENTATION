  
#------------------------ DataLoader Lite --------------------------------
import tiktoken
from torch.utils.data import Dataset, DataLoader
import os
import torch
from model import minimixtral, ModelArgs,MoeArgs
import time,tqdm

class DataLoaderLite(Dataset):
    
    def __init__(self, B, T):
        super().__init__()
        self.B = B
        self.T = T
        
        #load the data
        with open('./input.txt', 'r') as f:
            text = f.read()
        #encode the text
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)// (B*T)} batches")
        
        #state
        self.current_position = 0
        
    def __len__(self):
        return len(self.tokens) // (self.B * self.T)
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = buf[:-1].view(B,T) # (4, 256) = 4*256 = 1024
        y = buf[1:].view(B,T) #(4,32)
        self.current_position += B*T
        
        #if loading the next batch goes out of the range of the tokens reset
        if self.current_position + (B*T +1) > len(self.tokens):
            self.current_position = 0
        return x, y

    def __getitem__(self, index):
        #get the next batch
        x, y = self.next_batch()
        return x, y
    


    
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
    
print(f"Using {device} device")

B = 4
T = 256
train_loader = DataLoaderLite(B=B, T=T)

args = ModelArgs(vocab_size=50257,moe=MoeArgs())
model = minimixtral(args)

model.train()
model.to(device)
steps = 50
optimizer = torch.optim.AdamW(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()
# model = torch.compile(model) #faster training of the model

for step in tqdm(range(steps)):
    t0 = time.time()
    optimizer.zero_grad() #reset the gradients


    x,y = train_loader.next_batch() #get the next batch
    x,y = x.to(device), y.to(device) #move the batch to GPU
    #forward the model
    # with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        #forward the model
    logits= model(x,start_pos = 0) #(4,256,50257)
    logits_flat = logits.view(-1, logits.size(-1))
    y_flat = y.view(-1)
    loss = loss_fn(logits_flat, y_flat) #(1024,50257) , (1024)
    #send loss backward
    loss.backward() #calculate the gradients
    #this prevents the model from getting bigger shock of gradients (which means higher loss it could be because of the any unlucky batch of the data)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #clip the gradients
    #determine and set the learning rate for this iteration
    optimizer.step() #update the weights
    
    t1 = time.time()
    dt = (t1-t0) * 1000 #convert to ms
    tokens_per_second = B * T / (dt/1000)
    print(f"iter: {step} | loss: {loss.item():.6f} | norm: {norm:.4f}  | dt: {dt:.2f} ms | tokens/sec: {tokens_per_second:.2f}")



    
