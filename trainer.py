import torch
from tqdm import tqdm

def evaluate(model, dataloader, criterion, device):
    model.eval()
    
    correct = 0
    total = 0
    val_loss = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in tqdm(enumerate(dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
    
            _, predictions = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            val_loss += loss.item()
            
    val_loss = val_loss / len(dataloader)
    
    print(f"Validate loss: {loss:4f}")
    
    val_acc = correct / total
    
    return val_loss, val_acc


def train_one_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    model = model.train()
    epoch_train_losses = 0
    
    for batch_idx, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        # compute loss
        train_loss = criterion(outputs, labels)
        epoch_train_losses += train_loss.item()
        
        # Backward and optimization
        optimizer.zero_grad()
        train_loss.backward()
        # update weights
        optimizer.step() 

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, loss: {train_loss.item():4f}")
        
    epoch_train_losses = epoch_train_losses / len(dataloader)
        
    return epoch_train_losses



def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, logger=None):
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(epochs)):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        logger.log({"val/loss": train_loss, "val/acc": val_acc})
        logger.log({"train/loss": val_loss})
        
        print(f"Epoch {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}\tVal accuracy: {val_acc:.4f}")
        
    return train_losses, val_losses



            