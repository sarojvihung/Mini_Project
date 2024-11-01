# train.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders
from model import initialize_model
from utils import save_model

def train_model(data_dir, num_epochs=50, learning_rate=0.0001, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Initialize the FaceNet-based model
    model = initialize_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loader, val_loader = get_dataloaders(root_dir=data_dir, batch_size=batch_size)
    #print(train_loader)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        #torch.load("anti_spoofing_model.pth")  
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
        print('Valiation', end=' ')
        validate_model(model, val_loader, device, criterion)
        print('Train', end=' ')
        validate_model(model, train_loader, device, criterion)
    
    save_model(model, 'anti_spoofing_model1.pth')

def validate_model(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    data_dir = 'real_and_fake_face'  # Replace with your dataset path
    train_model(data_dir)

