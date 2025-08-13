import pandas as pd
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Function for loading the MNIST dataset
def load_datasets():
    print("Downloading the data...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST( root='/data',train=True,transform=transform,download=True )
    test_dataset = datasets.MNIST( root='/data',train=False,transform=transform,download=True )

    print("Length of train and test data")
    print(len(train_dataset),len(test_dataset))

    return train_dataset,test_dataset

# Function for splitting the data into batches
def data_batches(train_dataset,test_dataset):
    train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=False)

    print("Length of train and test loader")
    print(len(train_loader),len(test_loader))

    return train_loader,test_loader

# Function for model training
def train_model(train_loader):
    print("Model Training...")
    import torch.nn as nn
    class Digitclassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64,10)
                )
        
        def forward(self,x):
              return self.network(x)
    
    model = Digitclassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    epochs = 5

    for epoch in range(epochs):
        running_loss = 0
        for images,labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs,labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
    
    return model


# Function for model evaluation
def evaluate_model(model,test_loader):
    print("Evaluation...")
    from sklearn.metrics import accuracy_score
    model.eval()
  
    all_predicted = []
    all_true = []

    with torch.no_grad():
        for images,labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            
            all_predicted.extend(predicted.numpy())
            all_true.extend(labels.numpy())

    print("Accuracy of the model = ",accuracy_score(all_true,all_predicted))   
    return all_predicted,all_true


# Function for classification report/ confusion matrix
def report(all_predicted,all_true):
    from sklearn.metrics import classification_report   
    report = classification_report(all_true,all_predicted)
    print(report)


# Main function for Combining the Logic
def main():
    train_dataset,test_dataset = load_datasets()
    train_loader,test_loader= data_batches(train_dataset,test_dataset)
    model = train_model(train_loader)
    all_predicted,all_true = evaluate_model(model,test_loader)
    report(all_predicted,all_true)
    

if __name__ == '__main__':
    main()


















































