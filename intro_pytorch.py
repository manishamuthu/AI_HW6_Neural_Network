import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    #torchvision.datasets.FashionMNIST()
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set = datasets.FashionMNIST('./data', train = True,download = True, transform = custom_transform)
    test_set = datasets.FashionMNIST('./data', train = False,transform = custom_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)
    if(training):
        return train_loader
    else:
        return test_loader
def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64,10),
    )
    return model

def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0.0
    # running_loss = criterion
    model.train()
    correct=0;
    total=0;
    for epoch in range(T):  # loop over the dataset multiple times
        running_loss=0.0
        total=0
        correct=0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            opt.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            # print statistics
            running_loss += loss.item()
            #if i % 2000 == 1999:  # print every 2000 mini-batches
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss=running_loss/len(data)
            #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        print("Train Epoch: ",epoch," Accuracy: ",correct,"/",total,"(",'{:.2f}'.format(100*correct/total),"%)  Loss: ",'{:.3f}'.format(running_loss) )


def evaluate_model(model, test_loader, criterion, show_loss ):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        running_loss = 0.0
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            loss=criterion(outputs,labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss+=loss.item()
        running_loss=running_loss/len(test_loader)
    if(show_loss==False):
        print("Accuracy: ",'{:.2f}'.format(100*correct/total),"%")
    else:
        print("Average Loss: ",'{:.4f}'.format(running_loss))
        print("Accuracy: ", '{:.2f}'.format(100 * correct / total), "%")


def predict_label(model, test_images, index):
    """
    TODO: implement this function.
    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1

    RETURNS:
        None
    """

    prob = F.softmax(model(test_images), dim=1)
    imageChosen=prob[index]
    max=[]
    indat=[]
    class_names = ["T - shirt / top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "AnkleBoot"]
    for i in range(3):
        currentmax=0;
        currentmaxIndex=0;
        indCnt=0;
        for tensor in imageChosen:
            isAlreadyMax=False;
            for item in max:
                if(tensor==item):
                    isAlreadyMax=True
            if ((isAlreadyMax==False) & (tensor>currentmax)):
                currentmax=tensor
                currentmaxIndex=indCnt
            indCnt+=1
        indat.append(currentmaxIndex)
        max.append(currentmax.item())

    print(class_names[indat[0]],":",'{:.2f}'.format(max[0]*100),"%")
    print(class_names[indat[1]], ":", '{:.2f}'.format(max[1] * 100), "%")
    print(class_names[indat[2]], ":", '{:.2f}'.format(max[2] * 100), "%")


if __name__=="__main__":
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)
    model = build_model()
    print(model)
    train_model(model,train_loader,criterion=nn.CrossEntropyLoss(),T=5)
    evaluate_model(model, test_loader, criterion=nn.CrossEntropyLoss(), show_loss=False)
    evaluate_model(model, test_loader, criterion=nn.CrossEntropyLoss(), show_loss=True)
    pred_set, _= next(iter(test_loader))
    predict_label(model, pred_set, 1)