from PIL import Image
import numpy as np
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def predict2(image_path):

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    

    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20
    # percentage of training set to use as validation
    valid_size = 0.2

    # convert data to a normalized torch.FloatTensor
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    print("Reached here!")

    data_dir='./DATASET_test'
    train_data = datasets.ImageFolder(data_dir + '/TRAIN', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/TEST', transform=test_transforms)

    for test in test_data:
        print(test)

    print(len(train_data))
    print(len(test_data))



    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # print("")
    print(num_train)

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)

    print("Train Loader length = "+str(len(train_loader)))

    print("Test Loader length = "+str(len(test_loader)))

    # for data,target in train_loader:
    #     print(data)
    #     print(target)

    classes=['beer-bottle','book', 'can', 'cardboard', 'egg', 'flower', 'food-peels', 'fruit', 'jute', 'leaf', 'meat', 'newspaper', 'paper-plate', 'pizza-box', 'plant', 'plastic-bag', 'plastic-bottle', 'spoilt-food', 'steel-container', 'thermocol']

    # helper function to un-normalize and display an image
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0))) 

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print("Labels = "+str(labels))

    images = images.numpy() # convert images to numpy for display

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])

    batch = next(iter(train_loader))
    print(batch[0].shape)
    plt.imshow(batch[0][0].permute(1, 2, 0))
    print(batch[1][0])

    model = models.densenet121(pretrained=True)
    # print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.densenet121(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(256, len(classes)),
                                    nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    model.to(device)

    # number of epochs to train the model
    n_epochs = 20

    valid_loss_min = np.Inf # track change in validation loss

    # Load pre-saved model for testing
    model.load_state_dict(torch.load('model_final.pt', map_location=torch.device('cpu')))

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(test_transforms(img), 0)

    model.eval()
    out = model(batch_t)


    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:1]]






# def predict(image_path):
#     resnet = models.resnet101(pretrained=True)

#     #https://pytorch.org/docs/stable/torchvision/models.html
#     transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
#     )])

#     img = Image.open(image_path)
#     batch_t = torch.unsqueeze(transform(img), 0)

#     resnet.eval()
#     out = resnet(batch_t)

#     with open('imagenet_classes.txt') as f:
#         classes = [line.strip() for line in f.readlines()]

#     prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
#     _, indices = torch.sort(out, descending=True)
#     return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

