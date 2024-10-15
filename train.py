import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import pandas as pd

from config import Config

def train(cfg):

     # Data Transformation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Data Loading
    data_dir = cfg.dataset_dir
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        # 'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val_test']),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['val_test']),
    }
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True, num_workers=4),
        # 'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=False, num_workers=4),
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #"cuda" if torch.cuda.is_available() else 
    print(f"\nUsing device: {str(device).upper()}")

    # AlexNet model
    model_ft = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    os.makedirs(cfg.model_save_dir, exist_ok=True)

    # Get training stuff
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model_ft.parameters(), 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay
    )

    for epoch in range(cfg.num_epochs):
        print('Epoch {}/{}'.format(epoch, cfg.num_epochs - 1))
        print('-' * 10)

        for phase in ['train']:
            if phase == 'train':
                model_ft.train()
            else:
                model_ft.eval()

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_precision = precision_score(all_labels, all_preds, average='weighted')
            epoch_recall = recall_score(all_labels, all_preds, average='weighted')
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

            print('{} Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1))
    
        torch.cuda.empty_cache()

    print("\nTraining complete. Evaluating best validation model on test set...")
    # Evaluate on test set
    model_ft.eval()
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []
    all_outputs = []

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

        all_outputs.extend(outputs.cpu())
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())


    test_acc = running_corrects.double() / dataset_sizes['test']
    test_precision = precision_score(all_labels, all_preds, average='weighted')
    test_recall = recall_score(all_labels, all_preds, average='weighted')
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    print('Test Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format( test_acc, test_precision, test_recall, test_f1))


    pred = pd.DataFrame(all_labels , all_preds)
    pred.to_csv("AlexNet.csv")

    # Saving Model
    torch.save(model_ft.state_dict(), f"{cfg.model_save_dir}/AlexNet.pth")

if __name__ == '__main__':
    cfg = Config()
    train(cfg)
