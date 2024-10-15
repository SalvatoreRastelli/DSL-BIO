import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_score, recall_score, f1_score
import os

from config import ConfigG


def train(cfg):
    # Data Transformation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(331),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(331),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Data Loading
    data_dir = cfg.dataset_dir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=cfg.batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {str(device).upper()}")

    # GoogLeNet (Inception v3) model
    model_ft = models.inception_v3(weights='Inception_V3_Weights.DEFAULT')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    os.makedirs(cfg.model_save_dir, exist_ok=True)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model_ft.parameters(), 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay
    )

    best_val_acc = 0.0

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
                    if phase == 'train':
                        outputs, aux_outputs = model_ft(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model_ft(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # print(loss)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(set(all_labels) - set(all_preds))
            epoch_precision = precision_score(all_labels, all_preds, average='weighted')
            epoch_recall = recall_score(all_labels, all_preds, average='weighted')
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

            print('{} Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1))
            
        

    # Save the model if this epoch's validation accuracy is the best so far
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = model_ft.state_dict()
                torch.save(best_model_wts, os.path.join(cfg.model_save_dir, 'best_model.pth'))
                print("Best model updated with val_acc: {:.4f}".format(best_val_acc))

        torch.cuda.empty_cache()

        
    print("\nTraining complete. Evaluating best validation model on test set...")
    # Load best model weights
    model_ft.load_state_dict(torch.load(os.path.join(cfg.model_save_dir, 'best_model.pth')))

    # Evaluate on test set
    model_ft.eval()
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    test_acc = running_corrects.double() / dataset_sizes['test']
    test_precision = precision_score(all_labels, all_preds, average='weighted')
    test_recall = recall_score(all_labels, all_preds, average='weighted')
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    print('Test Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(test_acc, test_precision, test_recall, test_f1))

    # Saving Model
    torch.save(model_ft.state_dict(), f"{cfg.model_save_dir}/GoogLeNet.pth")

if __name__ == '__main__':
    cfg = ConfigG()
    train(cfg)
