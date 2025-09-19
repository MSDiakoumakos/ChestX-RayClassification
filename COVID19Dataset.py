# 1. Συνδέουμε το Google Drive και προετοιμάζουμε το dataset
from google.colab import drive
drive.mount('/content/drive')

!cp '/content/drive/MyDrive/datasets/COVID-19_Radiography_Dataset.zip' .
!unzip -q -n COVID-19_Radiography_Dataset.zip
data_dir = '/content/COVID-19_Radiography_Dataset'

# 2. Εισαγωγή των απαραίτητων βιβλιοθηκών
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from torchvision import transforms
import torch.optim as optim
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

###################
#### Ερ΄ώτημα 2 ####
################### 
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Ορισμός της κλάσης COVID19Dataset
class COVID19Dataset(Dataset):
    def __init__(self, root_dir, transform=None, load_fraction=0.3):
        self.transform = transform
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        
        # Ορισμός των κλάσεων και αντιστοίχιση με αριθμούς
        self.classes = ['Normal', 'COVID', 'Viral Pneumonia', 'Lung_Opacity']
        self.class_to_idx = {
            'Normal': 0,
            'COVID': 1,
            'Viral Pneumonia': 2,
            'Lung_Opacity': 3
        }
        
        # Συλλογή όλων των εικόνων και των ετικετών τους
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            # Έλεγχος αν είναι directory
            if os.path.isdir(class_path):
                # Get all images for this class
                class_images = []
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    # Έλεγχος αν το αρχείο είναι εικόνα
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        class_images.append(img_path)
                
                # Calculate how many images to take from this class
                num_images = max(1, int(len(class_images) * load_fraction))
                selected_images = class_images[:num_images]
                
                # Προσθήκη των επιλεγμένων εικόνων και των ετικετών τους
                self.images.extend(selected_images)
                self.labels.extend([self.class_to_idx[class_name]] * len(selected_images))
                
                print(f"Loaded {len(selected_images)} images from {class_name}")
        
        print(f"Total loaded images: {len(self.images)} ({load_fraction*100}% of dataset)")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def display_batch(self, indexes):
        n = len(indexes)
        grid_size = int(np.ceil(np.sqrt(n)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle('COVID-19 Dataset Images', fontsize=16)
        
        for i, idx in enumerate(indexes):
            row = i // grid_size
            col = i % grid_size
            
            img, label = self[idx]
            if isinstance(img, torch.Tensor):
                img = img.squeeze().numpy()  # Convert to numpy
        
            # Display the image
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f'Class: {self.classes[label]}')
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(n, grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_class_distribution(self):
        class_counts = {}
        for label in self.labels:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        plt.figure(figsize=(10, 6))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title('Distribution of Classes in COVID-19 Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

###################
#### Ερ΄ώτημα 3 ####
################### 
## Function to calculate the confusion matrix
# Input: y - true labels, y_pred - predicted labels     
# Output: confusion matrix
def confusion_matrix(y, y_pred):
    return sk_confusion_matrix(y.cpu().numpy(), y_pred.cpu().numpy(), 
                             labels=range(4))  # Explicitly specify 4 classes


## Function to train the model for one epoch
# Input: DataLoader - data loader, optimizer - optimizer, loss function - loss function, device - device
# Output: loss - loss value
def train_one_epoch(model, data_loader, optimizer, loss_function, device):
    model.train()  # Θέτουμε το μοντέλο σε κατάσταση εκπαίδευσης
    total_loss = 0
    all_labels = []
    all_preds = []

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Μηδενίζουμε τους βαθμούς
        outputs = model(images)  # Προβλέψεις του μοντέλου
        loss = loss_function(outputs, labels)  # Υπολογισμός της απώλειας
        loss.backward()  # Υπολογισμός των παραγώγων
        optimizer.step()  # Ενημέρωση των βαρών

        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = (np.array(all_labels) == np.array(all_preds)).mean()  # Calculate accuracy
    return avg_loss, accuracy

## Function to test the model
# Input: model - model, DataLoader - data loader, device - device
# Output: accuracy - accuracy value
def test(model, data_loader, loss_function, device):
    model.eval()  # Θέτουμε το μοντέλο σε κατάσταση δοκιμής
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Απενεργοποιούμε την παρακολούθηση των gradients
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # Προβλέψεις του μοντέλου
            loss = loss_function(outputs, labels)  # Υπολογισμός της απώλειας

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = (np.array(all_labels) == np.array(all_preds)).mean()
    conf_matrix = confusion_matrix(torch.tensor(all_labels), torch.tensor(all_preds))

    return avg_loss, accuracy, conf_matrix


###################
#### Ερ΄ώτημα 4 ####
################### 

# Ορισμός της κλάσης CNN1
class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)  # 8 filters, 3x3
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with stride 2
        self.conv2 = nn.Conv2d(8, 16, 3)  # 16 filters, 3x3
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with stride 2
        self.fc1 = nn.Linear(85264, 32)  # Fully connected layer with 32 neurons
        #self.fc1 = nn.Linear(576, 32)  # Fully connected layer with 32 neurons
        self.fc2 = nn.Linear(32, 4)  # Output layer with 4 outputs

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Convolution + ReLU + Pooling
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Fully connected layer + ReLU
        x = self.fc2(x)  # Output layer
        return x

###################
#### Ερ΄ώτημα 5 ####
################### 

#CNN2 Padded
class CNN2(nn.Module):
    def __init__(self, num_classes):
        super(CNN2, self).__init__()
        # First block - add padding=1 to maintain dimensions
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4) 
        
        # Second block
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third block
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fourth block
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Fifth block
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.num_features = 8192
        self.fc1 = nn.Linear(self.num_features, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def _forward_features(self, x):
        # First block
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        # Second block
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        
        # Third block
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        
        # Fourth block
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)
        
        # Fifth block
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

###################
#### Ερ΄ώτημα 6 ####
################### 

class ResNet50(nn.Module):
    def __init__(self, num_classes=4, feature_extracting=True):
        super(ResNet50, self).__init__()
        # Load pre-trained ResNet50
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if feature_extracting:
            # Freeze all layers except first conv and final fc
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze first conv layer since we modified it
            for param in self.model.conv1.parameters():
                param.requires_grad = True
            
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=1)

# Helper function 
def create_resnet50(num_classes=4):
    return ResNet50(num_classes)

###################
#### Ερ΄ώτημα 7 ####
################### 

class BasicBlock(nn.Module):
    def __init__(self, n_in, n_filters, stride=1):
        super(BasicBlock, self).__init__()
        
        # 1. First convolutional layer (3x3, no padding, stride 1 or 2)
        self.conv1 = nn.Conv2d(n_in, n_filters, kernel_size=3, 
                              stride=stride, padding=0)
        
        # 2. First batch normalization
        self.bn1 = nn.BatchNorm2d(n_filters)
        
        # 3. ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # 4. Second convolutional layer (3x3, with padding to maintain dimensions)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3,
                               stride=1, padding=2)  # Add padding=1
        
        # 5. Second batch normalization
        self.bn2 = nn.BatchNorm2d(n_filters)
        
        # Optional path for stride=2 case or when input and output dimensions differ
        self.downsample = None
        if stride != 1 or n_in != n_filters:
            self.downsample = nn.Sequential(
                # 1x1 convolution to match dimensions
                nn.Conv2d(n_in, n_filters, kernel_size=1, 
                          stride=stride, padding=0, bias=False),  
                nn.BatchNorm2d(n_filters)
            )
            
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)        # First conv
        out = self.bn1(out)        # First batch norm
        out = self.relu(out)       # First ReLU
        
        out = self.conv2(out)      # Second conv
        out = self.bn2(out)        # Second batch norm
        
        # Handle stride=2 case with 1x1 convolution
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # Add skip connection and final ReLU
        out += identity            # Add skip connection
        out = self.relu(out)       # Final ReLU
        
        return out
    
class CustomResNet(nn.Module):
    def __init__(self, num_classes=4, layers=[2, 2, 2, 2], base_channels=64, dropout_rate=0.2):
        super(CustomResNet, self).__init__()
            
        # Initial convolution with larger kernel for medical images
        self.conv1 = nn.Conv2d(1, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
        # Residual layers with increasing channels
        self.layer1 = self._make_layer(base_channels, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels*2, layers[1], stride=1)
        self.layer3 = self._make_layer(base_channels*2, base_channels*4, layers[2], stride=1)
        self.layer4 = self._make_layer(base_channels*4, base_channels*8, layers[3], stride=1)
            
        # Global average pooling and classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(base_channels*8, num_classes)
            
        # Initialize weights
        self._initialize_weights()
            
    def _make_layer(self, n_in, n_filters, blocks, stride):
        layers = []
        layers.append(BasicBlock(n_in, n_filters, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(n_filters, n_filters, stride=1))
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
            
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
            
        return x

# Function to create different model configurations
def create_custom_resnet(model_size='medium'):
    configs = {
        'small': {
            'layers': [1, 1, 1, 1],
            'base_channels': 32,
            'dropout_rate': 0.1
        }
    }
    return CustomResNet(**configs[model_size])

################
#### ΜΑΙΝ #####
################
if __name__ == "__main__":
    # 4. Δημιουργία και χρήση του dataset
    # Define transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Single channel normalization
    ])

    # Abjasast the percentage of the dataset to load
    dataset = COVID19Dataset(data_dir, transform=transform, load_fraction=1.0)

    # Split the dataset
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 5. Δημιουργία 25 τυχαίων indexes
    random_indexes = np.random.choice(len(dataset), size=25, replace=False)

    # 6. Απεικόνιση των εικόνων
    dataset.display_batch(random_indexes)

    # 7. Απεικόνιση της κατανομής των κλάσεων
    dataset.plot_class_distribution()

    # Initialize the model, loss function, and optimizer
    model = CNN1().to(device)
    #model = CNN2(num_classes=4).to(device)
    #model = ResNet50(num_classes=4, feature_extracting=False).to(device) 

    #Initialize the model for the bonus requirement
    #model = create_custom_resnet().to(device)
    
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
    optimizer = optim.Adam(model.parameters(), 
                          #lr=1e-2,      # Learning rate 10^-2  for expiermenting with CNN2
                          lr=1e-3,      # Learning rate 10^-3
                          #lr=1e-4,      # Learning rate 10^-4 for ResNet50 & experiment with CNN2
                          betas=(0.9, 0.99),  # β1 = 0.9, β2 = 0.99
                          )
    
    #Schedulers to experiment with
    # Option 1: StepLR
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Option 2: ReduceLROnPlateau
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Training loop with early stopping
    divergence_threshold = 0.5  # Maximum allowed difference between train and val loss
    divergence_patience = 5     # Number of epochs to wait for divergence
    divergence_counter = 0      # Counter for consecutive divergent epochs

    #for epoch in range(5):
    for epoch in range(20):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy, _ = test(model, val_loader, criterion, device)

        #print(f'Epoch {epoch+1}/{5}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print(f'Epoch {epoch+1}/{20}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Check for divergence between train and validation loss
        loss_difference = abs(train_loss - val_loss)
        if loss_difference > divergence_threshold:
            divergence_counter += 1
            if divergence_counter >= divergence_patience:
                print(f"Early stopping triggered: Train-Val loss difference ({loss_difference:.4f}) exceeded threshold for {divergence_patience} epochs.")
                break
        else:
            divergence_counter = 0  # Reset counter if difference is within threshold

    # After the training loop, get final accuracies
    final_train_loss, final_train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
    final_val_loss, final_val_accuracy, _ = test(model, val_loader, criterion, device)
    final_test_loss, final_test_accuracy, conf_matrix = test(model, test_loader, criterion, device)

    print("\nFinal Results:")
    print(f'Training Accuracy: {final_train_accuracy:.4f}')
    print(f'Validation Accuracy: {final_val_accuracy:.4f}')
    print(f'Test Accuracy: {final_test_accuracy:.4f}')
    print("\nConfusion Matrix:\n", conf_matrix)