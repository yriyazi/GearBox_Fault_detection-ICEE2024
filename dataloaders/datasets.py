
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.utils.data import random_split
import tqdm
import utils
import os
import platform
import multiprocessing

def get_system_info():
    system = platform.system()
    if system == "Windows" or system == "Darwin":  # Windows or macOS
        return 0
    elif system == "Linux" and os.path.exists("/proc/cpuinfo"):  # Linux
        return multiprocessing.cpu_count()
    else:
        return "Unsupported operating system"

data_root       = utils.dataset_path[2:-1]
_p_train        = utils.train_split
_p_validation   = utils.validation_split
_p_test         = 1 - _p_train - _p_validation
batch_size      = utils.batch_size                  # set batch size
num_workers     = get_system_info()                 # set number of workers

# Define the data transformations
transform = transforms.Compose([transforms.Resize((utils.input_image_size, utils.input_image_size)),
                                transforms.ToTensor()])        #convert to tensor

# Use ImageFolder to load the dataset
dataset = ImageFolder(root=data_root, transform=transform)

# Obtain the total length of the dataset
dataset_size = len(dataset)

# Specify the desired ratios for training, validation, and test sets
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate the number of samples for each set
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size

# Use stratified sampling to obtain indices for each set
class_counts = {}  # Dictionary to store the count of samples for each class
for idx in range(dataset_size):
    sample, label = dataset[idx]  # Assuming your dataset provides samples and labels
    if label not in class_counts:
        class_counts[label] = 0
    class_counts[label] += 1

class_train_size = {label: int(train_ratio * count) for label, count in class_counts.items()}
class_val_size = {label: int(val_ratio * count) for label, count in class_counts.items()}

train_indices, val_indices, test_indices = [], [], []
class_counts_current = {label: 0 for label in class_counts}

for idx in tqdm.tqdm(range(dataset_size)):
    sample, label = dataset[idx]
    if class_counts_current[label] < class_train_size[label]:
        train_indices.append(idx)
        class_counts_current[label] += 1
    elif class_counts_current[label] < class_train_size[label] + class_val_size[label]:
        val_indices.append(idx)
        class_counts_current[label] += 1
    else:
        test_indices.append(idx)

# Use the obtained indices to create DataLoader for each set
train_set = torch.utils.data.Subset(dataset, train_indices)
val_set = torch.utils.data.Subset(dataset, val_indices)
test_set = torch.utils.data.Subset(dataset, test_indices)


# Use random_split to create train, validation, and test datasets
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
