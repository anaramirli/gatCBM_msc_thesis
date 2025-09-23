import os, random
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch


class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        label = self.data.iloc[idx, 1]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def _set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _tensors_from_loader(dl: DataLoader):
    xs, ys = [], []
    for x, y in dl:
        xs.append(x)
        ys.append(y)
    if xs:
        X = torch.cat(xs, dim=0)
        Y = torch.cat(ys, dim=0)
    else:
        X = torch.empty(0, 3, 224, 224)
        Y = torch.empty(0, dtype=torch.long)
    return X, Y

model_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pre-trained model normalization
])

def _safe_argmax(patches_U, ignore_list):
    sorted_indices = np.argsort(-patches_U, axis=1)  # sort in descending order
    patches_C = np.zeros(patches_U.shape[0], dtype=int)

    for i in range(patches_U.shape[0]):
        for idx in sorted_indices[i]:
            if idx not in ignore_list:
                patches_C[i] = idx
                break  # pick the first valid index
    return patches_C


###### for conept saving
def _reverse_preprocess(images, mean, std):
    """
    Reverses the preprocessing done by tf.keras.applications.vgg19.preprocess_input.
    
    Args:
        images (np.array): The preprocessed images.
        
    Returns:
        np.array: The images after reversing the preprocessing.
    """

    mean = np.array(mean)
    std = np.array(std)

    # Reverse normalization
    unnormalized_images = (images * std[None, None, :]) + mean[None, None, :]
    unnormalized_images = np.clip(unnormalized_images, 0, 1)
        
    return unnormalized_images  

def _save_concepts(crops, crops_u, reverse=True, start=0, nb_crops=20, save=False, save_dir=None):

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    for c_id in range(crops_u.shape[1]):
        # Convert crops_u to a NumPy array if it's a TensorFlow tensor
        crops_u_np = crops_u.numpy() if hasattr(crops_u, 'numpy') else crops_u
        
        # Sort crop scores in descending order and get their indices
        sorted_indices = np.argsort(crops_u_np[:, c_id])[::-1]
        

        best_crops = np.array(crops)[sorted_indices[start:start+nb_crops]]


        # Adjust the figure size based on the number of crops to display
        # E.g., increase the size if there are more crops to show
        rows = ceil(nb_crops / 10)  # Number of rows needed
        plt.figure(figsize=(30, 4 * rows))  # Adjust width and height here

        # Show the selected crops
        for i in range(nb_crops):
            plt.subplot(rows, 10, i + 1)
            img = np.array(best_crops[i])
            if reverse==True:
                img = _reverse_preprocess(img, mean, std)
            plt.imshow(img)
            plt.axis('off')
        
        
        filename = f"concept_{c_id}.png"
        if save==True and save_dir!=None:
            full_path = os.path.join(save_dir, filename)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
        # plt.show()  # Display the figure
        # plt.close()  # Close the figure to free memory
        print('\n\n')
