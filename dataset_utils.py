import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader

import unicodedata
import cv2
import os

def tokenize(line):
    line = line.lower()
    tokens = line.split()
    return tokens


def resize_with_aspect(np_image, target_height):
    if np_image is None:
        return None

    aspect_ratio = target_height / np_image.shape[0]
    target_width = int(np_image.shape[1] * aspect_ratio)
    dim = (target_width, target_height)
    image = cv2.resize(np_image, dsize=dim, interpolation=cv2.INTER_AREA)
    return image


def pad_to_width(np_image, max_width):
    orig_width = np_image.shape[1]
    delta_w = max_width - orig_width
    if delta_w > 0:
        return cv2.copyMakeBorder(
            src=np_image, top=0, bottom=0, left=0,
            right=delta_w,  # Padding only from the right side
            borderType=cv2.BORDER_CONSTANT,
            value=0  # grayscale black padding
        )
    return np_image


class CyrillicHandwritingDataset(Dataset):
    def __init__(self, dataframe, images_dir, image_height=64, transform=None, char2idx=None):
        self.data = dataframe
        self.images_dir = images_dir
        self.image_height = image_height
        self.transform = transform  # convert an image to a tensor
        self.char2idx = char2idx
        self.labels = [
            ' '.join(tokens) if isinstance(tokens, list) else tokens
            for tokens in dataframe['tokens']
        ]

        # Load and resize images
        self.images = self._load_and_resize_images()

        # Get max width for padding
        self.max_width = max(img.shape[1] for img in self.images)

        # Pad all resized images
        self.images = [
            pad_to_width(image, self.max_width) for image in self.images
        ]

    def _load_and_resize_images(self):
        """
        Loading and resizing with aspect
        """
        resized_images = []
        for index, row in self.data.iterrows():
            filename = row["filename"]
            image_path = os.path.abspath(os.path.join(self.images_dir, filename))
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Resize this image
            resized_image = resize_with_aspect(image, self.image_height)
            resized_images.append(resized_image)

        return resized_images

    def text_to_indices(self, text):
        text = unicodedata.normalize('NFC', text).lower()
        indices = []
        for ch in text:
            if ch in self.char2idx and ch != '<BLANK>':
                indices.append(self.char2idx[ch])
            else:
                indices.append(self.char2idx['<UNK>'])
        return indices

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image_tensor = self.transform(image)
        label_indices = self.text_to_indices(label)
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        return image_tensor, label_tensor, label


def cyrillic_collate_fn(batch):
    """
    batch: (image_tensor, label_tensor, label_str)
    """

    batch.sort(key=lambda x: x[0].shape[-1], reverse=True)

    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    label_strs = [item[2] for item in batch]
    images_padded = torch.stack(images, dim=0)
    # All labels concatenated
    labels_concat = torch.cat(labels)

    # Effective sequence lengths after CNN
    input_lengths = torch.full(
        size=(len(images),),
        fill_value=images_padded.shape[-1] // 4,
        dtype=torch.long
    )
    # Label lenght (number of characters)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    return images_padded, labels_concat, input_lengths, target_lengths, label_strs


# ==========================
# Greedy Decoding for CTC
# ==========================
def greedy_decode(output, idx2char):
    # output of a model: (T, B, C) - time_steps, batch_size, num_classes
    # get the index of the most probably class (hence dim=2, C)
    out = output.argmax(dim=2).permute(1, 0)  # [B, T]
    decoded_batch = []
    for seq in out:
        decoded = []
        prev = None
        for idx in seq:
            char = idx2char[idx.item()]
            if char != '<BLANK>' and char != prev:
                decoded.append(char)
            prev = char
        decoded_batch.append(''.join(decoded))
    return decoded_batch


# Main
# Loading the tsv
train_df = pd.read_csv("./archive/train.tsv", sep="\t", header=None)
test_df = pd.read_csv("./archive/test.tsv", sep="\t", header=None)

train_df.columns = ["filename", "line"]
test_df.columns = ["filename", "line"]

# Get rid of rows with NaN values
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Get rid of bad rows containing latin or non-cyrillic chars
train_df.drop(index=37735, inplace=True)
train_df.drop(index=47799, inplace=True)
train_df.drop(index=65611, inplace=True)

# Reset index, to generate new indexing
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Add a new column to the dataframes
train_df["tokens"] = train_df["line"].apply(tokenize)
test_df["tokens"] = test_df["line"].apply(tokenize)

transform_pipeline = transforms.Compose([
    transforms.ToTensor()
])

