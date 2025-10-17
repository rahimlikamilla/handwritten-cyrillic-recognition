# Build a vocab
import unicodedata
from matplotlib import pyplot as plt

from dataset_utils import train_df

# For testing purposes:
def depict(image, label_tensor, label_str):
    image_copy = image.squeeze(0)
    plt.imshow(image_copy, cmap="gray")
    plt.title(label_str)


def build_char_vocab(df, text_col='line', lowercase=True, add_blank=True, add_unseen=True):
    chars = set()
    for index, text in df[text_col].items():
        if not isinstance(text, str):
            continue

        # normalize unicode (important for cyrillic)
        # bc these chars have multiple valid ways to encode a single visual
        # character, 'NFC' - Normalization Form C
        text = unicodedata.normalize('NFC', text)
        text = text.lower() if lowercase else text

        # update the character set (including punctuation and spaces)
        chars.update(list(text))

    sorted_chars = sorted(chars)

    # Build mappings
    char2idx = {}
    idx = 0

    # Start by adding special symbols
    if add_blank:
        char2idx['<BLANK>'] = idx
        idx += 1

    for ch in sorted_chars:
        char2idx[ch] = idx
        idx += 1

    if add_unseen:
        char2idx['<UNK>'] = idx

    # Build reverse mapping
    idx2char = {val: key for key, val in char2idx.items()}

    print(f"Vocabulary successfully built: {len(char2idx)} total characters")

    return char2idx, idx2char


def text_to_indices(text, char2idx):
    text = unicodedata.normalize('NFC', text).lower()
    indices = []
    for ch in text:
        if ch in char2idx and ch != '<BLANK>':
            indices.append(char2idx[ch])
        else:
            indices.append(char2idx['<UNK>'])
    return indices


def indices_to_text(indices, idx2char):
    text = ""
    for i in indices:
        text += idx2char[i] if idx2char[i] != '<BLANK>' else ''
    return text


# The <BLANK> is reserved for CTC, <UKN> for unseen chars

char2idx, idx2char = build_char_vocab(df=train_df, text_col='line')
