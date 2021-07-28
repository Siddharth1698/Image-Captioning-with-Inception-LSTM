#!/usr/bin/env python
# coding: utf-8

# ### Main Ideas for building custom Datasets for Text 
# 
# The idea is to convert the text we have into numerical value.
# 1. We have an index and we need a vocabulory mapping of each word to index.
# 2. We setup a pytorch dataset to load the data.
# 3. The sequence length must be same for all the batches, so we make padding of every batch
# 
# 

# #### Importing all the required libraries.

# In[1]:


import os  
import pandas as pd 
import spacy  # we use spacy for implementation of tokenizer.
import torch
from torch.nn.utils.rnn import pad_sequence  # padding of batch.
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load imgage
import torchvision.transforms as transforms


# In[5]:


spacy_eng = spacy.load('en_core_web_sm') # to know tokenizer it is working with.


# In[6]:


# freq_threshold tells, if the word isn't repeating those frequent amount of time, we can ignore it.
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"} # Padd Token , Start of sentence, End of sentence, Unknown.
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self): # getting length of our vocabulory.
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)] # we get lower case of the tokenizer of text we send.
        # example:>  "Get along soon" -> ["get","along","soon"]

    def build_vocabulary(self, sentence_list): # used to go through each of text and see if its over the threshold and if so we ignore it.
        frequencies = {}
        idx = 4  # we are starting with an index of 4 because we already included first three.

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies: 
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1 

                if frequencies[word] == self.freq_threshold: # here we see if frequency of word is equad to the threshold frequency.
                    self.stoi[word] = idx # So we set the index starting at 4.
                    self.itos[idx] = word # and we set word into that index.
                    idx += 1 # we increment the index.

    def numericalize(self, text): # we take the sentence and convert them to numerical values.
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] # if token are in stoi, then it surparses the frequency threshold.
                                             #Else it wouldnt be in self.toi and we just return the index of unkown token.
            for token in tokenized_text
        ]


# #### Implementation of Dataset class.

# In[7]:


class FlickrDataset(Dataset): # Talking the class dataset.
    
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5): # root directory of image is passed with caption file and a fequency threshold.
        self.root_dir = root_dir # getting the root directory, in our case, we have flickr8k folder.
        self.df = pd.read_csv(captions_file) # reading the captions from caption file.
        self.transform = transform # 

        
        self.imgs = self.df["image"]  # we get the image from image column.
        self.captions = self.df["caption"] # we get the caption assosiated with image from image column.

        
        self.vocab = Vocabulary(freq_threshold) # Initialize vocabulary with respect to threshold we specified.
        self.vocab.build_vocabulary(self.captions.tolist()) # We build the vocabulory here and the captions is passed as a list into the function's parameters.

    def __len__(self): # we get length of dataframe here.
        return len(self.df)

    def __getitem__(self, index): # we use to get a single example with corresponding caption.
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB") # loading of image.

        if self.transform is not None: # in case we have a stransform we can use it.
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]] # string to index -> index of start token.
        numericalized_caption += self.vocab.numericalize(caption) # we numericalize the caption.
        numericalized_caption.append(self.vocab.stoi["<EOS>"]) # append end of sentence.

        return img, torch.tensor(numericalized_caption) #return image by converting numnericalized caption to tensor.


# In[ ]:





# In sequence mode, the sequence length must all be same. But caption length may be different for different examples.
# 
# Check the maximum length of longest sentence and padd to that length , but could be unnecessary computation.
#  

# In[ ]:





# In[8]:


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx # getting the padd index.

    def __call__(self, batch): # we have batch, which is list of all examples we have.
        # unsqueze -> Returns a new tensor with a dimension of size one inserted at the specified position.
        imgs = [item[0].unsqueeze(0) for item in batch] # 1st item returned for each item in batch.
        # torch.cat -> Concatenates the given sequence of seq tensors in the given dimension. 
        imgs = torch.cat(imgs, dim=0) #Concates the images we unsquezed to given dimension. All images must be of same size.
        targets = [item[1] for item in batch] # targets are the captions.
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx) # targets are papdded with pad_sequence function.
        # if batch_first = True,  If True, then the input and output tensors are provided as (batch, seq, feature).

        return imgs, targets # images and targets are returned.


# In[ ]:





# In[9]:


def get_loader(  # loads everything for us.
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


# In[ ]:





# In[10]:



if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    loader, dataset = get_loader(
        "flickr8k/images/", "flickr8k/captions.txt", transform=transform
    )

    for idx, (imgs, captions) in enumerate(loader):
        print("index number: ",idx)
        print("Shape of image is: ",imgs.shape)
        print("numericalized captions: ",captions)
        print("Shape of captions: ",captions.shape)

