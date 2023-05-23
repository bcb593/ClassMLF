#!/usr/bin/env python
# coding: utf-8

# In[4]:


import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('default')

from transformers import logging
logging.set_verbosity_error()


# In[5]:


import torch.nn as nn
import torchvision
import torchvision.models as cv_models
from torchvision import transforms
import torch
import os
from transformers import BertConfig, BertModel, RobertaForMaskedLM, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig
import math
from transformers import BertConfig, BertModel
import matplotlib.pyplot as plt
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
import pandas as pd
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
from PIL import Image
import re
import glob
from torch.utils.data import TensorDataset,Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from textaugment import EDA
from PIL import Image, ImageEnhance, ImageOps
import random
import nltk
from tqdm import tqdm
nltk.download('stopwords')


# In[6]:


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)


# In[7]:


def text_rep(text):
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',return_tensors='pt')
    model=BertModel.from_pretrained('bert-base-uncased')

    #padded_tokens=tokens +['[PAD]' for _ in range(T-len(tokens))]
    #attn_mask=[ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]
    sent_ids=tokenizer.convert_tokens_to_ids(tokens)
    token_ids = torch.tensor(sent_ids).unsqueeze(0) 
    #attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
    hidden_reps, cls_head = model(token_ids,return_dict=False)
    return hidden_reps


# In[8]:


def textencoder(tweets):
    max_seq_length=32
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',return_tensors='pt')
   
    tokenized_tweets=[tokenizer.tokenize(sent) for sent in tweets]
    tokenized_inputs=[tokenizer.convert_tokens_to_ids(x) for x in tokenized_tweets]
    inputs = pad_sequences(tokenized_inputs, maxlen=max_seq_length, dtype="long", truncating="post", padding="post")
    masks = []
    for seq in inputs:
        seq_mask = [float(i>0) for i in seq]
        masks.append(seq_mask)
    tensor_inputs=torch.tensor(inputs)
    tensor_masks=torch.tensor(masks)
    model=BertModel.from_pretrained('bert-base-uncased')
    hidden_reps, cls_head = model(tensor_inputs,return_dict=False)
    return hidden_reps


# In[9]:


def images_rep(image):
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])
    image_tensor = preprocess(image).unsqueeze(0)
    resnet = cv_models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-2]
    resnet = torch.nn.Sequential(*modules)
    resnet.eval()
    # Compute hidden representation
    with torch.no_grad():
        hidden = resnet(image_tensor)
    image_rep=hidden.permute(2, 3, 1, 0).contiguous().view(7, 7, 2048)
    image_rep=image_rep.view(49,2048)
    return image_rep


# In[10]:


def unnormalize(image):
    '''
    The unnormalize function retains the original image from the normalized image. This is essential because the CLIP 
    requires the original PIL image. 
    '''
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    unnormalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std])
    unnormalized_image = unnormalize(image)
    return unnormalized_image

def clip(texts,images):
    unnormalized_images=[unnormalize(img) for img in images] # Getting back to original image from the normalized
    transform = T.ToPILImage()
    images = [transform(image) for image in unnormalized_images]
    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    #Text
    inputs = [tokenizer(text, return_tensors="pt") for text in texts]
    text_emb = [model.get_text_features(**input1) for input1 in inputs]
    #image
    images = [processor(text=None,images=image,return_tensors='pt')['pixel_values'] for image in images]
    img_emb = [model.get_image_features(image) for image in images]
    img_emb=torch.stack((img_emb)).transpose(1, 2)
    text_emb=torch.stack((text_emb)).transpose(1, 2)
    total=torch.concat((text_emb,img_emb),dim=1)
    tensor = torch.transpose(total, 1, 2)
    return text_emb,img_emb
    
    


# In[11]:


import torch
import torch.nn as nn

import torch.nn as nn
import torch

class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()
        
        self.query0_text = nn.Linear(768, 512, bias=False)
        self.value0_text = nn.Linear(768, 512, bias=False)
        
        
        self.text1=nn.Linear(512,256,bias=False)
        self.text2=nn.Linear(256,64,bias=False)
        self.text3=nn.Linear(64,3)
        self.relu = nn.ReLU()

    def forward(self, text_rep,clip_image):
       # clip_image = torch.stack((clip_image)).transpose(1, 2)
        
        Q_T = self.query0_text(text_rep)
        V_T = self.value0_text(text_rep)
        a_T = torch.bmm(Q_T, clip_image)
        a_Tt = torch.exp(a_T)
        
        x_prime_t=torch.mul(a_T,V_T)
        
        x_prime_t=self.text1(x_prime_t)
        x_prime_t=self.relu(x_prime_t)
        x_prime_t=self.text2(x_prime_t)
        x_prime_t=self.relu(x_prime_t)
        x_prime_t=self.text3(x_prime_t)
        
        return x_prime_t


# In[12]:


text_data=pd.read_excel('../Twitter1517/Data/Twitter1517_texts and labels.xlsx')


# In[13]:


text_data['multi_label']=text_data['multi_label'].apply(lambda x: 2 if x == -1 else x)


# In[14]:


image_dir='../Twitter1517/Data/image/'


# In[15]:


class MyMultimodalDataset(Dataset):
    def __init__(self, text_data, image_folder_path, transform=None):
        self.text_data = text_data
        self.image_folder_path = image_folder_path
        self.transform = transform

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder_path, pad_with_zeros(index+1,4) + '.jpg')
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        text = self.text_data.iloc[index]['txt']
        label = self.text_data.iloc[index]['multi_label']
        '''
        print('Text:', text)
        print('Image:', image_path)
        print('Label',label)
        '''
        return image, text,label


# In[16]:


def pad_with_zeros(n, num_digits):
    n_str = str(n)
    num_zeros_needed = num_digits - len(n_str)
    padded_n_str = '0' * num_zeros_needed + n_str
    
    return padded_n_str


# In[17]:


# Create datasets and dataloaders
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


# In[18]:


# Create datasets and dataloaders
dataset = MyMultimodalDataset(text_data, image_dir, transform=transform)


# In[19]:


from torch.utils.data import random_split

# Assuming you already have a dataset called 'my_dataset'
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8) # 80% for training, 20% for testing
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# In[20]:


# Create the data loaders
batch_size=32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# In[21]:


model=CrossAttention()
model.to(device)


# In[22]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)


# In[23]:


for epoch in range(1):
    for i,(image,text,label) in enumerate(tqdm(train_loader)):
        label = torch.nn.functional.one_hot(label, num_classes=3)
        label=label.float()
        label=label.to(device)
        with torch.no_grad():
                text_clip,image_clip=clip(text,image)
                image_clip=image_clip.to(device)
                text_rep=textencoder(text).to(device)
        optimizer.zero_grad()  
        output=model(text_rep,image_clip).squeeze(dim=1)
        outputs_pooled = torch.mean(output, dim=1)
        loss = criterion(outputs_pooled, label)
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 10, loss.item()))
       


# In[ ]:





# In[ ]:





# In[ ]:




