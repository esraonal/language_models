# Project Description

Author: Esra Onal

Date created: Jan 4, 2024    

<a target="_blank" href="https://colab.research.google.com/github/esraonal/language_models/blob/main/bert_lstm_models.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

# Seperation or corporation of recurrence and self-attention

This study investigates different aspects of the human language processing system and how they are reflected in recent deep neural network architectures.  Numerous research supports that neural networks that utilize recurrence not only show promising results in many natural language processing tasks but also give insights into how sentence comprehension takes place in humans and what type of complex cognitive operations underlie this process such as incrementality. However, successive neural networks such as [Transformers](https://arxiv.org/pdf/1706.03762.pdf) that make use of a self-attention mechanism are now considered the state-of-the-art in language modeling due to their strength in drawing direct relations between words in sequential data without recurrence.  They separately attend to different aspects of the human language processing system, but why one performs better than the other one is not yet clear. Therefore, what constitutes the core of this study is an architecture that combines both mechanisms. We specifically use BERT (Bidirectional Encoder Representations from Transformers) and LSTM layers to create and compare different language models.

In this regard, we will train three different language models with self-attention (BERT) and recurrence (LSTM) using [**masking approach**](https://aclanthology.org/N19-1423.pdf) for training.
* BERT model
* LSTM model
* BERT + LSTM model

For the nature of our project, we won't be using pretrained word embeddings in our models so we will train these models from scratch using [**TensorFlow**](https://www.tensorflow.org/install) on the [**Amazon Polarity**](https://huggingface.co/datasets/Siki-77/amazon6_polarity) dataset loaded from Hugging Face Datasets.

## Masked Language Modeling (MLM) (Pretraining)
We will make use of masking approach for training which is called **Masked Language Modeling** (MLM).  Before feeding word sequences into our models, 15% of the words in each sequence are replaced with a **[mask]** token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence.

## NEXT: Next Sentence Prediction (NSP)! (Fine-tuning)
(In progress)

The NSP task forces the model to understand the relationship between two sentences. In this task, BERT is required to predict whether the second sentence is related to the first one. During training, the model is fed with 50% of connected sentences and another half with random sentence sequence.

# How to train and evaluate the models from scratch

The code is available to run on Google Colab

<a target="_blank" href="https://colab.research.google.com/github/esraonal/language_models/blob/main/bert_lstm_models.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Set-up

We will train the models in [**TensorFlow**](https://www.tensorflow.org/install) with keras layers on Google Colab.  

* Necessay libraries

```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

from dataclasses import dataclass
import pandas as pd
import numpy as np
import glob
import re
from pprint import pprint
```

*  Environment variables
```
@dataclass
class Config:
    MAX_LEN = 128
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 7000
    EMBED_DIM = 64
    NUM_HEAD = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 1
    TRAIN_SIZE = 50000

config = Config()
```

# Dataset

Models are trained with 50,000 reviews (both negative and positive sentiment) from [**Amazon Polarity**](https://huggingface.co/datasets/Siki-77/amazon6_polarity) dataset from Hugging Face.


* Install following libraries in order to load the dataset from Hugging Face.
```
! pip install datasets
! pip install apache_beam
```

* Import the libraries.  We will use ```load_dataset```
```
import apache_beam
from datasets import load_dataset
```

Dataset is split into training and test samples.  Additionally, the dataset has data fields such as label showing negative and positive rating scores, title, context containing the text and feeling encoding 0 as negative and 1 as positive. 

See the data structure
```
DatasetDict({
    train: Dataset({
        features: ['label', 'title', 'context', 'feeling'],
        num_rows: 249624
    })
    test: Dataset({
        features: ['label', 'title', 'context', 'feeling'],
        num_rows: 207317
    })
})
```

For training our language models, only context data which contains the body of the document is used, without the label/feeling or the title data. In total, there are 249,624 training samples and 207,317 test samples in this dataset as seen below.  We will only make use of the first 50,000 samples to train our models.

One sample can be seen below.
```
'"Boutique" quality sailor suit. I liked it so much I even bought the coordinating dress for my daughter. You will not be disappointed with this find!'
```

# Data preprocessing

Since we need to feed numbers as vectors, not raw text to train our language models, we will vectorize the reviews. Also, when masked, the mask token replaces the token ID with 6999 as seen below.

Example:
```
Text sample: Definitely a good buy. I strongly recommend it!
Token IDs: [  271    7   40   99    4 2521 144    3  ] # this sequence is padded to the maxlen which is 128

Masked text sample: Definitely a good [mask]. I strongly recommend it!
Masked token IDs: [  271    7   40 6999    4 2521  144    3   ] # this sequence is padded to the maxlen which is 128
```

## Step 1: Create vocabulary and vectorize data

We will use the [**TextVectorization**](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization) to index the vocabulary found in the dataset. Later, we'll use the same layer instance to vectorize the samples .

Our layer will only consider the top 7,000 words, and will truncate or pad sequences to be actually 128 tokens long.

```
vocab_size = 7000  # Only consider the top 7k words
maxlen = 128  # Only consider the first 128 words of each amazon review
```
We will also use a customized standardization for this dataset and add the masked token to the vocabulary.
```
[mask] token ID: 6999
```
## Step 3: Mask 15% of the tokens

Since MLM is used to train our models, we will mask 15% of the tokens in our data. However, there is a problem with this masking approach since the model only tries to predict when the [mask] token is present in the input, while we want the model to try to predict the correct tokens regardless of what token is present in the input. To deal with this issue, out of the 15% of the tokens selected for masking:
- 80% of the tokens are actually replaced with the token [mask].
- 10% of the time tokens are replaced with a random token.
- 10% of the time tokens are left unchanged.

Masked data is used as the input while unmasked data as the labels.
 
# Masked Language Models

## Model 1: BERT
![bert](https://github.com/esraonal/language_models/blob/main/bert.png)
## Model 2: LSTM
![lstm](https://github.com/esraonal/language_models/blob/main/lstm.png)
## Model 3: BERT + LSTM
We will combine the self-attention module with two LSTM layers.

![bert_lstm](https://github.com/esraonal/language_models/blob/main/bert_lstm.png)

# Training

## Callbacks

Before we train, we will create some [callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback) to see the progress of the model with each epoch.

An output is seen below after 10 epoch.  The model tries to predict the masked token and this specific callback function gives us the top 5 most candidates for the masked token.

```
{'input_text': 'definitely a good buy i strongly [mask] it',
 'predicted mask token': 'recommend',
 'prediction': 'definitely a good buy i strongly recommend it',
 'probability': 0.73725724}
{'input_text': 'definitely a good buy i strongly [mask] it',
 'predicted mask token': 'buy',
 'prediction': 'definitely a good buy i strongly buy it',
 'probability': 0.019802665}
{'input_text': 'definitely a good buy i strongly [mask] it',
 'predicted mask token': 'read',
 'prediction': 'definitely a good buy i strongly read it',
 'probability': 0.01913188}
{'input_text': 'definitely a good buy i strongly [mask] it',
 'predicted mask token': 'give',
 'prediction': 'definitely a good buy i strongly give it',
 'probability': 0.01881243}
{'input_text': 'definitely a good buy i strongly [mask] it',
 'predicted mask token': 'waste',
 'prediction': 'definitely a good buy i strongly waste it',
 'probability': 0.013175488}
```

# Evaluation

## Predict next word

One of the predictions for this sample from the test set is given below.

```
Input: I would
Predition:  I would recommend this book it again
```
## Predict masked token

One of the predictions for this sample from the test set is given below. 

```
Input: Definitely a good [mask]. I strongly recommend it!
(Expected) Output: Definitely a good buy. I strongly recommend it! # (real example)
(Predicted) Output: Definitely a good product. I strongly recommend it!
```

