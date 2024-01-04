# Seperation or corporation of recurrence and self-attention 
# Project Description

This study investigates different aspects of the human language processing system and how they are reflected in recent deep neural network architectures.  Numerous research supports that neural networks that utilize recurrence not only show promising results in many natural language processing tasks but also give insights into how sentence comprehension takes place in humans and what type of complex cognitive operations underlie this process such as incrementality. However, successive neural networks such as Transformers that make use of a self-attention mechanism are now considered the state-of-the-art in language modeling due to their strength in drawing direct relations between words in sequential data without recurrence.  They separately attend to different aspects of the human language processing system, but why one performs better than the other one is not yet clear for language processing and what the results of an architecture that combines both mechanisms constitutes the core of this study. We specifically use BERT (Bidirectional Encoder Representations from Transformers) and (Bi)LSTM layers to create different language models.
 
In this regard, we will train three different language models with self-attention (BERT) and recurrence (LSTM). For the nature of our project, we won't be using pretrained word embeddings in our models so we will train these models from scratch using **TensorFlow** on the [**Amazon Polarity**](https://huggingface.co/datasets/Siki-77/amazon6_polarity) dataset loaded from Hugging Face Datasets. 

## Masked Language Modeling (MLM)
We will make use of masking approach for training which is called **Masked Language Modeling** (MLM).  Before feeding word sequences into our models, 15% of the words in each sequence are replaced with a **[mask]** token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence.

## NEXT: Next Sentence Prediction (NSP)! 
(In progress)

The NSP task forces the model to understand the relationship between two sentences. In this task, BERT is required to predict whether the second sentence is related to the first one. During training, the model is fed with 50% of connected sentences and another half with random sentence sequence.

## Set-up
We will train the models in **TensorFlow** with keras layers.  Let's import necessay libraries!

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
## Dataset

Models are trained with 50,000 reviews (both negative and positive sentiment) from [**Amazon Polarity**](https://huggingface.co/datasets/Siki-77/amazon6_polarity) dataset from Hugging Face. 

Install following libraries in order to load the dataset from Hugging Face.
```
! pip install datasets
! pip install apache_beam
```

Import the libraries.  We will use **load_dataset**
```
import apache_beam
from datasets import load_dataset
```

Load the dataset
```
dataset = load_dataset("Siki-77/amazon6_polarity")
print(dataset)
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

For training our language models, only context data which contains the body of the document is used, without the label/feeling or the title data. In total, there are 249,624 training samples and 207,317 test samples in this dataset as seen below.  We will only make use of the first 50,000 samples to train our models

Get trainign and test samples
```
train_size = 50000
train_amazon_review = []
test_amazon_review = []

length_train = len(dataset['train'])
for i in range(length_train):
    train_amazon_review.append(dataset['train'][i]['context'])

length_test = len(dataset['test'])

test_amazon_review = []
for i in range(length_test):
    test_amazon_review.append(dataset['test'][i]['context'])

train_amazon_review_subset = train_amazon_review[:train_size]
print(dataset['test'][100]['context'])
```

One sample can be seen below.
```
'"Boutique" quality sailor suit. I liked it so much I even bought the coordinating dress for my daughter. You will not be disappointed with this find!'
```

## Data preprocessing

Since we need to feed numbers as vectors, not raw text to train our language models, we need to vectorize the reviews.
 
```
Text sample: Definitely a good buy. I strongly recommend it!
Token IDs: [  321     5  38    87     4  1755   135     7  ]
```
Create the vocabulary

We will use the [**TextVectorization**](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization) to index the vocabulary found in the dataset. Later, we'll use the same layer instance to vectorize the samples .

Our layer will only consider the top 7,000 words, and will truncate or pad sequences to be actually 128 tokens long. 

```
vocab_size = 7000  # Only consider the top 20k words
maxlen = 128  # Only consider the first 200 words of each movie review
```
We will also use a customized standardization and add the masked token to the vocabulary.
```
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}\"~"), ""
    )

vectorizer = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=maxlen,
    standardize=custom_standardization
    )

texts = train_amazon_review_subset
vectorizer.adapt(texts)
vocab = vectorizer.get_vocabulary()
vocab = vocab[: vocab_size - 1] + ["[mask]"]
vectorizer.set_vocabulary(vocab)

mask_token_id = vectorizer(["[mask]"]).numpy()[0][0]
print(mask_token_id)

```
Additionally, since we are using BERT and self-attention, masking has been chosen as the appropriate approach to prepare the data for training. This way, the task is to predict the masked token in a given contect. The input fed into the model is the masked token IDs, and the expected output is the actual token IDs. 
 
```
Masked sample: Definitely a good [mask]. I strongly recommend it!
Masked token IDs: [  321     5    38 29999     4  1755   135     7  ]
```

One of the predictions for this sample from the test set is given below. 

```
Input: Definitely a good [mask]. I strongly recommend it!
(Expected) Output: Definitely a good buy. I strongly recommend it!
(Predicted) Output: Definitely a good product. I strongly recommend it!
```

However, there is a problem with this masking approach since the model only tries to predict when the [MASK] token is present in the input, while we want the model to try to predict the correct tokens regardless of what token is present in the input. To deal with this issue, out of the 15% of the tokens selected for masking:
- 80% of the tokens are actually replaced with the token [MASK].
- 10% of the time tokens are replaced with a random token.
- 10% of the time tokens are left unchanged.

For this project we are not using pre-trained word embeddings so models will also learn these embeddings. 
## autoagressive vs masked

First langauge model is trained with the encoder from the classical Transformers architecture.

[put a pic]

# Installation

```
pip install -r requirements.txt

```
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.
# Usage

Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.
# Support

Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.
# Roadmap

If you have ideas for releases in the future, it is a good idea to list them in the README.
Contributing

State if you are open to contributions and what your requirements are for accepting them.
For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.
You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.
Authors and acknowledgment

Show your appreciation to those who have contributed to the project.
License

For open source projects, say how it is licensed.
Project status

If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
