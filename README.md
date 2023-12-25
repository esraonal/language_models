# language_models

Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

# Incorporating incrementality into self-attention mechanism 
# Description

This study investigates different aspects of the human language processing system and how they are reflected in recent deep neural network architectures.  Numerous research supports that neural networks that utilize recurrence not only show promising results in many natural language processing tasks but also give insights into how sentence comprehension takes place in humans and what type of complex cognitive operations underlie this process such as incrementality. However, successive neural networks such as Transformers that make use of a self-attention mechanism are now considered the state-of-the-art in language modeling due to their strength in drawing direct relations between words in sequential data without recurrence.  They separately attend to different aspects of the human language processing system, but why one performs better than the other one is not yet clear for language processing and what the results of an architecture that combines both mechanisms constitutes the core of this study. We specifically use BERT (Bidirectional Encoder Representations from Transformers) and (Bi)LSTM layers to create the language models.

## 
In this regard, we trained four different language models layered using self-attention (BERT), recurrence (LSTM) and bidirectionality (BiLSTM).

## dataset

Models are trained with 50,000 random reviews (both negative and positive sentiment) from the Amazon Polarity dataset from Hugging Face. The dataset is mostly used for text-classification and sentiment-classification, but for training our language models, only content data which contains the body of the document is used, without the label or the title data. In total, there are 3,600,000 training samples and 400,000 test samples in this dataset as seen below.

```
DatasetDict({
    train: Dataset({
        features: ['label', 'title', 'content'],
        num_rows: 3600000
    })
    test: Dataset({
        features: ['label', 'title', 'content'],
        num_rows: 400000
    })
})
```

One sample can be seen below

```
'This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^'
```

@dataclass
class Config:
    MAX_LEN = 256
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 30000
    EMBED_DIM = 128
    NUM_HEAD = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 1


config = Config()

## data preprocessing

Since we need to feed numbers as vectors, not strings to train our language models, we need to vectorize the reviews. Additionally, since we are using BERT and self-attention, masking has been chosen as the appropriate approach to prepare the data for training. This way, the task is to predict the masked token in a given contect. The input fed into the model is the masked token IDs, and the expected output is the actual token IDs. 

```
Actual sample: Definitely a good buy. I strongly recommend it!
Actual token IDs: [  321     5  38    87     4  1755   135     7  ]

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

## autoagressive vs masked

First langauge model is trained with the encoder from the classical Transformers architecture.

[put a pic]

# Installation

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
