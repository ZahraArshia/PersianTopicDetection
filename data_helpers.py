import numpy as np
import re
import itertools
from collections import Counter
from PreProccess import delete_Stopword,normalizer
from blaze.expr.strings import len

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(numdoc):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    from xlrd import open_workbook
    
    wb = open_workbook('data/TopicDetectionData.xls')
    for s in wb.sheets():
        # print 'Sheet:',s.name
        Data = []
        lab = []
        values = []
        for row in range(s.nrows):
            col_value = []
            for col in range(s.ncols):
                value = (s.cell(row, col).value)
                try:
                    value = str(int(value))
                except:
                    pass
                col_value.append(value)
            values.append(col_value)
            Data.append((col_value[2]))
            lab.append(col_value[0])
    
    Data=Data[1:5001]
    lab=lab[1:5001]
    
    
    setLabel = list(set(lab))
    
    label     = []
    
    
    # joda kardane barchasb ha az baghiye dar har sanad
    class1_examples=[]
    class2_examples=[]
    class3_examples=[]
    class4_examples=[]
    class5_examples=[]
    i=-1
    counter = []
    for i in range(5):
        counter.append(numdoc)
    # joda kardane barchasb ha az baghiye dar har sanad
    class1_examples = []
    class2_examples = []
    class3_examples = []
    class4_examples = []
    class5_examples = []
    i = -1
    for data in ((Data)):
        i += 1
        k = normalizer(delete_Stopword(data))
        l = lab[i]
        if setLabel[0] == l:
            if counter[0] < 0:
                continue
            else:
                label.append(1)
                class1_examples.append(k)
                counter[0] -= 1
        if setLabel[1] == l:
            if counter[1] < 0:
                continue
            else:
                label.append(2)
                class2_examples.append(k)
                counter[1] -= 1
        if setLabel[2] == l:
            if counter[2] < 0:
                continue
            else:
                label.append(3)
                class3_examples.append(k)
                counter[2] -= 1
        if setLabel[3] == l:
            if counter[3] < 0:
                continue
            else:
                label.append(4)
                class4_examples.append(k)
                counter[3] -= 1
        if setLabel[4] == l:
            if counter[4] < 0:
                continue
            else:
                label.append(5)
                class5_examples.append(k)
                counter[4] -= 1
    
    
    # Load data from files
    class1_examples = [s.strip() for s in class1_examples]
    class2_examples = [s.strip() for s in class2_examples]
    class3_examples = [s.strip() for s in class3_examples]
    class4_examples = [s.strip() for s in class4_examples]
    class5_examples = [s.strip() for s in class5_examples]
    # Split by words
    x_text = class1_examples + class2_examples+class3_examples + class4_examples + class5_examples
    #x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    class1_labels = [[0,0,0,0, 1] for _ in class1_examples]
    class2_labels = [[0,0,0,1, 0] for _ in class2_examples]
    class3_labels = [[0,0,1,0, 0] for _ in class3_examples]
    class4_labels = [[0,1,0,0, 0] for _ in class4_examples]
    class5_labels = [[1,0,0,0, 0] for _ in class5_examples]
    
    
    y = np.concatenate([class1_labels, class2_labels,class3_labels ,class4_labels,class5_labels], 0)
    print('Data loaded')
    return [x_text, y]


def pad_sentences(sentences,numdoc, padding_word="<PAD/>" ):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length=0
    ls=[]
    for x in sentences:
        x=list(x)
        j=0
        for s in x:
            j+=1
        ls.append(j)
        if j>sequence_length:
            sequence_length=j
    padded_sentences = []
    for i in range(numdoc*5):
        sentence = sentences[i]
        num_padding = sequence_length - ls[i]
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(numdoc):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    import numpy as np
    sentences, labels = load_data_and_labels(numdoc)
    sentences_padded = pad_sentences(sentences,numdoc)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)


    # save file
    '''import json
    import numpy as np
    with open('vocabulary.json', 'w') as outfile:
        json.dump(vocabulary, outfile)
    with open('vocabulary_inv_list.json', 'w') as outfile:
        json.dump(vocabulary_inv, outfile)
    np.save('xData', x)
    np.save('yData', x)



    #load file
    with open('vocabulary.json', 'r') as fp:
        vocabulary = json.load(fp)
    
    with open('vocabulary_inv_list.json', 'r') as fp:
        vocabulary_inv = json.load(fp)
        
    x=np.load('xData2.npy')
    y=np.load('yData.npy')'''
    
    return [x, y, vocabulary, vocabulary_inv]


def load_my_data():
    
    import json
    import numpy as np
    with open('vocabulary.json', 'r') as fp:
        vocabulary = json.load(fp)
    
    with open('vocabulary_inv_list.json', 'r') as fp:
        vocabulary_inv = json.load(fp)
        
    x=np.load('xData.npy')
    y=np.load('yData.npy')
    
    return [x, y, vocabulary, vocabulary_inv]



def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]




