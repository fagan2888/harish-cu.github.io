
### Tensorflowjs has to be installed for the execution of the model in the browser

### Importing necessary modules

import os
import nltk
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from matplotlib import pyplot as plt
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import tensorflowjs as tfjs


### Downloading the books from Gutenberg using NLTK

nltk.download('gutenberg')
nltk.download('punkt')


### Using the first 1000 sentences from the three books for the model

num_sentences = 1000
sentences = []
for book in ['austen-emma.txt', 'bryant-stories.txt', 'chesterton-ball.txt']:
  sentences.append(nltk.corpus.gutenberg.sents(book)[3:num_sentences+3])
sentences = [word for book in sentences for word in book]
sentence_classes = [0]*num_sentences + [1]*num_sentences + [2]*num_sentences

### Using stratified shuffle split from Scikit to split training and test data

sss = StratifiedShuffleSplit(n_splits = 2, train_size=0.8, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(sentences, sentence_classes):
  x_train = [sentences[i] for i in train_index]
  x_test = [sentences[i] for i in test_index]
  y_train = [sentence_classes[i] for i in train_index]
  y_test = [sentence_classes[i] for i in test_index]


x_test_copy = x_test.copy()
y_test_copy = y_test.copy()

max_len = 20
num_words = 1000

### Fitting the tokenizer on the training data

t = Tokenizer(num_words=num_words)
t.fit_on_texts(x_train)

### Storing the metadata for model deployment

metadata = {
  'word_index': t.word_index,
  'max_len': max_len,
  'vocabulary_size': num_words,
}

### Vectorizing and Padding

x_train = t.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')


### Dense Model (Feedforward)

embedding_size = 8
n_classes = 3
epochs = 20

model = keras.Sequential()
model.add(keras.layers.Embedding(num_words, embedding_size, input_shape=(max_len,)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(3, activation='softmax'))
model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.2)

x_test = t.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')


### Testing the model

loss, accuracy = model.evaluate(x_test, y_test)


def plot_history(histories, key):
  plt.figure(figsize=(20,10))
    
  for name, history in histories:
    val = plt.plot([x+1 for x in history.epoch], history.history['val_'+key], '--', label=name.title()+' Val')
    plt.plot([x+1 for x in history.epoch], history.history[key], color=val[0].get_color(), label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.xticks([x+1 for x in history.epoch])
  plt.ylabel(key.replace('_',' ').title())
  plt.title('Plot of '+ name.title() +' at different Epochs', fontsize=18)
  plt.legend()

### Plotting the loss and accuracy for the Dense Model

plot_history([('Dense Model', history)], key='loss')
plot_history([('Dense Model', history)], key='acc')


## LSTM Model

embedding_size = 128
n_classes = 3
epochs = 20


model_LSTM = keras.Sequential()
model_LSTM.add(keras.layers.Embedding(num_words, embedding_size, input_shape=(max_len,)))
model_LSTM.add(keras.layers.LSTM(32, return_sequences=True))
model_LSTM.add(keras.layers.LSTM(32, return_sequences=True))
model_LSTM.add(keras.layers.LSTM(32))
model_LSTM.add(keras.layers.Dense(32, activation='relu'))
model_LSTM.add(keras.layers.Dense(3, activation='softmax'))
model_LSTM.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

history_LSTM = model_LSTM.fit(x_train, y_train, epochs=epochs, validation_split=0.2)

loss, accuracy = model_LSTM.evaluate(x_test, y_test)

### Plotting the loss and accuracy for the LSTM Model

plot_history([('LSTM Model', history_LSTM)], key='loss')
plot_history([('LSTM Model', history_LSTM)], key='acc')


### Using the Dense Feedforward Model as the LSTM does not show better performance for this dataset (chosen books) despite testing out different architectures

metadata_json_path = os.path.join(MODEL_DIR, 'metadata.json')
json.dump(metadata, open(metadata_json_path, 'wt'))
tfjs.converters.save_keras_model(model, MODEL_DIR)
print('\nSaved model artifcats in directory: %s' % MODEL_DIR)


### Writing an index.html and an index.js file configured to load our model

index_html = """
<!doctype html>

<body>
  <style>
    #textfield {
      font-size: 120%;
      width: 60%;
      height: 200px;
    }
  </style>
  <h1>
    Title
  </h1>
  <hr>
  <div class="create-model">
    <button id="load-model" style="display:none">Load model</button>
  </div>
  <div>
    <div>
      <span>Vocabulary size: </span>
      <span id="vocabularySize"></span>
    </div>
    <div>
      <span>Max length: </span>
      <span id="maxLen"></span>
    </div>
  </div>
  <hr>
  <div>
    <select id="example-select" class="form-control">
      <option value="example1">Austen-Emma</option>
      <option value="example2">Bryant-Stories</option>
      <option value="example3">Chesterton-Ball</option>
    </select>
  </div>
  <div>
    <textarea id="text-entry"></textarea>
  </div>
  <hr>
  <div>
    <span id="status">Standing by.</span>
  </div>

  <script src='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js'></script>
  <script src='index.js'></script>
</body>
"""


index_js = """
const HOSTED_URLS = {
  model:
      'model_js/model.json',
  metadata:
      'model_js/metadata.json'
};

const examples = {
  'example1':
      'And therefore I think I will beg your excuse and take my three turns -- my winter walk.',
  'example2':
      'And then each saw the rope in the others hold.',
  'example3':
      'It is a parable of you and all your rationalists.'      
};

function status(statusText) {
  console.log(statusText);
  document.getElementById('status').textContent = statusText;
}

function showMetadata(metadataJSON) {
  document.getElementById('vocabularySize').textContent =
      metadataJSON['vocabulary_size'];
  document.getElementById('maxLen').textContent =
      metadataJSON['max_len'];
}

function settextField(text, predict) {
  const textField = document.getElementById('text-entry');
  textField.value = text;
  doPredict(predict);
}

function setPredictFunction(predict) {
  const textField = document.getElementById('text-entry');
  textField.addEventListener('input', () => doPredict(predict));
}

function disableLoadModelButtons() {
  document.getElementById('load-model').style.display = 'none';
}

function doPredict(predict) {
  const textField = document.getElementById('text-entry');
  const result = predict(textField.value);
  score_string = "Class scores: ";
  for (var x in result.score) {
    score_string += x + " ->  " + result.score[x].toFixed(3) + ", "
  }
  //console.log(score_string);
  status(
      score_string + ' elapsed: ' + result.elapsed.toFixed(3) + ' ms)');
}

function prepUI(predict) {
  setPredictFunction(predict);
  const testExampleSelect = document.getElementById('example-select');
  testExampleSelect.addEventListener('change', () => {
    settextField(examples[testExampleSelect.value], predict);
  });
  settextField(examples['example1'], predict);
}

async function urlExists(url) {
  status('Testing url ' + url);
  try {
    const response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

async function loadHostedPretrainedModel(url) {
  status('Loading pretrained model from ' + url);
  try {
    const model = await tf.loadModel(url);
    status('Done loading pretrained model.');
    disableLoadModelButtons();
    return model;
  } catch (err) {
    console.error(err);
    status('Loading pretrained model failed.');
  }
}

async function loadHostedMetadata(url) {
  status('Loading metadata from ' + url);
  try {
    const metadataJson = await fetch(url);
    const metadata = await metadataJson.json();
    status('Done loading metadata.');
    return metadata;
  } catch (err) {
    console.error(err);
    status('Loading metadata failed.');
  }
}

class Classifier {

  async init(urls) {
    this.urls = urls;
    this.model = await loadHostedPretrainedModel(urls.model);
    await this.loadMetadata();
    return this;
  }

  async loadMetadata() {
    const metadata =
        await loadHostedMetadata(this.urls.metadata);
    showMetadata(metadata);
    this.maxLen = metadata['max_len'];
    console.log('maxLen = ' + this.maxLen);
    this.wordIndex = metadata['word_index']
  }

  predict(text) {
    // Convert to lower case and remove all punctuations.
    const inputText =
        text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
    // Look up word indices.
    const inputBuffer = tf.buffer([1, this.maxLen], 'float32');
    for (let i = 0; i < inputText.length; ++i) {
      const word = inputText[i];
      inputBuffer.set(this.wordIndex[word], 0, i);
      //console.log(word, this.wordIndex[word], inputBuffer);s
    }
    const input = inputBuffer.toTensor();
    //console.log(input);

    status('Running inference');
    const beginMs = performance.now();
    const predictOut = this.model.predict(input);
    //console.log(predictOut.dataSync());
    const score = predictOut.dataSync();//[0];
    predictOut.dispose();
    const endMs = performance.now();

    return {score: score, elapsed: (endMs - beginMs)};
  }
};

async function setup() {
  if (await urlExists(HOSTED_URLS.model)) {
    status('Model available: ' + HOSTED_URLS.model);
    const button = document.getElementById('load-model');
    button.addEventListener('click', async () => {
      const predictor = await new Classifier().init(HOSTED_URLS);
      prepUI(x => predictor.predict(x));
    });
    button.style.display = 'inline-block';
  }

  status('Standing by.');
}

setup();
"""

with open('index.html','w') as f:
  f.write(index_html)
  
with open('index.js','w') as f:
  f.write(index_js)