# Neural_NER
Some Neural network models for named entity recognition(NER) Implementation a baseline window-based model, as well as a series of rnn model.

### Note:
* The project is educational purpose for implement different rnn models and exploring their pros and cons in NER task.
* The repo is a extension of cs224n assigment3. More detail can be find [here](http://web.stanford.edu/class/cs224n/assignment3/index.html)
* There are simple implementation of rnn cells (rnn,gru,lsm) in `models/` which can be replaced by Tensorflow built-in functions.
* The baseline model is relative fast under CPU. However, a GPU is strongly recommended for training RNN models.


### Basic Usage:
* Please check the cs224n [assignment 3](http://web.stanford.edu/class/cs224n/assignment3/index.html) for installing dependencies etc.
* Set the hyper-parameters in `config.py`.
* To run baseline model with existing dataset
```shell
python window.py train
```
* To run rnn models
```shell
python rnn.py train --cell [rnn|gru|lstm]
```

### TODO
- [x] implement lstm cell
- [ ] add tensorboard
- [ ] implement BiLSTM+CRF architecture

### Further Observations