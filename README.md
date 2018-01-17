# Seqence labeling models
* Simple Implementation of some sequence labeling models, include HMM, DNN, RNN, GRU, LSTM, biLSTM+CRF.
* These architecture are evaluated by a NER(named entity recognition) task. 
* The HMM is used as baseline discrete model that cannot fit well on NER task.

### Note:
* The project is educational purpose for implement different rnn models and exploring their pros and cons in NER task.
* The repo is a extension of cs224n assigment3. More detail can be find [here](http://web.stanford.edu/class/cs224n/assignment3/index.html)
* There are simple implementation of rnn cells (rnn,gru,lsm) in `models/` which can be replaced by Tensorflow built-in functions.
* The baseline model (HMM, window-based DNN) is relative fast under CPU. However, a GPU is strongly recommended for training RNN models.


### Basic Usage:
* Please check the cs224n [assignment 3](http://web.stanford.edu/class/cs224n/assignment3/index.html) for installing dependencies etc.
* To run baseline model with existing dataset
```shell
python hmm.py train
python window.py train
```
* To run rnn models, first set the hyper-parameters in `config.py`
* CRF layer and biLSTM layer is triggered by `use_biRNN` and `use_crf`
* Then, run the script `rnn.py`
```shell
python rnn.py train --cell [rnn|gru|lstm]
```

### TODO
- [x] implement lstm cell
- [x] implement BiLSTM+CRF architecture
- [x] implement HMM baseline
- [ ] add tensorboard
