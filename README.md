# ML-deep-learning
use pytorch to implement a two-layer NN and a convolutional NN  
  
PyTorch Instructions:  
https://pytorch.org/docs/stable/index.html  
  
FashionMNIST dataset:  
https://github.com/zalandoresearch/fashion-mnist  
Download as train.feats.npy, train.labels.npy, dev.feats.npy, dev.labels.npy, and test.feats.npy and store them in a file named data.

Command Line Arguments:  
mode: train or predict.    
--data-dir: point to the directory where data files are stored.  
--log-file: (used during train mode) point to the location where a csv file containing logs should be stored.  
--model-save: (used during train mode) point to the location where a model should be stored and (used during predict mode) loaded from.  
--prediction-file: (used during predict mode) point to where to output model predictions.  
--model: 3 models are simple-ff, simple-cnn and best.  
--train-steps: each step trains on one batch.  
--batch-size: the number of examples in the batch.  
--learning-rate: the learning rate to use with optimizer during training.  
--ff-hunits: the number of hidden units in feed-forward layer 1.  
--cnn-n1-channels: the number of channels in the first conv layer.  
--cnn-n1-kernel: the size of the square kernel of the first conv layer.  
--cnn-n2-kernel: the size of the square kernel of the second conv layer.  
train:  
python main.py train --data-dir data --log-file ff-logs.csv --model-save ff.torch --model simple-ff  
predict:  
python main.py predict --data-dir data --model-save ff.torch --predictions-file ff-preds


  
