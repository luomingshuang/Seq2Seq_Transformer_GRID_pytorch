# Seq2Seq_Transformer_GRID_pytorch
Introduction
----
This is a project for seq2seq lip reading on a lip-reading dataset called GRID with transformer model.
In this project, we implemented it with Pytorch.

Dependencies
----
* Python: 3.6+
* Pytorch: 1.3+
* Others

Dataset
----
This project is trained on GRID (grayscale).

Training And Testing
----
About the modeling units in this work, we built our character table as following:
```
['<sos>', '<eos>', ' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
``` 
First, we need to config the parameters in the python file called "options.py". We also need to change 
the "pre_process.py" according to the entails about our data. 
And we preprocess the data with the following commands:
```
python pre_process.py  ##We can get grid.pickle
python ngram_lm.py  ##We can get bigram_freq.pkl
```
Then, we can set the number of GPUs for training according to our realistic devices. 
If we want to use 4 GPUs to train, we can change the parameters in "train.py", such as "model = nn.DataParallel(model, device_ids=[0,1,2,4])".
We can train our model with the following command:
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python train.py 
```
Last, when our training loss is converged, we can get the model called "BEST_checkpoint_characters.tar".
Here, we provide a final model which is available at [GoogleDrive]().
And copy the checkpoint to this folder. Here, we provide the beam search method with ngram_language model.
We can test the model as follows:
```
##When we test the model without language model, we set the beam size in "test_LM.py" to 1.
python test_LM.py
##When we test the model with language model, we set the beam size in "test_LM.py" to 2 (3,4,5).
python test_LM.py
```
Our testing results as follows (waiting):
```
beam size=1, WER=% (Baseline)
beam size=2, WER=%
beam size=3, WER=%
beam size=4, WER=%
beam size=5, WER=%
```