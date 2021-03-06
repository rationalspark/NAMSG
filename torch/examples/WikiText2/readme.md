# WikiText-2 training in PyTorch with ARSG

##How to run

This example is built based on the pytorch official example to train WikiText-2, available at https://github.com/pytorch/examples/tree/master/word_language_model. To run it, we should prepare the enviroment and data following the original md file, as listed below.

Examples of the scripts to run are:

RSG
python -u main.py --model LSTM --opt RSG  --lr 100.0 --beta 0.999000 --wd 1.000000e-05 --dropout 2.000000e-01 --clip 0.250000 --epochs 40 --epo_dec '20 30' --mu 0.1  --eps 1e-08

ARSG
python -u main.py --model LSTM --opt ARSG  --lr 0.020000 --beta 0.999000 --beta2 0.990000 --wd 1.000000e-05 --dropout 2.000000e-01 --clip 0.250000 --epochs 40 --epo_dec '20 30' --mu 0.1  --eps 1e-08

ARSGB
python -u main.py --model LSTM --opt ARSGB  --lr 0.050000 --beta 0.999000 --beta2 0.990000 --wd 1.000000e-05 --dropout 2.000000e-01 --clip 0.250000 --epochs 40 --epo_dec '10 20 30' --mu 0.1  --eps 1e-08

Other optimizers can be applied by setting "--opt", the options are:
SGD, NAG, ADAM, AMSG, NADAMA, and RANGER

# Original md file: Word-level language modeling RNN

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash 
python main.py --cuda --epochs 6           # Train a LSTM on Wikitext-2 with CUDA
python main.py --cuda --epochs 6 --tied    # Train a tied LSTM on Wikitext-2 with CUDA
python main.py --cuda --epochs 6 --model Transformer --lr 5   
                                           # Train a Transformer model on Wikitext-2 with CUDA
python main.py --cuda --tied               # Train a tied LSTM on Wikitext-2 with CUDA for 40 epochs
python generate.py                         # Generate samples from the trained LSTM model.
python generate.py --cuda --model Transformer
                                           # Generate samples from the trained Transformer model.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help                       show this help message and exit
  --data DATA                      location of the data corpus
  --model MODEL                    type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE                  size of word embeddings
  --nhid NHID                      number of hidden units per layer
  --nlayers NLAYERS                number of layers
  --lr LR                          initial learning rate
  --clip CLIP                      gradient clipping
  --epochs EPOCHS                  upper epoch limit
  --batch_size N                   batch size
  --bptt BPTT                      sequence length
  --dropout DROPOUT                dropout applied to layers (0 = no dropout)
  --decay DECAY                    learning rate decay per epoch
  --tied                           tie the word embedding and softmax weights
  --seed SEED                      random seed
  --cuda                           use CUDA
  --log-interval N                 report interval
  --save SAVE                      path to save the final model
  --onnx-export                    path to export the final model in onnx format
  --transformer_head N             the number of heads in the encoder/decoder of the transformer model
  --transformer_encoder_layers N   the number of layers in the encoder of the transformer model
  --transformer_decoder_layers N   the number of layers in the decoder of the transformer model
  --transformer_d_ff N             the number of nodes on the hidden layer in feed forward nn
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40           
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied    
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40        
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied 
```


