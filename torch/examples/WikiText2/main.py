# coding: utf-8
from __future__ import print_function
import sys
#sys.path.append('/home/aistudio/lib')
import numpy as np
import time
import torch.optim as optim
#from BayesOpt import * 

import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--opt', type=str, default='SGD',
                    help='type of optimizer (SGD, RSG, NAG, ADAM, NADAMA, AMSG, ARSG, ARSGB, RANGER)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--beta', type=float, default=0.9,
                    help='momentum coefficient') 
parser.add_argument('--beta2', type=float, default=0.99,
                    help='momentum coefficient for preconditioner') 
parser.add_argument('--mu', type=float, default=0.1,
                    help='Obs factor for RSG and ARSG')                    
parser.add_argument('--wd', type=float, default=0.,
                    help='weight decay')      
parser.add_argument('--nLA', type=int, default=6,
                    help='lookahead inner loop number, available in RSGLA mode')
parser.add_argument('--alphaLA', type=float, default=0.5,
                    help='update ratio for slow weights in the lookahead mode')             
parser.add_argument('-facDec', type=float, default=4.,
                    help='factor of LR decay')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='epsilon for adaptive methods')                     
parser.add_argument('--nLrDecInterv', type=int, default=1,
                    help='interval between lr decay, valid if epo_dec is not empty')
parser.add_argument('--epo_dec', type=str, default='',
                    help='epoch numbers to decay lr.')                     
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping, minus to disable')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size') #Original 20
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)') #Original 0.2
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed, minus to disable')
                  
#parser.add_argument('--log-interval', type=int, default=200, metavar='N',
#                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
if(args.seed>0):
    torch.manual_seed(args.seed)
if torch.cuda.is_available():
    gpuid = 0
    device = torch.device("cuda:%d" % gpuid)
    print("CUDA is available")
else:
    device = torch.device("cpu")
    gpuid = -1
    print("CUDA is not available, fall back to CPU.")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss()#nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()  
        
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip >0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        #for p in model.parameters():
        #    p.data.add_(-lr, p.grad.data)
        optimizer.step()

        total_loss += loss.item()
        '''
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.3f} | ms/batch {:5.3f} | '
                    'loss {:5.3f} | ppl {:8.3f}'.format(
                indEpo, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        '''
    cur_loss = total_loss / (batch+1.)
    elapsed = time.time() - start_time
    if cur_loss >= 100:
        print('Loss is {}, divergence'.format(cur_loss))
        pFileRes=open('Loss.txt','w')
        pFileRes.write("%f\n"%cur_loss)
        pFileRes.write("%f\n"%1e8)
        pFileRes.close()
        exit()
    print('| epoch {:3d} | {:5d} batches | lr {:5.3e} | ms/batch {:5.3f} | '
            'loss {:5.3f} | ppl {:8.3f}'.format(
        indEpo, batch+1, fLr*fScaleLr, elapsed * 1000 / (batch+1.), cur_loss, math.exp(cur_loss))) 
    return cur_loss

def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
fLr = args.lr
fBeta =args.beta
fBeta2 =args.beta2
fFacDec = args.facDec
fMu = args.mu
fWd = args.wd
fEps = args.eps
best_val_loss = None
listEpoDec = []
for strEpo in args.epo_dec.split():
    listEpoDec.append(int(strEpo))
if listEpoDec == []:
    print('Lr decay according to val loss, detecting interval {}'.format(args.nLrDecInterv))
else:
    print('Lr decay at the end of epo {}'.format(listEpoDec))

   
# At any point you can hit Ctrl + C to break out of training early.
try:
    #Initialize optimizer
    if args.opt=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=fLr, momentum=fBeta, weight_decay=fWd, nesterov=False)
    elif args.opt=='NAG':
        optimizer = optim.SGD(model.parameters(), lr=fLr, momentum=fBeta, weight_decay=fWd, nesterov=True)
    elif args.opt=='ADAM':
        optimizer = optim.Adam(model.parameters(), lr=fLr, betas=(fBeta, fBeta2),  amsgrad=False, weight_decay=fWd, eps=fEps)
    elif args.opt=='AMSG':
        optimizer = optim.Adam(model.parameters(), lr=fLr, betas=(fBeta, fBeta2),  amsgrad=True, weight_decay=fWd, eps=fEps)
    elif args.opt=='RSG':
        optimizer = optim.Arsg(model.parameters(), lr=fLr, betas=(fBeta, fBeta2), mu=fMu, adaptive=False, weight_decay=fWd, eps=fEps)    
    elif args.opt=='NADAMA':
        optimizer = optim.Arsg(model.parameters(), lr=fLr, betas=(fBeta, fBeta2), mu=fMu, amsgrad=False, weight_decay=fWd, eps=fEps)         
    elif args.opt=='ARSG':
        optimizer = optim.Arsg(model.parameters(), lr=fLr, betas=(fBeta, fBeta2), mu=fMu, amsgrad=True, weight_decay=fWd, eps =fEps)
    elif args.opt=='ARSGB':
        optimizer = optim.Arsg(model.parameters(), lr=fLr, betas=(fBeta, fBeta2), mu=fMu, amsgrad=True, weight_decay=fWd, eps =fEps)        
    elif args.opt=='RANGER':
        optimizer = optim.Ranger(model.parameters(), lr=fLr, betas=(fBeta, fBeta2), alpha=args.alphaLA, k=args.nLA, weight_decay=fWd, eps =fEps)
        #params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95,0.999), eps=1e-5, weight_decay=0
    fScaleLr = 1.
    if args.opt!='ARSG':
        lambLr = lambda iEpo: fScaleLr
        scheLr = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambLr)         
    nEpoLrDecLa = 0
    for indEpo in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.3f}s | valid loss {:5.3f} | '
                'valid ppl {:8.3f}'.format(indEpo, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save loss
        pFileRes=open('OptRes.txt','a')
        pFileRes.write("{}, {}\n".format(train_loss, val_loss))
        pFileRes.close()
        # Save the model if the validation loss is the best we've seen so far.
        bBest = False
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            bBest = True
        if (listEpoDec==[] and not bBest and indEpo % args.nLrDecInterv == 0) or indEpo in listEpoDec:
            # Anneal the learning rate if no improvement has been seen in the validation dataset, and the interval between lr decay >= least args.nLrDecInterv.
            fScaleLr /= fFacDec
            if args.opt!='ARSG' and args.opt!='ARSGB' and args.opt!='RSG' and args.opt!='NADAMA': 
                scheLr.step()
            else:
                #fBeta = max(1. -(1.-fBeta)*fFacDec, 0)
                if args.opt=='ARSGB':
                    fMu = args.mu *2
                optimizer.set_hyper_para(lr=fLr*fScaleLr, betas=(fBeta, fBeta2), mu=fMu, eps=fEps, weight_decay=fWd)
            print('Epo %d, reset lr %e, beta1 %f, mu %f'%(indEpo,fLr*fScaleLr,fBeta, fMu))
            #   optimizer.restart()
            #nEpoLrDecLa = indEpo
        
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.3f} | test ppl {:8.3f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

pFileRes=open('Loss.txt','w')
pFileRes.write("%f\n"%best_val_loss)
pFileRes.write("%f\n"%test_loss)
pFileRes.close()

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
