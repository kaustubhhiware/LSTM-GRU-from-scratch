# LSTM-GRU-from-scratch

LSTM, GRU cell implementation from scratch

Assignment 4 weights for Deep Learning, CS60010.

Currently includes weights for LSTM and GRU for hidden layer size as 32, 64, 128 and 256.

## Objective

The aim of this assignment was to compare performance of LSTM, GRU and MLP for a fixed number of iterations, with variable hidden layer size. Please refer to [Report.pdf](Report.pdf) for the details.

Suggested reading: [colah's blog on LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). That's really all you need.

* Loss as a function of iterations

![](https://imgur.com/axaJ0cU.jpg)


## Usage

`sh run.sh` - Run all hidden units LSTM GRU and report accuracy. [Output here](output.txt)

`python train.py --train` - Run training, save weights into `weights/` folder. Defaults to LSTM, hidden_unit 32, 30 iterations / epochs

`python train.py --train --hidden_unit 32 --model lstm --iter 5`: Train LSTM and dump weights. Run training with specified number of iterations. Default iterations are 50.

`python train.py --test --hidden_unit 32 --model lstm` - Load precomputed weights and report test accuracy.

## Code structure

* [`data_loader`](data_loader.py) is used to load data from zip files in `data` folder.
* [`module`](module.py) defines the basic LSTM and GRU code.
* [`train`](train.py) handles input and states the model.


## License

The MIT License (MIT) 2018 - [Kaustubh Hiware](https://github.com/kaustubhhiware).