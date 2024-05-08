# scoRNN

Code update for the paper "Orthogonal Recurrent Neural Networks with Scaled Cayley Transform", https://arxiv.org/abs/1707.09520. If you find this code useful, please cite the paper.

Uses Tensorflow. To run, download the desired experiment code as well as the "scoRNN.py" script. 

Each script uses command line arguments to specify the desired architecture. For example, to run the MNIST experiment with a hidden size of 170 and RMSprop optimizer with learning rate 1e-3 to update the recurrent weight, type in the command line: 

Due to the changes in the version of tensorflow, I have modified in order to run the scoRNN_mnist.py experiment in the paper, see command below
```
python scoRNN_MNIST.py "your_model" 170 "rmsprop" 1e-3
```

```
Ex: python scoRNN_mnist.py "scoRNN" 170 "rmsprop" 1e-3
```
