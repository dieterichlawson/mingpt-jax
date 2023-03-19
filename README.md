# MinGPT-JAX

This is a didactic JAX implementation of a GPT-style transformer that trains on Shakespeare's collected works, inspired by Andrej Karpathy's 
MinGPT: https://github.com/karpathy/minGPT.

### Getting Started

After cloning the repository, `pip install` the requirements:

```
pip install -r requirements.txt
```

Then prepare the dataset by running `prepare.py`:

```
python3 data/shakespeare_char/prepare.py
```

This will create three files.

* `input.txt`, a text file containing Shakespeare's collected works.
* `train.npy`, a numpy archive containing the training set.
* `val.npy`, a numpy archive containing the validation set.
* `meta.pkl` a pickle file containing an encoder and decoder for the dataset.

Then, you can run training with

```
python3 train.py
```

This will train the transformer using a default set of parameters, including

* Number of layers: 6
* Number of heads: 6
* Embedding dimension: 192
* MLP hidden dimension: 4 * embedding dimension
* Block size: 128
* Batch size: 64
* Learning rate: 5e-4

While training, the script will periodically print the loss as well as a completion
for the prefix "O god, O god!".
