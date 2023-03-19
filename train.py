from absl import app
from pathlib import Path
import pickle

from ml_collections import config_dict
from ml_collections import config_flags

import jax
import jax.numpy as jnp

import numpy as onp
import optax

from model import Transformer, make_weight_decay_mask

config = config_dict.ConfigDict()
config.seed = 42

config.model = config_dict.ConfigDict()
config.model.num_layers = 6
config.model.num_heads = 6
config.model.embed_dim = 192
config.model.mlp_hdim = None # Defaults to 4 * embd_dim
config.model.block_size = 128

config.train = config_dict.ConfigDict()
config.train.batch_size = 64
config.train.lr = 5e-4
config.train.beta1 = 0.9
config.train.beta2 = 0.95
config.train.weight_decay = 0.1
config.train.grad_norm_clip = 1.

config.summary = config_dict.ConfigDict()
config.summary.summarize_every = 10

_CFG = config_flags.DEFINE_config_dict('cfg', config)

def make_model(cfg, key, vocab_size):
  return Transformer(
          key,
          vocab_size=vocab_size,
          embed_dim=cfg.embed_dim,
          block_size=cfg.block_size,
          num_layers=cfg.num_layers,
          num_heads=cfg.num_heads,
          mlp_hdim=cfg.mlp_hdim,
          causal_mask=True)


def load_dataset(cfg, split):
  path_prefix = Path("data") / "shakespeare_char"
  meta_path = path_prefix / "meta.pkl"
  with open(meta_path, "rb") as f:
    dataset_info = pickle.load(f)
  data_path = path_prefix / f"{split}.npy"
  data = onp.load(data_path)
  data = jnp.array(data, dtype=jnp.uint8)
  num_tokens = data.shape[0]
  # Add one because we don't want the targets to run off the end.
  num_blocks = num_tokens - (cfg.model.block_size + 1)
  batches_per_epoch = num_blocks // cfg.train.batch_size

  @jax.jit
  def index_fn(i):
    seq = jax.vmap(jax.lax.dynamic_slice_in_dim, in_axes=(None, 0, None)
           )(data, i, cfg.model.block_size + 1)
    # Split data into inputs and targets
    return seq[:, :cfg.model.block_size], seq[:, 1:]

  def gen(key):
    inds = jax.random.permutation(key, num_blocks)
    inds = inds[:batches_per_epoch * cfg.train.batch_size]
    inds = inds.reshape([batches_per_epoch, cfg.train.batch_size])

    for i in range(batches_per_epoch):
      yield index_fn(inds[i])

  return gen, dataset_info

def train(cfg):
  key = jax.random.PRNGKey(cfg.seed)
  data_itr, ds_info = load_dataset(cfg, 'train')

  def encode(s: str):
    return [ds_info['stoi'][c] for c in s]

  def decode(a: jnp.ndarray) -> str:
    return "".join([ds_info['itos'][int(a[i])] for i in range(a.shape[0])])

  key, subkey = jax.random.split(key)
  model = make_model(cfg.model, subkey, ds_info['vocab_size'])

  def loss(model, inputs, targets):
    # (block_size, vocab_size)
    logits = model(inputs)
    # (block_size, vocab_size)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    ce_loss = - jnp.mean(jnp.take_along_axis(log_probs, targets[:, None], axis=1))
    return ce_loss

  def batch_loss(model, batch_inputs, batch_targets):
    batch_loss = jax.vmap(loss, in_axes=(None, 0, 0))(model, batch_inputs, batch_targets)
    return jnp.mean(batch_loss)

  @jax.jit
  def summary_generate(key, model):
    prefix_str = "O god, O god!"
    prefix_enc = jnp.array(encode(prefix_str))
    generation = model.generate(key, prefix_enc)
    return generation

  def summarize(key, model, loss_val, step):
    print(f"Step {step} loss: {loss_val:0.4f}")
    generation = summary_generate(key, model)
    generated_str = decode(generation)
    print(generated_str)

  opt = optax.adamw(cfg.train.lr,
                    b1=cfg.train.beta1, b2=cfg.train.beta2,
                    weight_decay=cfg.train.weight_decay, mask=make_weight_decay_mask)

  opt_state = opt.init(model)

  @jax.jit
  def train_step(model, batch_inputs, batch_targets, cur_opt_state):
    loss, grads = jax.value_and_grad(batch_loss)(model, batch_inputs, batch_targets)
    updates, new_opt_state = opt.update(grads, cur_opt_state, model)
    model = optax.apply_updates(model, updates)
    return loss, model, new_opt_state

  for t, (inps, targs) in enumerate(data_itr(key)):
    loss_val, model, opt_state = train_step(model, inps, targs, opt_state)
    key, subkey = jax.random.split(key)
    if t % cfg.summary.summarize_every == 0:
      summarize(subkey, model, loss_val, t)


def main(_):
  train(config)


if __name__ == '__main__':
  app.run(main)
