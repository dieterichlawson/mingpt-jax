import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import equinox as eqx
from jax.nn import initializers
from typing import List, Optional, Callable
import snax


def gelu(x):
  return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))


class LayerNorm(eqx.Module):

  W: jnp.ndarray
  b: Optional[jnp.ndarray]

  eps: float = eqx.static_field()

  def __init__(self, norm_shape, bias: bool, eps: float = 1e-5):
    self.eps = eps
    self.W = jnp.ones(norm_shape)
    if bias:
      self.b = jnp.zeros(norm_shape)
    else:
      self.b = None

  def __call__(self, x):
    norm_ndim = self.W.ndim
    assert x.shape[-norm_ndim:] == self.W.shape
    red_axes = tuple(range(-norm_ndim, 0))
    means = jnp.mean(x, axis=red_axes, keepdims=True)
    squared_diffs = jnp.mean(jnp.square(x - means), axis=red_axes, keepdims=True)
    x = (x - means) * jnp.power(squared_diffs + self.eps, -0.5)
    x = x * self.W
    if self.b is not None:
      x = x + self.b
    return x


class Embedding(eqx.Module):

  E: jnp.ndarray

  def __init__(self,
          key,
          vocab_size,
          embed_dim,
          W_init=initializers.glorot_normal()):
    self.E = W_init(key, (vocab_size, embed_dim))

  def __call__(self, indices):
    return jnp.take(self.E, indices, axis=0)


class SelfAttention(eqx.Module):

  W: jnp.ndarray

  causal_mask: bool = eqx.static_field()
  qvk_dim: int = eqx.static_field()

  def __init__(self,
               key,
               in_dim,
               qvk_dim,
               causal_mask=True,
               W_init=initializers.glorot_uniform()):
    self.W = W_init(key, (in_dim, 3 * qvk_dim))
    self.causal_mask = causal_mask
    self.qvk_dim = qvk_dim

  def __call__(self, x):
    """Runs self-attention.

    Args:
      x: A [seq_len, in_dim] tensor.
    Returns:
      outs: A [seq_len, val_dim] tensor
    """
    seq_len = x.shape[0]
    qvk = x @ self.W # (seq_len, 3 * qvk_dim)
    queries, vals, keys = jnp.split(qvk, 3, axis=1)
    raw_weights = (queries @ keys.T) / jnp.sqrt(self.qvk_dim) # (seq_len, seq_len)

    if self.causal_mask:
      raw_weights = raw_weights.at[jnp.triu_indices(seq_len, 1)].set(-jnp.inf)

    weights = jax.nn.softmax(raw_weights, axis=1)
    return weights @ vals # (seq_len, val_dim)


class MultiheadSelfAttention(eqx.Module):

  # The matrix that produces the queries, keys, and values for all attention heads
  # as a linear function of the input.
  W_qkv: jnp.ndarray

  # The matrix that produces the outputs of the attention block by linearly combining outputs
  # from each attention head.
  W_o: jnp.ndarray

  causal_mask: bool = eqx.static_field()
  num_heads: int = eqx.static_field()
  rep_dim: int = eqx.static_field()
  qkv_dim: int = eqx.static_field()

  def __init__(self,
               key,
               rep_dim,
               num_heads,
               causal_mask=True,
               W_init=initializers.glorot_uniform()):
    """Construct a multi-headed self attention block.

    Args:
      key: A JAX PRNGKey.
      rep_dim: The dimension of the token representation that are inputs to and outputs
        from this block. Most often set equal to the dimension of the token embeddings.
      num_heads: The number of heads. Must evenly divide rep_dim. The dimensionality of
        the query, key, and value vectors are all assumed to be equal to num_heads / rep_dim.
      causal_mask: If true, mask out the upper-triangular portion of the attention weights,
        preventing any token from attending to tokens in the past.
      W_init: A weight initializer.
    """
    assert rep_dim % num_heads == 0, \
            f"Num heads {num_heads} must evenly divide rep dim {rep_dim}."
    self.causal_mask = causal_mask
    self.num_heads = num_heads
    self.rep_dim = rep_dim
    self.qkv_dim = self.rep_dim // self.num_heads
    k1, k2 = jax.random.split(key)
    self.W_qkv = W_init(k1, (rep_dim, 3 * rep_dim))
    self.W_o = W_init(k2, (rep_dim, rep_dim))

  def __call__(self, x):
    # Multiply (seq_len, rep_dim) x (rep_dim, 3 * rep_dim)
    # gives (seq_len, rep_dim * 3) = (seq_len, 3 * num_heads * qkv_dim)
    qkv_raw_out = x @ self.W_qkv

    # Reshape to (seq_len, 3, num_heads, qkv_dim)
    qkv_reshape = jnp.reshape(qkv_raw_out, [-1, 3, self.num_heads, self.qkv_dim])

    # Each of qs, ks, and vs is (seq_len, num_heads, qkv_dim).
    # Squeeze is necessary because split leaves in a singleton dimension.
    queries, keys, vals = [jnp.squeeze(x, axis=1) for x in jnp.split(qkv_reshape, 3, axis=1)]

    # qs and vs are (seq_len, num_heads, qkv_dim). We want to batched matrix multiply
    # with batch dim num_heads and contracting dim qkv_dim.
    # We can use dot_general to do this, specifying dimension 1 as the batch dim
    # and dimension 2 as the contracting dim.
    # The result is shape (num_heads, seq_len, seq_len)
    raw_weights = jax.lax.dot_general(queries, keys, ((2, 2), (1, 1)))
    raw_weights = raw_weights / jnp.sqrt(self.qkv_dim) # (num_heads, seq_len, seq_len)

    if self.causal_mask:
      # Set above-diagonal elements to - infinity.
      f = lambda x: x.at[jnp.triu_indices_from(x, 1)].set(-jnp.inf)
      raw_weights = jax.vmap(f)(raw_weights)

    # (num_heads, seq_len, seq_len), normalized in the last dimension
    weights = jax.nn.softmax(raw_weights, axis=2)

    # ws is (num_heads, seq_len, seq_len) and vals is (seq_len, num_heads, qkv_dim).
    # We want to do a batched matrix multiply with batch dimension num_heads
    # and contracting dimension seq_len.
    # The result is (num_heads, seq_len, qkv_dim).
    head_outs = jax.lax.dot_general(weights, vals, ((2, 0), (0, 1)))
    # Transpose and reshape to (seq_len, num_heads * qkv_dim) = (seq_len, rep_dim)
    head_outs_reshape = jnp.reshape(
            jnp.transpose(head_outs, axes=(1, 0, 2)), [-1, self.rep_dim])

    # (seq_len, rep_dim) x (rep_dim, rep_dim) = (seq_len, rep_dim)
    out = head_outs_reshape @ self.W_o
    return out


class TransformerBlock(eqx.Module):

  sa: MultiheadSelfAttention
  mlp: snax.MLP
  ln_att: LayerNorm
  ln_mlp: LayerNorm

  def __init__(self,
               key,
               rep_dim: int,
               num_heads: int,
               mlp_hdim: int,
               causal_mask: bool = True,
               mlp_act_fn: Callable = gelu,
               ln_bias: bool = True,
               ln_eps: float = 1e-5,
               W_init=initializers.glorot_uniform(),
               b_init=initializers.zeros):
    k1, k2 = jax.random.split(key)
    self.sa = MultiheadSelfAttention(
        k1, rep_dim, num_heads, causal_mask=causal_mask, W_init=W_init)
    self.mlp = snax.MLP(
        k2, rep_dim, [mlp_hdim, rep_dim], mlp_act_fn, W_init=W_init, b_init=b_init)
    self.ln_att = LayerNorm((rep_dim,), ln_bias, ln_eps)
    self.ln_mlp = LayerNorm((rep_dim,), ln_bias, ln_eps)

  def __call__(self, x):
    x = x + self.sa(self.ln_att(x)) # (seq_len, rep_dim)
    mlp_out = jax.vmap(self.mlp)(self.ln_mlp(x)) #(seq_len, rep_dim)
    return x + mlp_out


class Transformer(eqx.Module):

  tok_embed: Embedding
  pos_embed: Embedding

  blocks: List[TransformerBlock]

  ln: LayerNorm
  out_affine: snax.Affine

  block_size: int = eqx.static_field()

  def __init__(self,
               key,
               vocab_size: int,
               embed_dim: int,
               block_size: int,
               num_layers: int,
               num_heads: int,
               mlp_hdim: Optional[int] = None,
               causal_mask: bool = True,
               mlp_act_fn: Callable = gelu,
               ln_bias: bool = True,
               ln_eps: float = 1e-5,
               W_init=initializers.glorot_uniform(),
               b_init=initializers.zeros):
    if mlp_hdim is None:
      mlp_hdim = 4 * embed_dim

    self.block_size = block_size

    key, sk1, sk2 = jax.random.split(key, num=3)
    self.tok_embed = Embedding(sk1, vocab_size, embed_dim)
    self.pos_embed = Embedding(sk2, block_size, embed_dim)

    self.blocks = []
    for _ in range(num_layers):
      key, subkey = jax.random.split(key)
      self.blocks.append(
          TransformerBlock(
              subkey, embed_dim, num_heads, mlp_hdim,
              causal_mask=causal_mask, mlp_act_fn=mlp_act_fn,
              ln_bias=ln_bias, ln_eps=ln_eps,
              W_init=W_init, b_init=b_init))

    self.ln = LayerNorm((embed_dim,), bias=ln_bias, eps=ln_eps)

    self.out_affine = snax.Affine(key, embed_dim, vocab_size, W_init=W_init)

  def __call__(self, x):
    seq_len = x.shape[0]
    assert seq_len <= self.block_size
    x = self.tok_embed(x) + self.pos_embed.E[:seq_len]
    for block in self.blocks:
      x = block(x)
    x = self.ln(x)
    logits = self.out_affine(x)
    return logits

  def generate(self, key, prefix):
    prefix_len = prefix.shape[0]
    num_new_tokens = self.block_size - prefix_len
    assert num_new_tokens > 0
    padded_prefix = jnp.pad(prefix, (0, num_new_tokens))
    assert padded_prefix.shape[0] == self.block_size

    def sample_next_token(key, in_tokens, prefix_len):
      logits = self.__call__(in_tokens)
      next_token_logits = logits[prefix_len - 1]
      next_token = jax.random.categorical(key, next_token_logits)
      return next_token

    def while_body(state):
      key, out_seq, t = state
      key, subkey = jax.random.split(key)
      next_token = sample_next_token(subkey, out_seq, t)
      out_seq = out_seq.at[t].set(next_token)
      return (key, out_seq, t+1)

    def while_cond(state):
      _, _, t = state
      return t < prefix_len + num_new_tokens

    init_state = (key, padded_prefix, prefix_len)

    outs = jax.lax.while_loop(while_cond, while_body, init_state)

    return outs[1]


def make_weight_decay_mask(model: Transformer):

  def mask_map(x):
    if type(x) is snax.Affine:
      # Don't weight decay the biases of dense layers.
      out = eqx.tree_at(lambda y: y.b, x, False)
      out = eqx.tree_at(lambda y: y.W, out, True)
      return out
    elif type(x) is Embedding or type(x) is LayerNorm:
      # Don't weight decay embeddings or layernorm parameters
      return False
    else:
      return True

  mask_is_leaf = lambda x: type(x) in [snax.Affine, Embedding, LayerNorm]
  mask = tree_map(mask_map, model, is_leaf=mask_is_leaf)
  return mask
