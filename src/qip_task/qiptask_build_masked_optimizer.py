# qiptask_build_masked_optimizer.py
import jax.numpy as jnp
import optax
import jax
from optax._src.base import GradientTransformation
#These functions vwere written using AI and validated by running optimizations and checking that
#the desired parameters were trained and that the desired transformation was obtained
def _full_shape_mask(train_mask: dict, params_tree: dict) -> dict:
    """
    Build a bool mask pytree with EXACTLY the same shapes as params_tree leaves.
    Accepts:
      - bool                -> whole-leaf on/off
      - 1D len==first dim   -> row-wise, broadcast across remaining dims
      - 1D len==last  dim   -> last-dim (e.g., time) wise, broadcast across leading dims
      - exact-shape array   -> use as-is
    """
    mask_full = {}
    for k, p in params_tree.items():
        if k not in train_mask:
            raise KeyError(f"Mask missing entry for key '{k}'")
        rule = train_mask[k]

        if isinstance(rule, (bool, jnp.bool_)):
            mask_full[k] = jnp.full(p.shape, bool(rule), dtype=bool)
            continue

        r = jnp.asarray(rule, dtype=bool)

        if r.shape == p.shape:
            mask_full[k] = r
            continue

        if r.ndim == 1 and p.ndim >= 1 and r.shape[0] == p.shape[0]:
            b = r.reshape((r.shape[0],) + (1,) * (p.ndim - 1))
            mask_full[k] = jnp.broadcast_to(b, p.shape)
            continue

        if r.ndim == 1 and p.ndim >= 2 and r.shape[0] == p.shape[-1]:
            b = r.reshape((1,) * (p.ndim - 1) + (r.shape[0],))
            mask_full[k] = jnp.broadcast_to(b, p.shape)
            continue

        raise ValueError(f"Incompatible mask for '{k}': rule {r.shape}, param {p.shape}")
    return mask_full

from jax import tree_util as jtu   # <<— add this

# ... _full_shape_mask stays the same ...

def _elementwise_mask_transform(mask_full: dict) -> GradientTransformation:
    """Optax transform that zeros entries element-wise according to mask_full."""
    def init_fn(params):
        return ()
    def update_fn(updates, state, params=None):
        # jax.tree_map -> jax.tree_util.tree_map
        masked = jtu.tree_map(lambda u, m: u * m.astype(u.dtype), updates, mask_full)
        return masked, state
    return GradientTransformation(init_fn, update_fn)

def make_adam_with_mask(
    train_mask: dict,
    params_tree: dict,
    lr: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
):
    mask_full = _full_shape_mask(train_mask, params_tree)
    mask_pre  = _elementwise_mask_transform(mask_full)
    mask_post = _elementwise_mask_transform(mask_full)

    opt = optax.chain(
        mask_pre,                                    # zero masked grads
        optax.adam(learning_rate=lr, b1=b1, b2=b2, eps=eps),
        mask_post,                                   # zero masked updates
    )
    return opt, mask_full