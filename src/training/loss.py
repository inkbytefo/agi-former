## Developer: inkbytefo
## Modified: 2025-11-27

import jax
import jax.numpy as jnp

PAD_ID = -1

def _masked_ce(logits, targets):
    mask = (targets >= 0).astype(jnp.float32)
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    safe_targets = jnp.where(targets >= 0, targets, 0)
    gather = jnp.take_along_axis(logp, safe_targets[..., None], axis=-1).squeeze(-1)
    loss = -(mask * gather)
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(loss) / denom

def morph_loss(outputs, batch, lambda_root=1.0, lambda_suffix=1.0):
    root_logits = outputs["root"]
    suffix_logits = outputs["suffix"]
    B, N, S, Vs = suffix_logits.shape
    root_targets = batch[:, :, 0]
    loss_root = _masked_ce(root_logits, root_targets)
    loss_sfx = 0.0
    for s in range(S):
        s_targets = batch[:, :, 1 + s]
        s_logits = suffix_logits[:, :, s, :]
        loss_sfx = loss_sfx + _masked_ce(s_logits, s_targets)
    return lambda_root * loss_root + lambda_suffix * loss_sfx

def byte_loss(outputs, targets):
    B, N, P, V = outputs.shape
    logp = jax.nn.log_softmax(outputs, axis=-1)
    safe_targets = jnp.clip(targets, 0, V - 1)
    gather = jnp.take_along_axis(logp, safe_targets[..., None], axis=-1).squeeze(-1)
    return -jnp.mean(gather)


def kolmogorov_complexity_loss(model_params: dict, lambda_k: float = 0.01) -> float:
    """
    Kolmogorov karmaşıklığını yaklaştır: en kısa algoritmik tanımlama mesafesi
    Model parametrelerinin minimum description length'ını hesapla.
    """
    # Tüm parametrelerin balinyasyonunu birleştir (flatten)
    total_params = 0
    param_count = 0

    for key, p in model_params.items():
        if isinstance(p, jnp.ndarray):
            flattened = p.flatten()
            param_count += flattened.size
            total_params += jnp.sum(flattened ** 2)

    # Kompleksite proxy: normalized parameter entropy + compression ratio
    avg_param_mag = jnp.sqrt(total_params / param_count) if param_count > 0 else 0.0

    # Kolmogorov loss: Parametre büyüklüğü minimize + diversity bonus
    # Küçük değerler = sıkıştırılabilir = basit model (iyi)
    k_loss = avg_param_mag * lambda_k

    return k_loss
