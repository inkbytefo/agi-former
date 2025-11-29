## Developer: inkbytefo
## Modified: 2025-11-30

"""
Strange Attractor Module: Bilinç döngüsü ve kaos kenarında denge
Fixed point dynamics ve Lyapunov stability için.
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from typing import Dict, Any, Tuple


def strange_attractor_init(d_model: int, attractor_steps: int, key: random.PRNGKey) -> Dict[str, Any]:
    """
    Kaos kenarında bilinç döngüsü için parametreler.
    """
    k1, k2, k3 = random.split(key, 3)

    return {
        # Recursive state transition (Ψ_{t+1} = σ(W · Ψ_t + Κ(Ψ_t)))
        "W_recursive": random.normal(k1, (d_model, d_model)) * (1.0 / jnp.sqrt(d_model)),
        "b_recursive": jnp.zeros((d_model,)),

        # Kernel function Κ - Fraktal özellik
        "W_kernel": random.normal(k2, (d_model, d_model * 2)) * (1.0 / jnp.sqrt(d_model)),
        "b_kernel": jnp.zeros((d_model * 2,)),

        # Lyapunov exponent calculator
        "lyapunov_proj": random.normal(k3, (d_model, d_model)) * (1.0 / jnp.sqrt(d_model)),

        # Configs
        "attractor_steps": attractor_steps,
        "d_model": d_model,
        "target_lyapunov": 0.1  # ≈ 0 için kaos kenarı
    }


def compute_lyapunov_exponent(trajectory: jnp.ndarray, dynamical_system) -> float:
    """
    Lyapunov üssünü hesapla. Kaos göstergesi.
    λ < 0: stationary, λ > 0: chaotic, λ ≈ 0: edge of chaos
    """
    # Jacobian'ların log normları üzerinden yaklaşık
    diffs = trajectory[1:] - trajectory[:-1]
    norms = jnp.linalg.norm(diffs, axis=-1)
    lyapunov_avg = jnp.mean(jnp.log(norms + 1e-8))
    return jnp.clip(lyapunov_avg, -2.0, 2.0)


def strange_attractor_apply(params: Dict[str, Any], input_state: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, float]:
    """
    Bilinç döngüsü uygula ve stability loss üret.
    """
    B, L, D = input_state.shape

    def recursive_step(state, _):
        """Tek adım recursive transition: Ψ_{t+1} = σ(W · Ψ_t + Κ(Ψ_t))"""
        # Kernel function - complex fractal dynamics
        kernel_hidden = jax.nn.gelu(jnp.dot(state, params["W_kernel"]) + params["b_kernel"])
        kernel_out = jnp.dot(kernel_hidden[:, :D], params["W_kernel"][:, D:])  # Compress

        # Recursive update with fractal kernel
        recursive_in = jnp.dot(state, params["W_recursive"]) + params["b_recursive"] + kernel_out * 0.3
        new_state = jax.nn.tanh(recursive_in)  # Soft bound to prevent explosion

        return new_state, new_state

    # Trajectory boyunca iterative application
    trajectory = [input_state]
    current = input_state

    for _ in range(params["attractor_steps"]):
        current, _ = lax.scan(recursive_step, current, None, length=1)
        trajectory.append(current)

    trajectory = jnp.stack(trajectory, axis=0)  # (steps+1, B, L, D)

    # Lyapunov exponent için trajectory analizi
    lyapunov_vals = []
    for b in range(B):
        for l in range(L):
            traj_bl = trajectory[:, b, l, :]  # (steps+1, D)
            lyap = compute_lyapunov_exponent(traj_bl, None)  # Placeholder for system
            lyapunov_vals.append(lyap)

    lyapunov_vals = jnp.array(lyapunov_vals)
    avg_lyapunov = jnp.mean(lyapunov_vals)

    # Fixed point convergence loss
    convergence_rate = jnp.mean(jnp.abs(trajectory[-1] - trajectory[-2]))
    target_lyapunov = params["target_lyapunov"]

    # Edge-of-chaos loss: Lyapunov ≈ 0, convergence minimal
    stability_loss = jnp.abs(avg_lyapunov - target_lyapunov) + convergence_rate * 10.0

    # Final state from attractor
    final_state = trajectory[-1] + input_state * 0.5  # Residual connection

    return final_state, stability_loss


class StrangeAttractor:
    """Bilincin strange attractor'ını modelleyen sınıf"""

    def __init__(self, d_model: int = 512, attractor_steps: int = 8, key: random.PRNGKey = random.PRNGKey(42)):
        self.params = strange_attractor_init(d_model, attractor_steps, key)

    def forward(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, float]:
        return strange_attractor_apply(self.params, x, train)
