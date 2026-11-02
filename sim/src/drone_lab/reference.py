"""Smooth reference trajectories for the tracking task (torch, batched over environments).

Every trajectory is a sum of sinusoids per axis around an offset (all analytic, so the velocity is exact)::

    p(t) = offset + sum_k amp_k * sin(freq_k * t + phase_k)        (per axis, K terms)
    v(t) =          sum_k amp_k * freq_k * cos(freq_k * t + phase_k)

which covers circles, figure-eights (lemniscate of Gerono), Lissajous curves and smooth random curves.
Sizes and speeds are sampled per environment; the peak speed never exceeds the sampled speed.
"""

from __future__ import annotations

import math

import torch

SHAPES = ("circle", "figure8", "lissajous", "random")
"""Reference shapes, in the order used for shape ids."""


class ReferenceBank:
    """Per-environment sum-of-sinusoids trajectories (positions relative to the env origin)."""

    def __init__(self, num_envs: int, device: str | torch.device = "cpu", num_terms: int = 2):
        self.num_envs = num_envs
        self.device = device
        self.num_terms = num_terms
        self.amp = torch.zeros(num_envs, 3, num_terms, device=device)
        self.freq = torch.zeros(num_envs, 3, num_terms, device=device)
        self.phase = torch.zeros(num_envs, 3, num_terms, device=device)
        self.offset = torch.zeros(num_envs, 3, device=device)
        self.shape_id = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.speed = torch.zeros(num_envs, device=device)
        self.size = torch.zeros(num_envs, device=device)

    def position(self, t: torch.Tensor, env_ids: torch.Tensor | slice = slice(None)) -> torch.Tensor:
        """Reference position at times ``t`` (n,) for ``env_ids`` -> (n, 3)."""
        arg = self.freq[env_ids] * t.view(-1, 1, 1) + self.phase[env_ids]
        return self.offset[env_ids] + (self.amp[env_ids] * torch.sin(arg)).sum(dim=-1)

    def velocity(self, t: torch.Tensor, env_ids: torch.Tensor | slice = slice(None)) -> torch.Tensor:
        """Reference velocity at times ``t`` (n,) for ``env_ids`` -> (n, 3)."""
        arg = self.freq[env_ids] * t.view(-1, 1, 1) + self.phase[env_ids]
        return (self.amp[env_ids] * self.freq[env_ids] * torch.cos(arg)).sum(dim=-1)

    def sample(
        self,
        env_ids: torch.Tensor,
        *,
        shape_weights: dict[str, float] | None = None,
        size_range: tuple[float, float] = (0.8, 2.5),
        speed_range: tuple[float, float] = (0.3, 1.5),
        z_offset_range: tuple[float, float] = (-0.5, 0.5),
        z_amp_range: tuple[float, float] = (0.0, 0.5),
        fixed_shape: str | None = None,
        fixed_size: float | None = None,
        fixed_speed: float | None = None,
        fixed_z_amp: float | None = None,
    ) -> None:
        """Draw new trajectories for ``env_ids`` (fixed_* values make them deterministic, e.g. for evaluation)."""
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device

        def uniform(lo, hi, fixed=None):
            if fixed is not None:
                return torch.full((n,), float(fixed), device=dev)
            return lo + (hi - lo) * torch.rand(n, device=dev)

        if fixed_shape is not None:
            shape = torch.full((n,), SHAPES.index(fixed_shape), dtype=torch.long, device=dev)
        else:
            weights = shape_weights or {s: 1.0 for s in SHAPES}
            probs = torch.tensor([float(weights.get(s, 0.0)) for s in SHAPES], device=dev)
            shape = torch.multinomial(probs / probs.sum(), n, replacement=True)
        size = uniform(*size_range, fixed_size)
        speed = uniform(*speed_range, fixed_speed)
        deterministic = fixed_shape is not None
        # split the speed budget: the horizontal shape gets speed*cos(alpha), the vertical wave speed*sin(alpha),
        # so the 3D peak speed never exceeds `speed`
        if deterministic:
            alpha = torch.full((n,), math.radians(20.0) if (fixed_z_amp or 0.0) > 0 else 0.0, device=dev)
        else:
            alpha = math.radians(30.0) * torch.rand(n, device=dev)
        total_speed = speed
        speed = total_speed * torch.cos(alpha)
        phi = torch.zeros(n, device=dev) if deterministic else 2 * math.pi * torch.rand(n, device=dev)
        direction = (
            torch.ones(n, device=dev) if deterministic else torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
        )
        amp = torch.zeros(n, 3, self.num_terms, device=dev)
        freq = torch.zeros(n, 3, self.num_terms, device=dev)
        phase = torch.zeros(n, 3, self.num_terms, device=dev)

        # circle: x = r cos(w t + phi), y = dir * r sin(w t + phi); speed r * w
        m = shape == SHAPES.index("circle")
        if m.any():
            w = speed[m] / size[m]
            amp[m, 0, 0], freq[m, 0, 0], phase[m, 0, 0] = size[m], w, phi[m] + math.pi / 2
            amp[m, 1, 0], freq[m, 1, 0], phase[m, 1, 0] = size[m] * direction[m], w, phi[m]
        # figure-eight (Gerono): x = a sin(w t + phi), y = a/2 sin(2 (w t + phi)); peak speed a w sqrt(2)
        m = shape == SHAPES.index("figure8")
        if m.any():
            w = speed[m] / (size[m] * math.sqrt(2.0))
            amp[m, 0, 0], freq[m, 0, 0], phase[m, 0, 0] = size[m], w, phi[m]
            amp[m, 1, 0], freq[m, 1, 0], phase[m, 1, 0] = 0.5 * size[m] * direction[m], 2 * w, 2 * phi[m]
        # Lissajous: x = a sin(w t + phi), y = b sin(r w t + phi2), r in {1.5, 2, 3}
        m = shape == SHAPES.index("lissajous")
        if m.any():
            k = int(m.sum())
            ratios = torch.tensor([1.5, 2.0, 3.0], device=dev)
            r = (
                ratios[torch.zeros(k, dtype=torch.long, device=dev)]
                if deterministic
                else ratios[torch.randint(0, 3, (k,), device=dev)]
            )
            b = size[m] * (0.7 if deterministic else (0.5 + 0.5 * torch.rand(k, device=dev)))
            w = speed[m] / torch.sqrt(size[m] ** 2 + (b * r) ** 2)
            phi2 = torch.zeros(k, device=dev) if deterministic else 2 * math.pi * torch.rand(k, device=dev)
            amp[m, 0, 0], freq[m, 0, 0], phase[m, 0, 0] = size[m], w, phi[m]
            amp[m, 1, 0], freq[m, 1, 0], phase[m, 1, 0] = b, r * w, phi2
        # smooth random: two sinusoids per horizontal axis, time-scaled so the speed bound equals `speed`
        m = shape == SHAPES.index("random")
        if m.any():
            k = int(m.sum())
            g = torch.Generator(device=dev).manual_seed(0) if deterministic else None
            a = 0.5 * size[m].view(k, 1, 1) * (0.4 + 0.6 * torch.rand(k, 2, self.num_terms, device=dev, generator=g))
            f = 0.3 + 0.9 * torch.rand(k, 2, self.num_terms, device=dev, generator=g)
            p = 2 * math.pi * torch.rand(k, 2, self.num_terms, device=dev, generator=g)
            bound = torch.sqrt(((a * f).sum(dim=-1) ** 2).sum(dim=-1))  # >= peak horizontal speed
            f = f * (speed[m] / bound).view(k, 1, 1)
            amp[m, :2], freq[m, :2], phase[m, :2] = a, f, p
        # vertical: a slow oscillation on the last term (z uses no other term, so it never mixes with x/y)
        z_amp = uniform(*z_amp_range, fixed_z_amp)
        z_freq = (0.5 if deterministic else 0.3 + 0.4 * torch.rand(n, device=dev)) * total_speed / size
        z_amp = torch.minimum(z_amp, total_speed * torch.sin(alpha) / z_freq.clamp(min=1e-6))
        amp[:, 2, -1], freq[:, 2, -1] = z_amp, z_freq
        phase[:, 2, -1] = 0.0 if deterministic else 2 * math.pi * torch.rand(n, device=dev)
        offset = torch.zeros(n, 3, device=dev)
        offset[:, 2] = uniform(*z_offset_range, 0.0 if deterministic else None)

        self.amp[env_ids] = amp
        self.freq[env_ids] = freq
        self.phase[env_ids] = phase
        self.offset[env_ids] = offset
        self.shape_id[env_ids] = shape
        self.speed[env_ids] = total_speed
        self.size[env_ids] = size
