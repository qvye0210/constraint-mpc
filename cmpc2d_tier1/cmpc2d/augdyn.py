"""Wrap a learned model (with distractor dimensions) as an MPC dynamics function.

The distractor state is autonomous and does not affect the core dynamics, but it
IS a network input, so a model that predicts it badly can still degrade the core
prediction over a horizon.  That coupling is invisible to one-step offline
metrics and is exactly what closed-loop evaluation is for.

`z_mode` selects who propagates the distractors along the MPC horizon:
    'model'  the learned model's own prediction  (realistic; penalises arms that
             zero the distractor weight, e.g. 'prop' and 'mask')
    'true'   the true distractor dynamics        (optimistic upper bound)
Reporting both isolates that effect.
"""

import numpy as np
import torch

from .env import NX, Params, f_distract, f_nominal


class AugDyn:
    def __init__(self, model, z0, n_dist, params=Params, z_mode="model",
                 device="cpu"):
        self.model, self.n_dist, self.p = model, n_dist, params
        self.z_mode, self.device = z_mode, device
        self.z0 = np.asarray(z0, dtype=float).reshape(1, -1)
        self.z = self.z0.copy()
        model.eval()

    def reset(self):
        self.z = self.z0.copy()

    def set_z(self, z):
        self.z0 = np.asarray(z, dtype=float).reshape(1, -1)
        self.z = self.z0.copy()

    @torch.no_grad()
    def __call__(self, x, u):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        u = np.atleast_2d(np.asarray(u, dtype=float))
        B = len(x)
        base = f_nominal(x, u, self.p)
        if self.n_dist:
            zb = np.repeat(self.z, B, axis=0)
            inp = np.concatenate([x, zb], axis=1)
        else:
            inp = x
        out = self.model(torch.tensor(inp, dtype=torch.float32, device=self.device),
                         torch.tensor(u, dtype=torch.float32, device=self.device)
                         ).cpu().numpy().astype(float)
        if self.n_dist:
            self.z = (self.z + out[:1, NX:] if self.z_mode == "model"
                      else f_distract(self.z, self.p))
        return base + out[:, :NX]
