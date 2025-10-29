import torch


class FastLorentzRotation:
    def __init__(self, norm_scale, norm_bias, p, device):
        self.l1_scale = torch.tensor([144] * 1 + [144] * 4 + [576] * 4 + [144] * 10) / (2 * torch.pi)
        self.l1_scale = self.l1_scale.to(device)
        self.scale = torch.tensor(norm_scale[:, -1], device=device).float()
        self.bias = torch.tensor(norm_bias[:, -1], device=device).float()
        self.p = p
        self.device = device
        self.phi_indices = torch.arange(0, 19, 1) + 2

    def __call__(self, batch):        
        bool_mask = torch.rand(batch.shape[0], device=self.device)
        idx = bool_mask < self.p
        bool_mask[idx] = 1
        bool_mask[~idx] = 0
        original_phi = (batch[:, self.phi_indices] * self.scale + self.bias) / self.l1_scale
        rotation = (torch.rand(batch.shape[0], device=self.device) * 2 * torch.pi)[:, None]
        rotated_phi = (torch.remainder((original_phi + rotation), 2 * torch.pi)) * self.l1_scale
        batch[:, self.phi_indices] = (
            (bool_mask[:, None] * rotated_phi + (1 - bool_mask[:, None]) * original_phi).float() - self.bias
        ) / self.scale

        return batch


class FastObjectMask:
    def __init__(self, p, device):
        self.p = p
        self.device = device

    def __call__(self, batch):
        batch = batch.reshape((-1, 19, 3))
        mask = torch.rand((batch.shape[0], batch.shape[1]), device=self.device)
        idx = mask < self.p
        mask[idx] = 0
        mask[~idx] = 1
        batch = batch * mask[:, :, None]
        batch = batch.reshape((-1, 57))

        return batch


class FastFeatureBlur:
    def __init__(self, p, magnitude, strength, device):
        self.p = p  ## Number of event that will be blurred
        self.magnitude = magnitude
        self.strength = strength  ## Number of featurs that will be blurred
        self.device = device

    def __call__(self, batch):
        batch = batch.reshape((-1, 19, 3))

        mask_p = torch.rand((batch.shape[0], 1, 1), device=self.device)
        idx = mask_p < self.p
        mask_p[idx] = 1
        mask_p[~idx] = 0

        mask_strength = torch.rand_like(batch, device=self.device)
        idx = mask_strength < self.strength
        mask_strength[idx] = 1
        mask_strength[~idx] = 0

        mask_magnitude = torch.ones_like(batch) * self.magnitude

        mask_magnitude = mask_magnitude * mask_strength * mask_p

        blur = torch.rand_like(batch, device=self.device)

        batch = batch * (1 - mask_magnitude) + blur * mask_magnitude

        batch = batch.reshape((-1, 57))

        return batch
