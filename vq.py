import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class VQ(nn.Module):
    def __init__(self, latent_dim, num_tokens, dim=-1,
                 commitment=0.25):
        super(VQ, self).__init__()
        self.embedding = nn.Embedding(num_tokens, latent_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.dim = dim
        self.commitment = commitment

    def forward(self, x):
        dim = self.dim
        needs_transpose = dim != -1 or dim != x.dim() - 1
        if needs_transpose:
            x = x.transpose(-1, dim).contiguous()

        codes, indices = quantize(x, self.embedding.weight, self.commitment)

        if needs_transpose:
            codes = codes.transpose(-1, dim)
            indices = indices.transpose(-1, dim)

        self.last_indices = indices

        return codes


class VectorQuantization(Function):
    @staticmethod
    def compute_indices(inputs, codebook):
        with torch.no_grad():
            # inputs: NxD
            # codebook: KxD
            # Nx1xD
            inputs_horizontal = inputs.view(inputs.size(0), 1, inputs.size(1))
            # 1xKxD
            codebook_vertical = codebook.view(1, codebook.size(0), codebook.size(1))
            # NxKxD
            distances_matrix = (inputs_horizontal - codebook_vertical)**2
            # NxK
            distances_matrix = torch.sum(distances_matrix, dim=-1)
            # Nx1
            indices = torch.min(distances_matrix, dim=-1)[1].unsqueeze(1)
            return indices

    @staticmethod
    def flatten(x):
        code_dim = x.size(-1)
        return x.view(-1, code_dim)

    @staticmethod
    def restore_shapes(codes, indices, target_shape):
        idx_shape = list(target_shape)
        idx_shape[-1] = 1
        return codes.view(*target_shape), indices.view(*idx_shape)

    @staticmethod
    def forward(ctx, inputs, codebook, commitment=0.25):
        inputs_flat = VectorQuantization.flatten(inputs)
        indices = VectorQuantization.compute_indices(inputs_flat, codebook)
        codes = codebook[indices.view(-1), :]
        codes, indices = VectorQuantization.restore_shapes(
            codes, indices, inputs.shape)
        ctx.save_for_backward(codes, inputs, torch.FloatTensor([commitment]),
                              codebook, indices)
        ctx.mark_non_differentiable(indices)
        return codes, indices

    @staticmethod
    def backward(ctx, straight_through, unused_indices):
        codes, inputs, beta, codebook, indices = ctx.saved_tensors

        # TODO: figure out proper vq loss reduction
        vq_loss = F.mse_loss(inputs, codes).detach()

        # gradient of vq_loss
        diff = 2 * (inputs - codes) / inputs.numel()

        commitment = beta.item() * diff

        code_disp = VectorQuantization.flatten(-diff)
        indices = VectorQuantization.flatten(indices)
        code_disp = (torch
                     .zeros_like(codebook)
                     .index_add_(0, indices.view(-1), code_disp))
        return straight_through + commitment, code_disp, None

quantize = VectorQuantization.apply
