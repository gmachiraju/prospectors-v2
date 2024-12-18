import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Embedding table
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        # Flatten input for matching with embedding table
        inputs_flat = inputs.view(-1, self.embedding_dim)

        # Compute distances between input and embedding vectors
        distances = torch.cdist(inputs_flat.unsqueeze(1), self.embeddings.weight.unsqueeze(0))

        # Find nearest embedding index for each input
        encoding_indices = torch.argmin(distances, dim=-1)
        quantized = self.embeddings(encoding_indices).view_as(inputs)

        # Compute commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Pass-through gradient
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices

class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, encoding_indices):
        return self.embeddings(encoding_indices)

class VQVAE1D(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VQVAE1D, self).__init__()
        self.embedding_dim = embedding_dim
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(num_embeddings, embedding_dim)

    def forward(self, inputs):
        quantized, vq_loss, encoding_indices = self.vector_quantizer(inputs)
        decoded = self.decoder(encoding_indices)
        return decoded, vq_loss, encoding_indices
    
    
# older architecture:
#=====================
class VQVAE(nn.Module):
    def __init__(self, precomputed_embeddings):
        super(VQVAE, self).__init__()
        self.encoder = ... # Define your encoder architecture
        self.decoder = ... # Define your decoder architecture
        self.embedding = nn.Embedding.from_pretrained(precomputed_embeddings, freeze=True)
        self.embedding_dim = precomputed_embeddings.shape[1]

    def quantize(self, z_e):
        distances = torch.cdist(z_e, self.embedding.weight)
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)
        return z_q, indices

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices = self.quantize(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, z_e, z_q, indices