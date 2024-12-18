import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# from torch_kmeans import KMeans
import itertools

import utils
from data import ProteinSequenceDataset, ToEmbeddings
from architectures import VQVAE1D

#============================
# Prospector Head -- Layer 1
#============================
class QuantizerModel():
    """
    Model that allows for batch data loading from a dataset
    """
    def __init__(self, args):
        args = utils.dotdict(args)
        self.input_checks(args)
        
        self.dataloader = args.dataloader
        self.batch_size = args.batch_size
        self.method = args.method
        self.K = args.K
    
    def input_checks(self, args):
        """
        Checks that the input arguments are valid
        """
        if args.method not in ["VQVAE", "KMeans"]:
            raise ValueError("Quantizer method must be either VQVAE or KMeans")
        if int(args.K) < 1:
            raise ValueError("Please enter K > 0")
    
    def fit_vqvae(self, learn_rate=1e-3, epochs=100):
        optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)
        self.model.train()
        optimizer.zero_grad()
         
        # Training loop
        for epoch in range(epochs):
            for i_batch, x in enumerate(self.dataloader): # x is a batch of input embeddings           
                x = x.to(device=device)
                
                # Forward pass
                decoded, vq_loss, _ = self.model(x)
                decoded = decoded.view_as(x)
                assert decoded.shape == x.shape, f"Shape mismatch: {decoded.shape} vs {x.shape}"

                # Reconstruction loss (MSE)
                recon_loss = F.mse_loss(decoded, x)
                total_loss = recon_loss + vq_loss

                # Backward pass
                total_loss.backward()
                optimizer.step()

                # Print losses
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Recon Loss: {recon_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
                    
        print("Training complete.")
  
    def fit(self):
        """
        Fits the model by iterating through embedding ids and training and model
        """
        if self.method == "VQVAE":
            self.model = VQVAE1D(self.K, self.d)
            self.fit_vqvae()
        elif self.method == "KMeans":
            # self.model = KMeans(n_clusters=self.K) ## add normalization flag
            raise NotImplementedError("KMeans not yet implemented...Exiting")
            
#============================
# Prospector Head -- Layer 2
#============================
class SkipGramModel():
    def __init__(self, args):
        args = utils.dotdict(args)
        self.input_checks(args)
        
        self.n = args.n # level of skip-n-grams
        self.dataloader = args.dataloader
        self.quantizer = args.quantizer
        self.K = self.quantizer.K
        self.inductive_bias = args.inductive_bias
        self.r = args.r # receptive field
        
    def input_checks(self, args):
        """
        Checks that the input arguments are valid
        """
        if args.inductive_bias not in ["correlative", "causal"]:
            raise ValueError("Inductive bias must be either correlative or causal")
        if int(args.n) > 3:
            raise ValueError("Skip-n-grams only allow for n=1,2,3")
    
    def instantiate_kernel(self):
        """
        Make kernel dictionary
        """
        mono = list(range(self.K))
        bi = [x for x in itertools.combinations(mono, 2)] + [(x, x) for x in mono]
        kernel = dict.fromkeys(mono + bi, 0.0)
        if self.n > 2:
            tri = [x for x in itertools.combinations(mono, 3)] + [(x, x, x) for x in mono]
            kernel = dict.fromkeys(mono + bi + tri, 0.0)      
        return kernel
    
    def construct_sprite(self, G, key_in="emb", key_out="emb"):
        """
        Computes the frequency of each concept in each sample
        Note: default settings overwrite the embeddings with the concept labels
        """
        S = G.copy()
        S.graph.update({'d': 1})
        
        for node in S.nodes:
            embedding = S.nodes[node][key_in]
            if isinstance(embedding, np.ndarray):
                concept_label = self.quantizer.predict(embedding.reshape(1, -1).astype(float))[0]
            else: # if not using FM embedding (e.g. amino acid baseline)
                concept_label = self.quantizer.predict(embedding)
            S.nodes[node][key_out] = concept_label
        return S
    
    def sprite_embedding(self, G)
        pass
    

    def fit(self):
        pass
    
    def classify(self, x):
        pass