import torch
from torch.utils.data import Dataset
import pandas as pd
import networkx as nx

import pdb

import huggingface_hub
from transformers import AutoModelForCausalLM, AutoTokenizer
huggingface_hub.login(token="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")
model_id = "EvolutionaryScale/esm3-sm-open-v1" 

import esm
from esm.models.esm3 import ESM3
from esm.sdk import client
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig, GenerationConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation

from esm.pretrained import (
    ESM3_function_decoder_v0,
    ESM3_sm_open_v0,
    ESM3_structure_decoder_v0,
    ESM3_structure_encoder_v0,
) 

# adapted from tutorials:
# - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

def retrieve_encoder():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    client = ESM3.from_pretrained(ESM3_OPEN_SMALL).to(device)
    return client


def get_embeddings(client, seq):
    sequence = seq.upper()
    protein = ESMProtein(sequence=sequence)
    with torch.no_grad():
        protein_tensor = client.encode(protein)
        output = client.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
    return output.per_residue_embedding.detach().cpu().numpy()


class ToEmbeddings(object):
    """
    Convert tokens in sample to Tensors. This transformation uses ESM3-open embeddings.
    """
    def __call__(self, seq):
        client = retrieve_encoder()
        get_embeddings(client, seq)


class ToPathGraph(object):
    """
    Convert list of Tensors into an nx linear/path graph.
    """
    def __call__(self, embeds):
        """
        Convert embedded text (list of embeddings) to map graph
        """
        G = nx.Graph(origin=(None, None))
        l,d = embeds.shape[0], embeds.shape[1]
        G.graph.update({'array_size': (l)})
        G.graph.update({'d': d})

        origin_flag = False
        for i in range(l):
            if origin_flag == False:
                G.graph.update({'origin': (i)})
                origin_flag = True
            embed = embeds[i,:].reshape(1, -1).astype('double')
            G.add_node(i, pos=i, emb=embed)
            
        nodelist = list(G.nodes())
        for node in nodelist:
            i = G.nodes[node]["pos"] # same as "node"
            for di in range(i-2, i+3):
                if di == i:
                    continue
                if di < 0 or di >= l:
                    continue
                G.add_edge(node, di, weight=1)
        return G


class ProteinSequenceDataset(Dataset):
    """
    Dataset class that stores sequences
    Iterates over sequences to generate batches
    Can iterate over sequences or map graphs of embeddings
    """
    def __init__(self, csv_file, device="cpu", transform=None):
        """
        Arguments:
            csv_file (string): csv file all the data: <sequence, label>
            transform (callable, optional): Optional transform to be applied
                on a sample
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = self.data.iloc[idx, 0]
        raw = seq + "" # useful copy for embedding dataloader        
        if self.transform:
            if type(self.transform) != list:
                self.transform = [self.transform]
            for t in self.transform:
                seq = t(seq)
                                
        label = self.data.iloc[idx, 1]
        return raw, seq, label