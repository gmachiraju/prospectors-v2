# prospectors-v2
Welcome to our repo! This work builds on the original repo for the prospector head, an interpreter model architecture that extracts and uses concepts from Foundation Model (FM) embeddings to identify class-relevant evidence. 

This repo focuses on LLMs and sequence/1D data applications. In addition to a streamlined codebase, this repo now supports:
- [x] Alternative concept definitions: batched spherical K-Means, VQ-VAEs   
- [x] Faster inference: GPU acceleration with `nx-cugraph`
- [ ] Faster inference: approximate inference with Good-Turing estimators 
- [ ] Support for new inductive biases: causal/forward bias, token connectivity as defined by part-of-speech tagging and coreference resolution (in natural language)


In addition to the v2 codebase for prospector heads, we also demonstrate the usage of these kernels for LLM steering, specifically with protein langauge models (PLMs).
