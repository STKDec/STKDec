# STKDec
  Battery swapping station (BSS) networks have experienced rapid expansion in recent years. A fundamental issue during the expansion process is the BSS deployment. Generally, deploying BSS based on the estimated battery swapping demands is a reliable way. However, for a new city without historical data, it is challenging to predict the demands before actual deployment. To this end, we propose STKDec, a knowledge-enhanced conditional diffusion model for cross-city battery swapping demand prediction. STKDec leverages historical data from the city with deployed BSS networks to predict battery swapping demands in the target city. Specifically, it first construct an urban knowledge graph (UKG) to align environmental representations and design a multi-relation-aware GCN to transfer inter-station relationship embeddings between source and target cities. Furthermore, an MLP network is employed to capture and model the users' battery-swapping behavior representations. We input all these obtained embeddings into a diffusion model to guide the denoising process. Finally, by feeding the extracted station embeddings of the target city into the trained model, we can predict the demand for the target city. Extensive experiments on real-world battery swapping datasets demonstrate the superiority and effectiveness of STKDec compared to state-of-the-art baselines.
  
This codebase contains PyTorch implementation of the paper:
Break “Chicken-Egg”: Cross-city Battery Swapping Demand prediction via Knowledge-guided Diffusion, 2024.

The codebase is implemented in Python 3.6.6. Required packages are:
numpy      1.15.1
pytorch    1.0.1

![image](https://github.com/user-attachments/assets/22a935ad-6d2f-4fb4-ac14-cbc93baf7efb)
