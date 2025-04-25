# Modified_Selective_Attention
This repository contains an unofficial implementation of the Selective Attention module proposed by Zhang et. al. in NeurIPS 2024 and a proposed modification of it.

Selective Attention attempts to improve upon vanilla attention by introducing token aware and position aware temperature scaling before attention score computation. The aim of the scaling is to combat attention dilution for longer context, decouple token semantics from sparsity of attention map and suppress noisy tokens.

We aim to build upon the proposed selective attention module. We aim to vectorize the temperatures so as to gain a fine grained control over the temperature scaling and have more interpretability in terms what role does each component of a query vector play in deciding the sparsity of attention maps. Although the authors mention in the original paper that vectorization of temperatures didn't result in significant gains, we aim to pursue this direction more innovatively by manually controling the scaling factors corresponding to each dimension keeping in mind the role that temperature scaling plays.

The implementation is still in progress.
