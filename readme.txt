This file contains utilities of every code file. To run the code, there is no need of creating a separate environment or anything.

Selective_Attention_Module.py: Contains implementation of the selective attention module 

Modified_Selective_Attention_Module.py: Contains implementation of the modified selective attention module

ViT_SA.py: Contains implementation of ViT using SA module. To initialize a ViT, create an instance of the class ViTWithSelectiveAttention

ViT_MSA.py: Contains implementation of ViT using MSA module. To initialize a ViT, create an instance of the class ViTWithModifiedSelectiveAttention

Training_SA.py: Function to train SA based ViT. Call train_vit_cifar10 to intiate training 

Train_MSA.py: Function to train MSA based ViT. Call train_vit_cifar10 to intiate training

ViT_SA_Architecture.py: Contains architecture and optimization routine used in the project for SA based ViT

Architecture_MSA.py: Contains architecture and optimization routine used in the project for MSA based ViT

AM_Visualization_ViT_SA.py: Contains function to visualize attention maps of SA based ViT. Call visualize_cifar10_example to visualize an example 

AM_Visualization_ViT_MSA.py: Contains function to visualize attention maps of MSA based ViT. Call visualize_cifar10_example to visualize an example 

Temp_SA_visualization.py: Contains function to visualize temperature maps of SA based ViT. Call visualize_cifar10_example to visualize an example 

Temp_MSA_Visualization.py: Contains function to visualize temperature maps of MSA based ViT. Call visualize_cifar10_example to visualize an example 