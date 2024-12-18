# Residual Attention Network
Residual Attention Network for Image Classification (**CVPR-2017 Spotlight**)

Based on the research by Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Chen Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang


### Introduction
**Residual Attention Network** is a convolutional neural network using attention mechanism which can incorporate with state-of-the-art feed forward network architecture in an end-to-end training fashion.

**Residual Attention Networks** are described in the paper "Residual Attention Network for Image Classification"(https://arxiv.org/pdf/1704.06904.pdf).

This repository contains the prototxts of "Residual Attention Network". The trained model will be released soon. 

### Citation
If you find "Residual Attention Network" useful in your research, please cite:

	@article{wang2017residual,
  		title={Residual Attention Network for Image Classification},
  		author={Wang, Fei and Jiang, Mengqing and Qian, Chen and Yang, Shuo and Li, Cheng and Zhang, Honggang and Wang, Xiaogang and Tang, Xiaoou},
  		journal={arXiv preprint arXiv:1704.06904},
  		year={2017}	
	}

### Models
0. Attention-56 and Attention-92 are based on the pre-activation residual unit. 

1. According to the paper, we replace pre-activation residual unit with resnext unit to contruct the AttentionNeXt-56 and AttentionNeXt-92.



### Main Performance of original paper

0. Evaluation on ImageNet validation dataset.

|    Network       |Test Size|  top-1  |  top-5  |
|------------------|---------|---------|---------|
| Attention-56     | 224\*224|  21.76% |   5.9%  |
| AttentionNeXt-56 | 224\*224|  21.2%  |   5.6%  |
| Attention-92     | 320\*320|  19.5%  |   4.8%  |

### Analysis after the literature survey:

*The authors of the paper have scripted the original code using caffe. But in our project we approach the same concept using Tensorflow 2.0

*The datsets tested in the paper are CIFAR-10, CIFAR-100 and ImageNet based datasets.

*We have planned to have a focus on the Driver assistance related datasets.

*After our initial implementation of the Residual Attention network, we plan to find datsets specific to Driver assistance and try them out as well.

### Conceptual Understanding on the Paper after the Literature Survey:

A Residual Attention Network (RAN) is a sophisticated convolutional neural network (CNN) that combines two powerful components: attention mechanisms and residual units. Residual units, known for their skip connections, allow the network to bypass 2–3 layers with nonlinearities (e.g., ReLU activation and batch normalization), enabling more effective gradient flow and learning. The hallmark of RAN is its attention module, which dynamically focuses on the most relevant features as the network deepens.

### Key Components of the RAN Architecture

### Attention Modules:
RANs are constructed by stacking multiple Attention Modules, each of which produces attention-aware features. These features adapt as the layers go deeper, refining the network's ability to highlight important regions and patterns. This adaptability ensures that RANs capture complex hierarchical representations of the data.

### Attention Module Composition:
Each Attention Module consists of two complementary branches:

### Trunk Branch:
The trunk branch is responsible for core feature extraction and processing. It achieves this using residual units, which enhance learning efficiency by enabling the network to learn residual mappings instead of direct mappings.

### Mask Branch:
The mask branch refines the features by applying an attention weighting mechanism through a bottom-up and top-down process:

*Bottom-Up Step:
This step gathers global contextual information by downsampling the input using max-pooling. This operation ensures that the network captures holistic details about the input image.

*Top-Down Step:
After collecting global information, the top-down step upsamples the data (using interpolation) and combines it with the original feature maps. This ensures the refined features retain their original spatial resolution.

By focusing on both local and global information, the mask branch softens and adjusts the importance of different features, enhancing the trunk branch's effectiveness.

### Attention Residual Learning Formula:
The features from the trunk and mask branches are merged using a novel Attention Residual Learning formula. This technique optimally integrates the two branches, enabling the RAN to handle extremely deep architectures efficiently.

### Advantages of RANs
One of the key strengths of the RAN lies in its scalability. Unlike traditional networks, which often suffer from performance degradation as they grow deeper, RANs exhibit consistent performance improvement with additional Attention Modules. This is because each module captures distinct types of attention, enriching the network's ability to focus on diverse and complex data features.

### Summary of Understanding and Insights
The RAN architecture stands out because of its context-aware processing. By integrating attention with residual learning, the network selectively prioritizes the most important features while maintaining efficient learning dynamics. The two-branch structure, particularly the bottom-up and top-down operations in the mask branch, resembles mechanisms seen in human perception, where we process both broad context and fine details.

Additionally, the modular design of Attention Modules allows RANs to adapt to various tasks, from image classification to object recognition, making them versatile and robust for deep learning applications.


"# 4040" 
