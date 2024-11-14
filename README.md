# SVEA
## Introduction
Structural variations (SVs) are a pervasive and impactful class of genetic variation within the genome, significantly influencing gene function, impacting human health, and contributing to disease. Recent advances in deep learning have shown promise for SV detection; however, current methods still encounter key challenges in effective feature extraction and accurately predicting complex variations. To address these limitations, we introduce SVEA, an advanced deep learning model that incorporates a novel multi-channel image encoding approach specifically designed for precise SV prediction. This unique encoding method enhances feature representation by converting SVs into multi-dimensional image formats, which improves the model’s ability to capture subtle genomic variations. In addition, SVEA integrates multi-head self-attention mechanisms and multi-scale convolution modules, enabling it to capture global context and multi-scale features more effectively. Our results demonstrate that SVEA offers improved accuracy and generalizability in detecting complex SVs across diverse genomic regions, providing a robust and innovative tool to advance SV analysis in genetic research.
## Environment
1.numpy  
2.torch  
3.torchvision  
4.scikit-learn  
5.pillow  
6.pysam  

## Running
**1.encode_RGB.py**  
A script to encode alignment information into an RGB image. Requires a VCF file and a BAM file as input.  

**2.singledatasets_5fold.py**  

A script for single-dataset cross-validation. Requires a folder of images from a single dataset as input.  

**3.cross_train.py**  

A script for cross-dataset cross-validation. Requires folders of images from multiple datasets as input.  

**4.model_test.py**  

A script for comparative experiments. Requires folders of images from multiple datasets and model architectures as input.

## Contact
For advising, bug reporting and requiring help, please contact tx.qiu@siat.ac.cn.  
