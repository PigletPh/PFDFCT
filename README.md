# PFDFCT：For Efficient Image Super Resolution<sup>📌</sup>
<a href="https://github.com/Luckycat518"><img src="https://img.shields.io/badge/GitHub-@Luckycat518-000000.svg?logo=GitHub" alt="GitHub" target="_blank"></a>
<a href="https://charmve.github.io/computer-vision-in-action/" target="_blank"><img src="https://img.shields.io/badge/Computer Vision-000000.svg?logo=GitBook" alt="Computer Vision in Action"></a>
[![License](https://img.shields.io/github/license/Charmve/Surface-Defect-Detection)](LICENSE)

# Table of Contents

- [Introduction](#introduction)
- [Comparison with SOTA methods](#1-Comparison-with-SOTA-methods)

- [Notification](#notification)
- [Citation](#citation)


## Introduction


<p> Although the convolution neural network(CNN) and Transformer methods have greatly promoted the development of image super-resolution(SR), these two methods have their disadvantages. Making a trade-off between the two methods and effectively integrating their advantages can restore high-frequency information of images with fewer parameters and higher quality. Hence, in this study, a novel dual parallel fusion structure of distilled feature pyramid and serial CNN and Transformer(PFDFCT) model is proposed. In one branch, a lightweight serial structure of CNN and Transformer is implemented to guarantee the richness of the global features extracted by Transformer. In the other branch, an efficient distillation feature pyramid hybrid attention module is designed to efficiently purify the local features extracted by CNN and maintain integrity through cross-fusion. Such a multi-path parallel fusion strategy can ensure the richness and accuracy of features while avoiding the use of complex and long-range structures. The results show that the PFDFCT can reduce the mis-generated stripes and make the reconstructed image clearer for both easy-to-reconstruct and difficult-to-reconstruct targets compared to other advanced methods. Additionally, PFDFCT achieves a remarkable advance in model size and computational cost. Compared to the state-of-the-art(SOTA) model(i.e., efficient long-range attention network) in 2022, PFDFCT reduces parameters and floating point of operations(FLOPs) by more than 20% and 26% under all three scales, while maintaining a similar advanced reconstruction ability. The FLOPs of PFDFCT are as low as 31.8G, 55.3G, and 122.5G under scales of 2, 3 and 4, which are much lower than most current SOTA methods. </p>


## 1. Comparison with SOTA methods

<p> The proposed PFDFCT has great advantages in computational cost and image super-resolution restoration ability. The ability to avoid mis-generated strips and blur of PFDFCT is comparable to SOTA ELAN-light in 2022. However, the parameters and FLOPs decrease by 20.80% and 26.39% under scale of 4, respectively. Under ×2, ×3 and ×4 up-sampling, the model parameters are only 459K, 466K and 476K, respectively. Moreover, the FLOPs are as low as 31.8G, 55.3G and 122.5G, which are much lower than most current SOTA methods.</p>

![image](https://github.com/PigletPh/PFDFCT/blob/main/Cover_Image/Comparison_in_urban100.jpg)
<div align=center><img src="https://github.com/PigletPh/PFDFCT/blob/main/Cover_Image/Comparison_with_SOTA.jpg"></div>

👆 [<b>BACK to Table of Contents</b> -->](#table-of-contents)




## Notification
<b>The detailed description and independences will be gradually released. If you have any questions or requirements, you can email(email: wgq001@csu.edu.cn) me to get help before all files are released.</p>



👆 [<b>BACK to Table of Contents</b> -->](#table-of-contents)

## Citation
This paper is still under review......

👆 [<b>BACK to Table of Contents</b> -->](#table-of-contents)


