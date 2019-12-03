# DOTA-DOAI

## Abstract
This repo is the codebase for our team to participate in DOTA related competitions, including rotation and horizontal detection. We mainly use [FPN](https://arxiv.org/abs/1612.03144)-based two-stage detector, and it is completed by [YangXue](https://github.com/yangxue0827) and [YangJirui](https://github.com/yangJirui).    


## Performance
### DOTA1.0 (Task1)
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | InLD | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| [FPN](https://arxiv.org/abs/1612.03144) (baseline) | ResNet50_v1 (600,800,1024)->800 | DOTA1.0 trainval | DOTA1.0 test | 69.35 | [model](https://drive.google.com/file/d/1QRxFAQ_Nqj3kqagc-NtMWKlxKjJnQFLo/view?usp=sharing) | No | 1x | No | 2X GeForce RTX 2080 Ti | 1 | cfgs_dota1.0_res50_v2.py |    
| FPN | ResNet50_v1 (600,800,1024)->800 | DOTA1.0 trainval | DOTA1.0 test | 70.87 | [model](https://drive.google.com/file/d/1mdvfNgIuagFQfddIV12yx9TWcRQ65YTf/view?usp=sharing) | **Yes** | 1x | No | 2X GeForce RTX 2080 Ti | 1 | cfgs_dota1.0_res50_v3.py |    

## Performance of published papers on DOTA datasets
### DOTA1.0 (Task1)
| Model | Backbone | mAP | Paper Link | Code Link | 
|:-----:|:--------:|:---:|:----------:|:---------:|
| FR-O (DOTA) | ResNet101 | 52.93 | [CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/html/Xia_DOTA_A_Large-Scale_CVPR_2018_paper.html) | [code](https://github.com/jessemelpolio/Faster_RCNN_for_DOTA) |
| IENet | ResNet101 | 57.14 | [arXiv:1912.00969](https://arxiv.org/abs/1912.00969) | - |
| R<sup>2</sup>CNN | ResNet101 | 60.67 | [arXiv:1706.09579](https://arxiv.org/abs/1706.09579) | [code](https://github.com/DetectionTeamUCAS/R2CNN_Tensorflow_Rotation) |
| RRPN | ResNet101 | 61.01 | [TMM](https://ieeexplore_ieee.xilesou.top/abstract/document/8323240)   [arXiv:1703.01086](https://arxiv.org/pdf/1703.01086.pdf) | [code](https://github.com/DetectionTeamUCAS/RRPN_Tensorflow_Rotation) |
| RetinaNet-H | ResNet101 | 64.73 | [arXiv:1908.05612](https://arxiv.org/abs/1908.05612) | [code](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) |
| ICN | ResNet101 | 68.16 | [ACCV2018](https://link.springer.com/chapter/10.1007/978-3-030-20893-6_10) | - |
| RoI Transformer | ResNet101 | 69.56 | [CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Learning_RoI_Transformer_for_Oriented_Object_Detection_in_Aerial_Images_CVPR_2019_paper.pdf) | [code](https://github.com/dingjiansw101/RoITransformer_DOTA) |
| CAD-Net | ResNet101 | 69.90 | [TGRS](https://ieeexplore.ieee.org/document/8804364/)  [arXiv:1903.00857](https://arxiv.org/abs/1903.00857) | - |
| SCRDet | ResNet101 | 72.61 | [ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_SCRDet_Towards_More_Robust_Detection_for_Small_Cluttered_and_Rotated_ICCV_2019_paper.pdf) | [code](https://github.com/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow) |
| FADet | ResNet101 | 73.28 | [ICIP2019](https://ieeexplore.ieee.org/abstract/document/8803521) | - |
| R<sup>3</sup>Det | ResNet152 | 73.74 | [arXiv:1908.05612](https://arxiv.org/abs/1908.05612) | [code](https://github.com/SJTU-Det/R3Det_Tensorflow) |
| RSDet | ResNet152 | 74.10 | [arXiv:1911.08299](https://arxiv.org/abs/1911.08299) | - |
| Gliding Vertex | ResNet101 | 75.02 | [arXiv:1911.09358](https://arxiv.org/abs/1911.09358) | - |
| APE | ResNeXt-101(32x4) | 75.75 | [arXiv:1906.09447](https://arxiv.xilesou.top/abs/1906.09447) | - |
| OWSR | Ensemble (ResNet101 +  ResNeXt101 + mdcn-ResNet101) | 76.36 | [CVPR2019 WorkShop](http://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Li_Learning_Object-Wise_Semantic_Representation_for_Detection_in_Remote_Sensing_Imagery_CVPRW_2019_paper.pdf) | - |
| FPN-InLD / R<sup>3</sup>Det-InLD (R<sup>3</sup>Det++) | ResNet101 / ResNet152 | 76.81 / 76.56 | - | [code](https://github.com/SJTU-Det/R3Det_Tensorflow) |

### DOTA1.0 (Task2)
| Model | Backbone | mAP | Paper Link | Code Link | 
|:-----:|:--------:|:---:|:----------:|:---------:|
| FR-H (DOTA) | ResNet101 | 60.46 | [CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/html/Xia_DOTA_A_Large-Scale_CVPR_2018_paper.html) | [code](https://github.com/jessemelpolio/Faster_RCNN_for_DOTA) |
| SBL | ResNet50 | 64.77 | [arXiv:1810.08103](https://arxiv.org/abs/1810.08103) | - |
| ICN | ResNet101 | 72.45 | [ACCV2018](https://link.springer.com/chapter/10.1007/978-3-030-20893-6_10) | - |
| IoU-Adaptive R-CNN | ResNet101 | 72.72 | [Remote Sensing](https://www.mdpi.com/2072-4292/11/3/286) | - |
| EFR | VGG16 | 73.49 | [Remote Sensing](https://www.mdpi.com/2072-4292/11/18/2095) | [code](https://github.com/pioneer2018/dtdm-di) |
| SCRDet | ResNet101 | 75.35 | [ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_SCRDet_Towards_More_Robust_Detection_for_Small_Cluttered_and_Rotated_ICCV_2019_paper.pdf) | [code](https://github.com/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow) |
| FADet | ResNet101 | 75.38 | [ICIP2019](https://ieeexplore.ieee.org/abstract/document/8803521) | - |
| A<sup>2</sup>RMNet | ResNet101 | 78.45 | [Remote Sensing](https://www.mdpi.com/2072-4292/11/13/1594) | - |
| OWSR | Ensemble (ResNet101 +  ResNeXt101 + mdcn-ResNet101) | 78.79 | [CVPR2019 WorkShop](http://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Li_Learning_Object-Wise_Semantic_Representation_for_Detection_in_Remote_Sensing_Imagery_CVPRW_2019_paper.pdf) | - |
| DM-FPN | ResNet-Based | 79.27 | [Remote Sensing](https://www.mdpi.com/2072-4292/11/7/755/) | - |

### DOTA1.5 (Task1)
| Model | Backbone | mAP | Paper Link | Code Link | 
|:-----:|:--------:|:---:|:----------:|:---------:|
| APE | ResNeXt-101(32x4) | 78.34 | [arXiv:1906.09447](https://arxiv.xilesou.top/abs/1906.09447) | - |
| OWSR | Ensemble (ResNet101 +  ResNeXt101 + mdcn-ResNet101) | 76.6 | [CVPR2019 WorkShop](http://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Li_Learning_Object-Wise_Semantic_Representation_for_Detection_in_Remote_Sensing_Imagery_CVPRW_2019_paper.pdf) | - |


### DOTA1.5 (Task2)
| Model | Backbone | mAP | Paper Link | Code Link | 
|:-----:|:--------:|:---:|:----------:|:---------:|
| OWSR | Ensemble (ResNet101 +  ResNeXt101 + mdcn-ResNet101) | 79.5 | [CVPR2019 WorkShop](http://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Li_Learning_Object-Wise_Semantic_Representation_for_Detection_in_Remote_Sensing_Imagery_CVPRW_2019_paper.pdf) | - |

### Related Articles
| Model | Paper Link | Code Link | Remark |
|:-----:|:----------:|:---------:|:------:|
| SSSDET | [ICIP2019](https://ieeexplore.ieee.org/abstract/document/8803262)  [arXiv:1909.00292](https://arxiv.org/abs/1909.00292) | - | Vehicle Detection |
| AVDNet | [GRSL](https://ieeexplore.ieee.org/abstract/document/8755462)  [arXiv:1907.07477](https://arxiv.org/abs/1907.07477) | - | Vehicle Detection |
| ClusDet | [ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Clustered_Object_Detection_in_Aerial_Images_ICCV_2019_paper.pdf) | - | Partial Categories |