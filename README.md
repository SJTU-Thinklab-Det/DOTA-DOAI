# DOTA-DOAI

## Abstract
This repo is the codebase for our team to participate in DOTA related competitions, including rotation and horizontal detection. We mainly use [FPN](https://arxiv.org/abs/1612.03144)-based two-stage detector, and it is completed by [YangXue](https://yangxue0827.github.io/) and [YangJirui](https://github.com/yangJirui).      

**We also recommend a tensorflow-based [rotation detection benchmark](https://github.com/yangxue0827/RotationDetection), which is led by [YangXue](https://yangxue0827.github.io/).**

## Performance
### DOTA1.0 (Task1)
| Model |    Backbone   |    Training data    |    Val data    |    mAP   | Model Link | Tricks | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| FPN | **ResNet152_v1d (600,800,1024)->MS** | DOTA1.0 trainval | DOTA1.0 test | 78.99 | [model](https://drive.google.com/file/d/16lZEttBN3asEDP7lIryvH-PByjxm6A7K/view?usp=sharing) | **ALL** | **2x** | **Yes** | 2X GeForce RTX 2080 Ti | 1 | cfgs_dota1.0_res152_v1.py | 
### DOTA1.0 (Task2)
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | Tricks | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| FPN (memory consumption) | **ResNet152_v1d (600,800,1024)->MS** | DOTA1.0 trainval | DOTA1.0 test | 81.23 | [model](https://drive.google.com/file/d/1HaSU75llga_Em1O73Jp8ZwOoHPZtaVGj/view?usp=sharing) | **ALL** | **2x** | **Yes** | 2X Quadro RTX 8000 | 1 | cfgs_dota1.0_res152_v1.py |     


### Visualization
![1](demo.jpg)

## Performance of published papers on DOTA datasets
### DOTA1.0 (Task1)
| Model | Backbone | mAP | Paper Link | Code Link | Remark | Recommend |
|:-----:|:--------:|:---:|:----------:|:---------:|:---------:|:---------:|
| FR-O (DOTA) | ResNet101 | 52.93 | [CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/html/Xia_DOTA_A_Large-Scale_CVPR_2018_paper.html) | [MXNet](https://github.com/jessemelpolio/Faster_RCNN_for_DOTA) | DOTA dataset, baseline | :white_check_mark: |
| IENet | ResNet101 | 57.14 | [arXiv:1912.00969](https://arxiv.org/abs/1912.00969) | - | anchor free | |
| TOSO | ResNet101 | 57.52 | [ICASSP2020](https://ieeexplore.ieee.org/document/9053562) | - | geometric transformation | |
| PIoU Loss | DLA-34 | 60.5 | [ECCV2020](https://arxiv.org/abs/2007.09584) | [Pytorch](https://github.com/clobotics/piou) | IoU loss, anchor free | :white_check_mark: |
| R<sup>2</sup>CNN | ResNet101 | 60.67 | [arXiv:1706.09579](https://arxiv.org/abs/1706.09579) | [TF](https://github.com/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow) | scene text, multi-task, different pooled sizes, baseline | :white_check_mark: |
| RRPN | ResNet101 | 61.01 | [TMM](https://ieeexplore.ieee.org/document/8323240)   [arXiv:1703.01086](https://arxiv.org/pdf/1703.01086.pdf) | [TF](https://github.com/DetectionTeamUCAS/RRPN_Faster-RCNN_Tensorflow) | scene text, rotation proposals, baseline | :white_check_mark: |
| Axis Learning | ResNet101 | 65.98 | [Remote Sensing](https://www.mdpi.com/2072-4292/12/6/908) | - | single stage, anchor free | :white_check_mark: |
| MARNet | ResNet101 | 67.19 | [IJRS](https://www.tandfonline.com/doi/full/10.1080/01431161.2021.1910371) | - | based on scrdet |  |
| ICN | ResNet101 | 68.16 | [ACCV2018](https://link.springer.com/chapter/10.1007/978-3-030-20893-6_10) | - | image cascade, multi-scale | :white_check_mark: |
| GSDet | ResNet101 | 68.28 | [TIP](https://ieeexplore.ieee.org/document/9411691) | - | scale reasoning |  |
| RADet | ResNeXt101 | 69.09 | [Remote Sensing](https://www.mdpi.com/2072-4292/12/3/389) | - | enhanced FPN, mask rcnn | |
| KARNET | ResNet50 | 68.87 | [CISNRC 2020](https://www.dpi-proceedings.com/index.php/dtcse/article/view/35158) | - | attention denoising, anchor refining | |
| RoI Transformer | ResNet101 | 69.56 | [CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Learning_RoI_Transformer_for_Oriented_Object_Detection_in_Aerial_Images_CVPR_2019_paper.pdf) | [MXNet](https://github.com/dingjiansw101/RoITransformer_DOTA), [Pytorch](https://github.com/dingjiansw101/AerialDetection) | roi transformer | :white_check_mark: |
| CAD-Net | ResNet101 | 69.90 | [TGRS](https://ieeexplore.ieee.org/document/8804364/)  [arXiv:1903.00857](https://arxiv.org/abs/1903.00857) | - | attention |  |
| ProbIoU | ResNet50 | 70.04 | [arXiv:2106.06072](https://arxiv.org/abs/2106.06072) | [TF](https://github.com/ProbIOU) | gaussian bounding boxes, hellinger distance | :white_check_mark: |
| A<sup>2</sup>S-Det | ResNet101 | 70.64 | [Remote Sensing](https://www.mdpi.com/2072-4292/13/1/73/htm) | - | label assign |  |
| AOOD | ResNet101 | 71.18 | [Neural Computing and Applications](https://link.springer.com/article/10.1007/s00521-020-04893-9) | - | attention + [R-DFPN](https://www.mdpi.com/2072-4292/10/1/132) |  |
| CGP Box | ResNet18 | 71.35 | [IJRS](https://www.tandfonline.com/doi/full/10.1080/01431161.2021.1941389) | - | center-guide points |  |
| Cascade-FF | ResNet152 | 71.80 | [ICME2020](https://ieeexplore.ieee.org/abstract/document/9102807) | - | refined retinanet + feature fusion |  |
| SCPNet | Hourglass104 | 72,20 | [GRSL](https://ieeexplore.ieee.org/abstract/document/9439947) | - | corner points |  |
| P-RSDet | ResNet101 | 72.30 | [Access](https://ieeexplore.ieee.org/abstract/document/9272784/) | - | anchor free, polar coordinates | :white_check_mark: |
| BBAVectors | ResNet101 | 72.32| [WACV2021](https://arxiv.org/abs/2008.07043) | [Pytorch](https://github.com/yijingru/BBAVectors-Oriented-Object-Detection) |  keypoint based  | :white_check_mark: |
| ROPDet | ResNet101-DCN | 72.42 | [J REAL-TIME IMAGE PR](https://link.springer.com/article/10.1007/s11554-020-01013-7) | - | point set representation |  |
| SCRDet | ResNet101 | 72.61 | [ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_SCRDet_Towards_More_Robust_Detection_for_Small_Cluttered_and_Rotated_ICCV_2019_paper.pdf) | [TF: R<sup>2</sup>CNN++](https://github.com/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow), IoU-Smooth L1: [RetinaNet-based](https://github.com/SJTU-Thinklab-Det/R3Det_Tensorflow), [R<sup>3</sup>Det-based](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) | attention, angular boundary problem | :white_check_mark: |
| O<sup>2</sup>-DNet | Hourglass104 | 72.8 | [ISPRS](https://www.sciencedirect.com/science/article/pii/S0924271620302690), [arXiv:1912.10694](https://arxiv.org/abs/1912.10694) | - | centernet, anchor free | :white_check_mark: |
| HRPNet | HRNet-W48 | 72.83 | [GRSL](https://ieeexplore.ieee.org/abstract/document/9281309) | - | polar |  |
| SARD | ResNet101 | 72.95 | [Access](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8917630) | - | IoU-based weighted loss | |
| GLS-Net | ResNet101 | 72.96 | [Remote Sensing](https://www.mdpi.com/2072-4292/12/9/1435) | - | attention, saliency pyramid |  |
| ProjBB | ResNet101 | 73.03 | [Access](https://ieeexplore.ieee.org/abstract/document/9400416) | [code](https://github.com/tgis-top/TRD), [codebase](https://github.com/yangxue0827/RotationDetection) | new definition of bounding box |  |
| DRN | Hourglass104 | 73.23 | [CVPR(oral)](https://arxiv.org/abs/2005.09973) | [code](https://github.com/Anymake/DRN_CVPR2020) | centernet, feature selection module, dynamic refinement head, new dataset (SKU110K-R) | :white_check_mark: |
| FADet | ResNet101 | 73.28 | [ICIP2019](https://ieeexplore.ieee.org/abstract/document/8803521) | - | attention | |
| MFIAR-Net | ResNet152 | 73.49 | [Sensors](https://www.mdpi.com/1424-8220/20/6/1686/htm) | - | feature attention, enhanced FPN | |
| CFC-NET | ResNet101 | 73.50 | [arXiv:2101.06849](https://arxiv.org/abs/2101.06849) | [Pytorch](https://github.com/ming71/CFC-Net) | critical feature, label assign, refine | :white_check_mark: |
| R<sup>3</sup>Det | ResNet152 | 73.74 | [AAAI2021](https://arxiv.org/abs/1908.05612) | [TF](https://github.com/SJTU-Thinklab-Det/R3Det_Tensorflow), [Pytorch](https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection) | refined single stage, feature alignment | :white_check_mark: |
| RSDet | ResNet152 | 74.10 | [AAAI2021](https://arxiv.org/abs/1911.08299) | [TF](https://github.com/Mrqianduoduo/RSDet-8P-4R) | quadrilateral bbox, angular boundary problem | :white_check_mark: |
| SegmRDet | ResNet50 | 74.14 | [Neurocomputing](https://link.springer.com/article/10.1007/s10489-021-02570-5) | - | anchor free, mask guided, refine feature |  |
| MEAD | ResNet101 | 74.80 | [Applied Intelligence](https://www.sciencedirect.com/science/article/pii/S0925231220300837) | - | segmentation-baed, new training and inference mechanism |  |
| Gliding Vertex | ResNet101 | 75.02 | [TPAMI](https://ieeexplore.ieee.org/document/9001201)  [arXiv:1911.09358](https://arxiv.org/abs/1911.09358) | [Pytorch](https://github.com/MingtaoFu/gliding_vertex) | quadrilateral bbox | :white_check_mark: |
| OSSDet | ResNeXt-10 | 75.08 | [JSTARS](https://ieeexplore.ieee.org/document/9524549) | - | feature enhancement and alignment | |
| EFN | U-Net | 75.27 | [Preprints](https://search.proquest.com/docview/2442440949?pq-origsite=gscholar&fromopenview=true) | - | Field-based | :white_check_mark: |
| SAR | ResNet152 | 75.26 | [Access](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9256343) | - | boundary problem | :white_check_mark: |
| TricubeNet | Hourglass104 | 75.26 | [arXiv:2104.11435](https://arxiv.org/abs/2104.11435) | [code](https://github.com/qjadud1994/TricubeNet) | 2D tricube kernel  | :white_check_mark: |
| Mask OBB | ResNeXt-101 | 75.33 | [Remote Sensing](https://www.mdpi.com/2072-4292/11/24/2930/htm) | - | attention, multi-task | :white_check_mark: |
| - | DarkNet | 75.5 | [TGRS](https://arxiv.org/abs/2104.11854) | - | angle classification | |
| TS<sup>4</sup>Net | ResNet101 | 75.63 | [arXiv:2108.03116](https://arxiv.org/abs/2108.03116) | - | label assign  |  |
| FFA | ResNet101 | 75.7 | [ISPRS](https://www.sciencedirect.com/science/article/abs/pii/S0924271620300319) | - | enhanced FPN, rotation proposals | |
| CBDA-Net | DLA-34-DCN | 75.74 | [TGRS](https://ieeexplore.ieee.org/document/9400469) | - | dual attention | |     
| APE | ResNeXt-101(32x4) | 75.75 | [TGRS](https://ieeexplore.ieee.org/abstract/document/9057525) [arXiv:1906.09447](https://arxiv.org/abs/1906.09447) | - | adaptive period embedding, length independent IoU (LIIoU)| :white_check_mark: |     
| R<sup>4</sup>Det | ResNet152 | 75.54 | [Image Vis Comput](https://www.sciencedirect.com/science/article/pii/S0262885620301682) | - | feature recursion and refinement |  |     
| RIE | HRGANet-W48 | 75.94 | [Remote Sensing](https://www.mdpi.com/2072-4292/13/18/3622/htm) | - | center-based rotated inscribed ellipse |  |     
| F<sup>3</sup>-Net | ResNet152 | 76.02| [Remote Sensing](https://www.mdpi.com/2072-4292/12/24/4027) | - | feature fusion and filtration |  |
| CenterMap OBB | ResNet101 | 76.03| [TGRS](https://ieeexplore.ieee.org/abstract/document/9151222) | - | center-probability-map |  |
| CSL | ResNet152 | 76.17 | [ECCV2020](http://arxiv.org/abs/2003.05597) | [TF: CSL_RetinaNet](https://github.com/Thinklab-SJTU/CSL_RetinaNet_Tensorflow), [Pytorch: YOLOv5_DOTA_OBB (CSL)](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB) | angular boundary problem | :white_check_mark: |
| MRDet | ResNet101 | 76.24 | [arXiv:2012.13135](http://arxiv.org/abs/2012.13135) | - | arbitrary-oriented rpn, multiple subtasks |  |
| AFC-Net | ResNet101 | 76.27 | [Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231221005294?casa_token=Olsm-GqNOjYAAAAA:-fzJkIhsE0PLwwZNXCiv6K7it1FQ5PU_cwdXHcXxK0gRpkZs0XsN-DTBax4DoW9jx2u0-gyuslc) | - | adaptive feature concatenate |  |
| OWSR | Ensemble (ResNet101 +  ResNeXt101 + mdcn-ResNet101) | 76.36 | [CVPR2019 WorkShop](http://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Li_Learning_Object-Wise_Semantic_Representation_for_Detection_in_Remote_Sensing_Imagery_CVPRW_2019_paper.pdf)   [TGRS](https://ieeexplore.ieee.org/abstract/document/8960460) | - | enhanced FPN | |
| SLA | ResNet50 | 76.36 | [Remote Sensing](https://www.mdpi.com/2072-4292/13/14/2664/htm) | [Pytorch](https://github.com/ming71/SLA) | sparse label assignment | :white_check_mark: |
| OPLD | ResNet101 | 76.43 | [J-STARS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9252176) | [Pytorch](https://github.com/yf19970118/OPLD-Pytorch) | boundary problem, point-guided | :white_check_mark: |
| R<sup>3</sup>Det++ | ResNet152 | 76.56 | [arXiv:2004.13316](https://arxiv.org/abs/2004.13316) | [TF](https://github.com/SJTU-Thinklab-Det/R3Det_Tensorflow) | refined single stage, feature alignment, denoising | :white_check_mark: |
| PolarDet | ResNet101 | 76.64 | [IJRS](https://www.tandfonline.com/doi/epub/10.1080/01431161.2021.1931535?needAccess=true) [arXiv:2010.08720](https://arxiv.org/abs/2010.08720) | - | polar,  center-semantic | :white_check_mark: |
| Beyond Bounding-Box | ResNet152 | 76.67 | [CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Beyond_Bounding-Box_Convex-Hull_Feature_Adaptation_for_Oriented_and_Densely_Packed_CVPR_2021_paper.pdf) | [Pytorch](https://github.com/sdl-guozonghao/beyondboundingbox) | point-based, reppoints | :white_check_mark: |
| SCRDet++ | ResNet101 | 76.81 | [arXiv:2004.13316](https://arxiv.org/abs/2004.13316) | [TF](https://github.com/SJTU-Thinklab-Det/DOTA-DOAI) | angular boundary problem, denoising | :white_check_mark: |
| DAFNe | ResNet101 | 76.95 | [arXiv:2109.06148](https://arxiv.org/abs/2109.06148) | [Pytorch](https://github.com/steven-lang/DAFNe) | r-fcos |  |
| DAL+S<sup>2</sup>A-Net | ResNet50 | 76.95 | [AAAI2021](https://arxiv.org/abs/2012.04150) | [Pytorch](https://github.com/ming71/DAL) | label assign | :white_check_mark: |
| DCL | ResNet152 | 77.37 | [CVPR2021](https://arxiv.org/abs/2011.09670) | [TF](https://github.com/Thinklab-SJTU/DCL_RetinaNet_Tensorflow) | boundary problem | :white_check_mark: |
| MSFF | ResNet50 | 77.46 | [JSTARS](https://ieeexplore.ieee.org/abstract/document/9444845) | - | rotation invariance features | |
| RIDet | ResNet50 | 77.62| [arXiv:2103.11636](https://arxiv.org/abs/2103.11636) | [Pytorch](https://github.com/ming71/RIDet), [TF](https://github.com/yangxue0827/RotationDetection) | quad., representation ambiguity  | :white_check_mark: |
| RDD | ResNet101 | 77.75 | [Remote Sensing](https://www.mdpi.com/2072-4292/12/19/3262/htm) | [Pytorch](https://github.com/Capino512/pytorch-rotation-decoupled-detector) | rotation-decoupled |  |
| OSKDet | ResNet101 | 77.81 | [arXiv:2104.08697](https://arxiv.org/abs/2104.08697) | - | keypoint localization (very similar to FR-Est) |  |
| CG-Net | ResNet101 | 77.89 | [arXiv:2103.11399](https://arxiv.org/abs/2103.11399) | [Pytorch](https://github.com/WeiZongqi/CG-Net) | attention |  |
| Oriented RepPoints | ResNet101 | 78.12 | [arXiv:2105.11111](https://arxiv.org/abs/2105.11111) | [Pytorch](https://github.com/LiWentomng/OrientedRepPoints) | point-based, reppoints | :white_check_mark: |
| FoRDet | VGG16 | 78.13 | [TGRS](https://ieeexplore.ieee.org/abstract/document/9535140) | - | refinenet |  |
| AProNet | VGG16 | 78.16 | [ISPRS](https://www.sciencedirect.com/science/article/pii/S092427162100229X) | - | axis projection-based angle learning, feature enhancement |  |
| FR-Est | ResNet101-DCN | 78.49| [TGRS](https://ieeexplore.ieee.org/abstract/document/9194345) | - | point-based estimator | :white_check_mark: |
| S<sup>2</sup>A-Net | ResNet50/ResNet101 | 79.42/79.15 | [TGRS](https://ieeexplore.ieee.org/document/9377550) | [Pytorch](https://github.com/csuhan/s2anet) | refined single stage, feature alignment | :white_check_mark: |
| O<sup>2</sup>DETR | ResNet50 | 79.66| [arXiv:2106.03146](https://arxiv.org/abs/2106.03146) | - | deformable detr, transformer | :white_check_mark: |
| ROSD | ResNet101 | 79.76 | [Access](https://ieeexplore.ieee.org/abstract/document/9419068) | - | refined single stage, feature alignment | |
| SARA | ResNet50/ResNet101 | 79.91/79.13 | [Remote Sensing](https://www.mdpi.com/2072-4292/13/7/1318/htm) | - | self-adaptive aspect ratio anchor, refine |  |
| ADT-Det | ResNet152 | 79.95| [Remote Sensing](https://www.mdpi.com/2072-4292/13/13/2623/htm) | - | feature pyramid transformer, feature refineent |  |
| ReDet | ReR50-ReFPN | 80.10 | [CVPR2021](https://arxiv.org/abs/2103.07733) | [Pytorch](https://github.com/csuhan/ReDet) |  rotation-equivariant, rotation-invariant roI align | :white_check_mark: |
| GWD | ResNet152 | 80.23 | [ICML2021](https://arxiv.org/abs/2101.11952) | [TF](https://github.com/yangxue0827/RotationDetection), [Pytorch code (YOLOv5-GWD)](https://github.com/zhanggefan/rotmmdet) | boundary discontinuity, square-like problem, gaussian wasserstein distance loss | :white_check_mark: |
| KLD | ResNet152 | 80.63 | [arXiv:2106.01883](https://arxiv.org/abs/2106.01883) | [TF](https://github.com/yangxue0827/RotationDetection), [Pytorch code (YOLOv5-KLD)](https://github.com/zhanggefan/rotmmdet) | Kullback-Leibler divergence, high-precision, scale invariance | :white_check_mark: |
| Oriented R-CNN | ResNet50/ResNet101 | 80.87/80.52 | [ICCV2021](https://arxiv.org/abs/2108.05699) | [Pytorch](https://github.com/jbwang1997/OBBDetection) | Rotation FPN, Gliding Vertex |  |

### DOTA1.0 (Task2)
| Model | Backbone | mAP | Paper Link | Code Link | Remark | Recommend |
|:-----:|:--------:|:---:|:----------:|:---------:|:---------:|:---------:|
| FR-H (DOTA) | ResNet101 | 60.46 | [CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/html/Xia_DOTA_A_Large-Scale_CVPR_2018_paper.html) | [MXNet](https://github.com/jessemelpolio/Faster_RCNN_for_DOTA) | DOTA dataset, baseline | :white_check_mark: |
| Deep Active Learning | ResNet18 | 64.26 | [arXiv:2003.08793](https://arxiv.org/abs/2003.08793) | - | CenterNet, Deep Active Learning | :white_check_mark: |
| SBL | ResNet50 | 64.77 | [arXiv:1810.08103](https://arxiv.org/abs/1810.08103) | - | single stage |
| CenterFPANet | ResNet18 | 65.29 | [HPCCT & BDAI 2020](https://dl.acm.org/doi/abs/10.1145/3409501.3409545)  [arXiv:2009.03063](https://arxiv.org/abs/2009.03063) | - | light-weight |
| MARNet | ResNet101 | 71.73 | [IJRS](https://www.tandfonline.com/doi/full/10.1080/01431161.2021.1910371) | - | based on scrdet |  |
| FMSSD | VGG16 | 72.43 | [TGRS](https://ieeexplore.ieee.org/abstract/document/8930933) | - | IoU-based weighted loss, enhanced FPN |  |
| ICN | ResNet101 | 72.45 | [ACCV2018](https://link.springer.com/chapter/10.1007/978-3-030-20893-6_10) | - | image cascade, multi-scale | :white_check_mark: |
| IoU-Adaptive R-CNN | ResNet101 | 72.72 | [Remote Sensing](https://www.mdpi.com/2072-4292/11/3/286) | - | IoU-based weighted loss, cascade| |
| EFR | VGG16 | 73.49 | [Remote Sensing](https://www.mdpi.com/2072-4292/11/18/2095) | [Pytorch](https://github.com/pioneer2018/dtdm-di) | enhanced FPN | |
| AF-EMS | ResNet101 | 73.97 | [Remote Sensing](https://www.mdpi.com/2072-4292/13/2/160) | - | scale-aware feature, anchor free | |
| SCRDet | ResNet101 | 75.35 | [ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_SCRDet_Towards_More_Robust_Detection_for_Small_Cluttered_and_Rotated_ICCV_2019_paper.pdf) | [TF](https://github.com/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow) | attention, angular boundary problem | :white_check_mark: |
| FADet | ResNet101 | 75.38 | [ICIP2019](https://ieeexplore.ieee.org/abstract/document/8803521) | - | attention | |
| MFIAR-Net | ResNet152 | 76.07 | [Sensors](https://www.mdpi.com/1424-8220/20/6/1686/htm) | - | feature attention, enhanced FPN | |
| F<sup>3</sup>-Net | ResNet152 | 76.48| [Remote Sensing](https://www.mdpi.com/2072-4292/12/24/4027) | - | feature fusion and filtration |  |
| Mask OBB | ResNeXt-101 | 76.98 | [Remote Sensing](https://www.mdpi.com/2072-4292/11/24/2930/htm) | - | attention, multi-task | :white_check_mark: |
| CenterMap OBB | ResNet101 | 77.33| [TGRS](https://ieeexplore.ieee.org/abstract/document/9151222) | - | center-probability-map |  |
| ASSD | VGG16 | 77.8| [TGRS](https://ieeexplore.ieee.org/abstract/document/9467647) | - | feature aligned |  |
| AFC-Net | ResNet101 | 78.06 | [Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231221005294?casa_token=Olsm-GqNOjYAAAAA:-fzJkIhsE0PLwwZNXCiv6K7it1FQ5PU_cwdXHcXxK0gRpkZs0XsN-DTBax4DoW9jx2u0-gyuslc) | - | adaptive feature concatenate |  |
| CG-Net | ResNet101 | 78.26 | [arXiv:2103.11399](https://arxiv.org/abs/2103.11399) | [Pytorch](https://github.com/WeiZongqi/CG-Net) | attention |  |
| OPLD | ResNet101 | 78.35 | [J-STARS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9252176) | [Pytorch](https://github.com/yf19970118/OPLD-Pytorch) | boundary problem, point-guided | :white_check_mark: |
| A<sup>2</sup>RMNet | ResNet101 | 78.45 | [Remote Sensing](https://www.mdpi.com/2072-4292/11/13/1594) | - | attention, enhanced FPN, different pooled sizes | |
| OWSR | Ensemble (ResNet101 +  ResNeXt101 + mdcn-ResNet101) | 78.79 | [CVPR2019 WorkShop](http://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Li_Learning_Object-Wise_Semantic_Representation_for_Detection_in_Remote_Sensing_Imagery_CVPRW_2019_paper.pdf)   [TGRS](https://ieeexplore.ieee.org/abstract/document/8960460) | - | enhanced FPN | |
| Parallel Cascade R-CNN |ResNeXt-101 | 78.96 | [Journal of Physics: Conference Series](https://iopscience.iop.org/article/10.1088/1742-6596/1544/1/012124/meta) | - | cascade rcnn | |
| DM-FPN | ResNet-Based | 79.27 | [Remote Sensing](https://www.mdpi.com/2072-4292/11/7/755/) | - | enhanced FPN | |
| SCRDet++ | ResNet101 | 79.35 | [arXiv:2004.13316](https://arxiv.org/abs/2004.13316) | [TF](https://github.com/SJTU-Thinklab-Det/DOTA-DOAI) | denoising | :white_check_mark: |

### DOTA1.5 (Task1)
| Model | Backbone | mAP | Paper Link | Code Link | Remark | Recommend |
|:-----:|:--------:|:---:|:----------:|:---------:|:---------:|:---------:|
| APE | ResNeXt-101(32x4) | 78.34 | [TGRS](https://ieeexplore.ieee.org/abstract/document/9057525) [arXiv:1906.09447](https://arxiv.org/abs/1906.09447) | - | length independent IoU (LIIoU)| :white_check_mark: |
| OWSR | Ensemble (ResNet101 +  ResNeXt101 + mdcn-ResNet101) | 76.60 | [TGRS](https://ieeexplore.ieee.org/abstract/document/8960460) [CVPR2019 WorkShop](http://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Li_Learning_Object-Wise_Semantic_Representation_for_Detection_in_Remote_Sensing_Imagery_CVPRW_2019_paper.pdf) | - | enhanced FPN | |
| ReDet | ReR50-ReFPN | 76.80 | [CVPR2021](https://arxiv.org/abs/2103.07733) | [Pytorch](https://github.com/csuhan/ReDet) |  rotation-equivariant, rotation-invariant RoI Align, | :white_check_mark: |
| DAFNe | ResNet101 | 71.99 | [arXiv:2109.06148](https://arxiv.org/abs/2109.06148) | [Pytorch](https://github.com/steven-lang/DAFNe) | r-fcos |  |

### DOTA1.5 (Task2)
| Model | Backbone | mAP | Paper Link | Code Link | Remark | Recommend | 
|:-----:|:--------:|:---:|:----------:|:---------:|:---------:|:---------:|
| CDD-Net | ResNet101 | 61.3 | [GRSL](https://ieeexplore.ieee.org/abstract/document/9302742) | - | attention | |
| ReDet | ReR50-ReFPN | 78.08 | [CVPR2021](https://arxiv.org/abs/2103.07733) | [Pytorch](https://github.com/csuhan/ReDet) |  rotation-equivariant, rotation-invariant RoI Align, | :white_check_mark: |
| OWSR | Ensemble (ResNet101 +  ResNeXt101 + mdcn-ResNet101) | 79.50 | [TGRS](https://ieeexplore.ieee.org/abstract/document/8960460) [CVPR2019 WorkShop](http://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Li_Learning_Object-Wise_Semantic_Representation_for_Detection_in_Remote_Sensing_Imagery_CVPRW_2019_paper.pdf) | - | enhanced FPN | |

### Related Articles
| Model | Paper Link | Code Link | Remark | Recommend | 
|:-----:|:----------:|:---------:|:------:| :------: | 
| SSSDET | [ICIP2019](https://ieeexplore.ieee.org/abstract/document/8803262)  [arXiv:1909.00292](https://arxiv.org/abs/1909.00292) | - | vehicle detection, lightweight | |
| AVDNet | [GRSL](https://ieeexplore.ieee.org/abstract/document/8755462)  [arXiv:1907.07477](https://arxiv.org/abs/1907.07477) | - | vehicle detection, small object | |
| ClusDet | [ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Clustered_Object_Detection_in_Aerial_Images_ICCV_2019_paper.pdf) | [Caffe2](https://github.com/fyangneil/Clustered-Object-Detection-in-Aerial-Image) | object cluster regions | :white_check_mark: |
| DMNet | [CVPR2020 WorkShop](https://arxiv.org/abs/2004.05520) | - | object cluster regions | :white_check_mark: |
| AdaZoom | [arXiv:2106.10409](https://arxiv.org/abs/2106.10409) | - | object cluster regions, reinforcement learning | :white_check_mark: |
| OIS | [arXiv:1911.07732](https://arxiv.org/abs/1911.07732) | [related Pytorch code](https://github.com/mrlooi/rotated_maskrcnn) | Oriented Instance Segmentation | :white_check_mark: |
| ISOP | [IGARSS2020](https://ieeexplore.ieee.org/abstract/document/9323274) | - | Oriented Instance Segmentation | |
| LR-RCNN | [arXiv:2005.14264 ](https://arxiv.org/abs/2005.14264) | - | vehicle detection | - |
| GRS-Det | [TGRS](https://ieeexplore.ieee.org/abstract/document/9186810) | - | ship detection, rotation fcos | - |
| DRBox | [arXiv:1711.09405](https://arxiv.org/abs/1711.09405) | [Caffe](https://github.com/liulei01/DRBox) | sar object detection | :white_check_mark: |
| DRBox-v2 | [TGRS](https://ieeexplore.ieee.org/document/8746781) | [TF](https://github.com/ZongxuPan/DrBox-v2-tensorflow) | sar object detection | - |
| RAPiD | [arXiv:2005.11623](https://arxiv.org/abs/2005.11623) | [Pytorch](https://github.com/duanzhiihao/RAPiD) | overhead fisheye images | - |
| OcSaFPN | [arXiv:2012.09859](https://arxiv.org/abs/2012.09859) | - | denoising | - |
| CR2A-Net | [TGRS](https://ieeexplore.ieee.org/abstract/document/9285168) | - | ship detection | - |
| - | [TGRS](https://ieeexplore.ieee.org/abstract/document/9301236) | - | knowledge distillation | :white_check_mark: |
| CHPDet | [arXiv:2101.11189](https://arxiv.org/abs/2101.11189) | - | new ship dataset | :white_check_mark: |

### Other Rotation Detection Codes
| Base Method |  Code Link | 
|:-----------:|:----------:| 
| RetinaNet | [RetinaNet_Tensorflow_Rotation](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) | 
| YOLOv3 | [rotate-yolov3-Pytorch](https://github.com/ming71/rotate-yolov3), [YOLOv3-quadrangle-Pytorch](https://github.com/JKBox/YOLOv3-quadrangle), [yolov3-polygon-Pytorch](https://github.com/ming71/yolov3-polygon) | 
| YOLOv4 | [rotate-yolov4-Pytorch](https://github.com/kkkunnnnethan/R-YOLOv4) | 
| YOLOv5 | [rotation-yolov5-Pytorch](https://github.com/BossZard/rotation-yolov5), [YOLOv5_DOTA_OBB (CSL)](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB) |
| CenterNet | [R-CenterNet-Pytorch](https://github.com/ZeroE04/R-CenterNet) |

## Dataset
Some remote sensing related object detection dataset statistics are in [DATASET.md](DATASET.md)
