# Instance-Level Feature Denoising

## Performance
### DOTA1.0 (Task1)
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | Tricks | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| [FPN](https://arxiv.org/abs/1612.03144) (baseline) | ResNet50_v1 (600,800,1024)->800 | DOTA1.0 trainval | DOTA1.0 test | 69.35 | [model](https://drive.google.com/file/d/1QRxFAQ_Nqj3kqagc-NtMWKlxKjJnQFLo/view?usp=sharing) | No | 1x | No | 2X GeForce RTX 2080 Ti | 1 | cfgs_dota1.0_res50_v2.py |    
| FPN | ResNet50_v1d (600,800,1024)->800 | DOTA1.0 trainval | DOTA1.0 test | 70.87 | [model](https://drive.google.com/file/d/1mdvfNgIuagFQfddIV12yx9TWcRQ65YTf/view?usp=sharing) | [**+InLD**]() | 1x | No | 2X GeForce RTX 2080 Ti | 1 | cfgs_dota1.0_res50_v3.py |     
| FPN | **ResNet152_v1d (600,800,1024)->MS** | DOTA1.0 trainval | DOTA1.0 test | 76.20 (76.54) | [model](https://drive.google.com/file/d/16lZEttBN3asEDP7lIryvH-PByjxm6A7K/view?usp=sharing) | **ALL** | **2x** | **Yes** | 2X GeForce RTX 2080 Ti | 1 | cfgs_dota1.0_res152_v1.py |     
  
