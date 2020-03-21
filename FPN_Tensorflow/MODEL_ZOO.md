# Towards More Robust Detection for Small, Cluttered and Rotated Objects via Instance-Level Feature Denoising

## Performance
### DOTA1.0 (Task2)
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | Tricks | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| [FPN](https://arxiv.org/abs/1612.03144) (baseline) | ResNet50_v1 (600,800,1024)->800 | DOTA1.0 trainval | DOTA1.0 test | 76.03 | [model](https://drive.google.com/file/d/1aT_wvoV9ao2COZHOe6OfJglBbnFa8qqa/view?usp=sharing) | No | 1x | No | 2X Quadro RTX 8000 | 1 | cfgs_dota1.0_res50_v2.py |    
| FPN (memory consumption) | **ResNet152_v1d (600,800,1024)->MS** | DOTA1.0 trainval | DOTA1.0 test | 81.23 | [model](https://drive.google.com/file/d/1HaSU75llga_Em1O73Jp8ZwOoHPZtaVGj/view?usp=sharing) | **ALL** | **2x** | **Yes** | 2X Quadro RTX 8000 | 1 | cfgs_dota1.0_res152_v1.py |     

