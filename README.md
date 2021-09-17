## RT3
#### step1: Download the pre-compressed model

 - BaiDu NetDisk link:  https://pan.baidu.com/s/1OIV08F4tUqyFD1DGfbkJ5g 
 - Extraction code：1dtv 

#### step2: Environment
conda env create -f pytorch_BCM_pruning.yml
 - python=3.6.9
 - pytorch=1.3.0
 - tensorboard=1.14.0
 - torchtext=0.4.0
 - torchvision=0.4.1
 - ......

#### step3: start from rl_controller.py

 - Modify the rl_input.py   **"timing_constraint":115#115 for high, 104 for middle,94 for low** for different timing constraints
 - run: CUDA_VISIBLE_DEVICES=XXX nohup python -u d_rl_controller.py > result

 
