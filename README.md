# CiT2
CiT2: Single-Stage Automatic License Plate Recognition with Contrastive Denoising Learning
## Usage
### install requirements
install torch with CUDA first, then 
`pip install -r requirements.txt`  
`libjpeg-turbo` is needed. If not available, please use cv2 to import img at `datasets\CCPD.py` 
### Train 

- download [CCPD](https://github.com/detectRecog/CCPD), [CCPD18](https://github.com/tomorrow1210/CCPD) and put at `dataset/CCPD` and `datasets/CCPD2018` 
- generate csv file as exampled `datasets/CCPD2018/ccpd_base_A.csv`. 
- run `train_Danlu_script` function in `test.py` with proper yaml config


### Eval
run `eval_model` in test.py
