only the column pruning 
implement by yuhongsong

# Column_pruning 
You need to add transformer_model_lr_3.0_50.pt in directory "model" 
CUDA_VISIBLE_DEVICES=XXX nohup python -u c_only_random_column_pruning.py --epochs 10 --random > only_random_column_pruning_average
