# yolov5-pruning



Update 2021/10/4: adapt to the new version of yolov5



---

Channel-wise pruning of yolov5



Preparation:

1. Download **yolov5**

   ```
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   git reset --hard 59aae85a7e40701bb872df673a6ef288e99a4ae3
   ```

2. Download this compatible **Torch-Pruning**

   ```
   git clone https://github.com/VainF/Torch-Pruning
   cd Torch-Pruning
   git reset --hard ec12e0590aad28e607e1df9feb2baf60c8cda689
   ```

3. Copy `torch_pruning` to `yolov5`

4. Download this repo and copy to `yolov5`



Usage:

1. Sparse learning:  train new model with `--sl_factor`, L1 loss will be add to weights of all batchnorm layers
2. Pruning: `python prune.py --shape [batchsize channel height width] --prob 0.1 --weights [xxx.pt] --save_path [xxx_pruned.pt]`, channels with a batchnorm weight that is higher than a threshold will be removed
3. Fine-tuning: train the pruned model with `--ft_pruned`



Reference: 

* https://github.com/Syencil/mobile-yolov5-pruning-distillation
* https://github.com/VainF/Torch-Pruning