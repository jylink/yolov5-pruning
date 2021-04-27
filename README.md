# yolov5-pruning

Channel-wise pruning of yolov5



Preparation:

1. Download this compatible [yolov5](https://github.com/ultralytics/yolov5/tree/4890499344e21950d985e1a77e84a0a4161d1db0)
2. Download this compatible [Torch-Pruning](https://github.com/VainF/Torch-Pruning/tree/e94be9f8f4c641360a82ddb42338aae4798631c8)
3. Copy `torch_pruning` to `yolov5`
4. Download this repo and copy to `yolov5`



Usage:

1. Sparse learning:  train new model with `--sl_factor`, L1 loss will be add to weights of all batchnorm layers
2. Pruning: `python pruning.py --weights xxx.pt --thres 0.01`, channels with a batchnorm weight that is higher than a threshold will be removed
3. Fine-tuning: train the pruned model with `--ft_pruned`



Reference: 

* https://github.com/Syencil/mobile-yolov5-pruning-distillation
* https://github.com/VainF/Torch-Pruning