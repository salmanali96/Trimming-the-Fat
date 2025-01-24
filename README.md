
# [BMVC 2024] Trimming the Fat: Efficient Compression of 3D Gaussian Splats through Pruning


This code is build upon the official implementation of the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering". To set up the code, please refer to the original repository





For training the baseline model, please use the following command
```shell
python train_baseline.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path for the model to be saved>
```

For training the model with our pruning approach 
```shell
python train_prune.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path for the model to be saved> --pruning_level <gamma_iter value> --start_checkpoint <path to pre-trained model> --iteration <number of iteration to train>
```

For evaluating the model 
```shell
python render.py -m <path to trained model> --iteration <the iteration number at which the model is to be loaded> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

For evaluating FPS
```shell
python fps.py -m <path to trained model> --iteration <the iteration number at which the model is to be loaded> # Calculate FPS
```
