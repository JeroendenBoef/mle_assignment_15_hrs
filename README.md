### Design choices
LightGBM was used as classifier over other Gradient Boosting frameworks due to it's leaf-wise growth, resulting in better performance for the high amount of categorical features in this dataset. 

### Training
Train model with `python train.py`, alter training parameters by changing `LIGHTGBM_TRAINING_PARAMS` in `constants.py` 

### Build image
Build new docker image with: 
```
DOCKERBUILDKIT=1 docker build -t case:latest -f Dockerfiles/Dockerfile.server .
```


### Run API container
```
docker run -it -v $(pwd)/models/model_checkpoints:/opt/app/model_checkpoints -p 8050:8050 case:latest
```

TODO:
- ONNX export
- Analysis
- Runtime comparison
- Dockerfile for training, run with bash script
- Add internal tests for models in middle build stage 