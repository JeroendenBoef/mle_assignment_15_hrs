### MLE showcase
This repo is the resulting product of 15 - 20 hrs of work devided over a 6 day timespan. The dataset consists of textual, categorical and numerical data regarding water pipeline service status and influencing factors. The resulting repository contains dockerized training scripts to generate a LightGBM model, a dockerized API written in FastAPI with model wrappers for flexibility and dockerized unittest which can be launched together with the API using docker-compose. Finally, the repo performs automated flake8 and black styling reformatting using pre-commit hooks.

### Design choices
LightGBM was used as classifier over other Gradient Boosting frameworks due to it's leaf-wise growth, resulting in better performance for the high amount of categorical features in this dataset. 

FastAPI is employed to serve the model inference for quick deployment and adaptability. In case higher troughput and custimization is required, this could be swapped out for Triton.

### Training
Train model locally with `python train.py`, alter training parameters by changing `LIGHTGBM_TRAINING_PARAMS` in `constants.py`.

Train model in an isolated container with:
```
docker run -it \
    -v $(pwd)/models/model_checkpoints:/opt/app/models/model_checkpoints \
    -v $(pwd)/data:/opt/app/data \
    train:latest
```

Data preprocessing is handled by helper functions in `models.utils.py`, categorical features are encoded to numerical categories, timestamps are converted to unix timedeltas from current unix timestamp. NaN values for categorical features are filled as their own category, adding a "NaN category". Training is performed on a stratified split of 80/10/10 for train/validation/test, respectively. Due to high initial performance, API deployment and infrastructure was prioritized over hyperparameter tuning. Sweeping configurations can be included in the `train.py` script.

### Build image
Build new API docker image with: 
```
DOCKERBUILDKIT=1 docker build -t case:latest -f Dockerfiles/Dockerfile.server .
```

### Run API container
```
docker run -it -v $(pwd)/models/model_checkpoints:/opt/app/model_checkpoints -p 8050:8050 case:latest
```

### Tests
Launch both API container and test container with unittests to validate API and model outputs with:
```
docker-compose -f docker-compose-tests.yml up --build
```
