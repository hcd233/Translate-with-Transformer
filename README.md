# Translate with Transformer

## Introduction

this is a program which uses '*Transformer*' model to do Bilingual translation between Chinese and English
and uses '*Streamlit*' to build a WebUI which support docs translation and sentences translation.

## Change Logs
[2023/7/28] Upload all codes. you can run [train-pt.py](./train-pt.py) to train the model, and run [app.py](./app.py) to start the WebUI

## Dataset

Challenge 2017 中英翻译数据集

## Requirements
```shell
pip install -r requirements.txt
```
* Run the command to install the requirements.
## Usage

### Train 

```shell
accelerate launch train-pt.py
```

* Before you run the command, you must check and modify the hyperparameter in the train-pt.py(no argparser)
* the 'accelerate launch' is default using all of GPUs in your device. If you want to specify the specific GPUs for training, run the *'export CUDA_VISIBLE_DEVICES={gpu_idx}'* command before training.

### WebUI
```shell
streamlit run app.py
```

* Run the command and Open the url "https://localhost:8501" to interact with WebUI.

