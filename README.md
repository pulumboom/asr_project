# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

Project for automatic speech recognition.
Weights could be found <a href="https://disk.yandex.ru/client/disk/asr_models">here</a>
Report: <a href="https://api.wandb.ai/links/pulumboom/3nkl6e5n">here</a>

| Best Model | test-clean | test-other |
|:----------:|:----------:|:----------:|
|    CER     |    0.06    |    0.17    |
|    WER     |    0.21    |    0.42    |

## Installation

Follow these steps to install the project:

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

For custom inference you need to refactor /configs/inference.yaml config based on your needs, specify save_path and datasets.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
