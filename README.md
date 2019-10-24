# Enhancing Adversarial Example Transferability with an Intermediate Level Attack
This repository includes the official PyTorch implementation of [Enhancing
Adversarial Example Transferability with an Intermediate Level Attack
](https://arxiv.org/abs/1907.10823) (ICCV 2019).

Summary: We fine-tune adversarial perturbations for better transferability by
maximizing projection onto the perturbation at an intermediate layer. We
demonstrate improved transferability across a wide range of attacks,
including SOTA ones. In addition, we demonstrate that the choice of layer
makes a substantial impact on transferability.

## Software Requirements
This codebase requires Python 3, PyTorch 1.0+, Torchvision 0.2+, and pretrainedmodels ([Cadene's repo](https://github.com/Cadene/pretrained-models.pytorch), installed via `pip install pretrainedmodels`). In principle, this code can be run on CPU but we assume GPU utilization throughout the codebase.

## Usage

### Demo

To generate an adversarial example with enhanced transferability using our method (using I-FGSM as the baseline attack), please run a command such as the following:

```
python demo.py --modeltype ResNet18 --layerindex 4 --imagepath test_images/bear_test_image_label_296.JPEG --imagelabel 296 --outpath adv_out.jpg --epsilon 0.03
```

This command uses ILA to generate a transferable adversarial (with epsilon=0.03) for the given bear image.

The output is:
```color
True label: 296 (ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus)
ResNet18 (source model)
Prediction on original: 296 (ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus)
Prediction on I-FGSM: 257 (Great Pyrenees)
Prediction on ILA: 155 (Shih-Tzu)

---Transfer Results Follow---
DenseNet121 (transfer model)
Prediction on original: 296 (ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus)
Prediction on I-FGSM: 296 (ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus)
Prediction on ILA: 222 (kuvasz)

alexnet (transfer model)
Prediction on original: 296 (ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus)
Prediction on I-FGSM: 296 (ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus)
Prediction on ILA: 222 (kuvasz)

SqueezeNet1.0 (transfer model)
Prediction on original: 296 (ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus)
Prediction on I-FGSM: 257 (Great Pyrenees)
Prediction on ILA: 257 (Great Pyrenees)
```

As can be seen from the output, the original I-FGSM perturbation on ResNet18 transfers to only 1 model (SqueezeNet), whereas the ILA perturbation transfers to all 3 models.

### Evaluate on All Images of a Dataset

To evaluate the performance of the attack on cifar10, run

```
python all_in_one_cifar10.py --source_models ResNet18 --transfer_models ResNet18 DenseNet121 GoogLeNet SENet18 --out_name=test.csv  --attacks ifgsm --num_batches=50 --batch_size=32

```
[Checkpoints](https://drive.google.com/drive/folders/1RGtlPCc2vTqeQc5utOgLb1Y3_vIO5JVi?usp=sharing) of our models. You can replace the checkpoints paths in `cifar10_config.py`. 

Full usage:

```
usage: all_in_one_cifar10.py [-h] --source_models SOURCE_MODELS
                             [SOURCE_MODELS ...] --transfer_models
                             TRANSFER_MODELS [TRANSFER_MODELS ...] --attacks
                             ATTACKS [ATTACKS ...] --num_batches NUM_BATCHES
                             --batch_size BATCH_SIZE --out_name OUT_NAME

optional arguments:
  -h, --help            show this help message and exit
  --source_models SOURCE_MODELS [SOURCE_MODELS ...]
                        <Required> source models
  --transfer_models TRANSFER_MODELS [TRANSFER_MODELS ...]
                        <Required> transfer models
  --attacks ATTACKS [ATTACKS ...]
                        <Required> base attacks
  --num_batches NUM_BATCHES
                        <Required> number of batches
  --batch_size BATCH_SIZE
                        <Required> batch size
  --out_name OUT_NAME   <Required> out file name
```

Run `all_in_one_imagenet.py` to evaluate on imagenet with similar usage, with the imagenet val folder path. 

## Attribution

If you use this code or our results in your research, please cite:

```
@article{Huang2019EnhancingAE,
  title={Enhancing Adversarial Example Transferability with an Intermediate Level Attack},
  author={Qian Huang and Isay Katsman and Horace He and Zeqi Gu and Serge J. Belongie and Ser-Nam Lim},
  journal={ArXiv},
  year={2019},
  volume={abs/1907.10823}
}
```
