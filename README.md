# Fully Convolutional Boundary Regression for Retina OCT Segmentation

A code implement from 1[https://link.springer.com/content/pdf/10.1007%2F978-3-030-32239-7_14.pdf]

## Getting Started

Download and preprocess data from https://github.com/heyufan1995/oct_preprocess.

## Train
```
python train.py --cfg [your_config] --exp [exp-name]
```

## Test

```
python test.py --cfg [your_config] --checkpoint [model pth]
```

## Eval with picture

You need to modify the config name and experiment name in code.
### Eval with Hc dataset
```
python eval_hc.py
```

### Eval with Dme dataset
```
python eval_dme.py
```