# animal-classification

This repostory contains the code for animal-calssifcation task published in class.

## Dataset
Click [here](https://bhpan.buaa.edu.cn/#/link/7398EF6A7925D8A23C48EE4E9ED919E1) to download.

## Installation

`requirements.txt` includes its dependencies, you migh want to change Python's
version, Python 3.6 or later should work fine.

## Download Model Checkpoints

By default, the script will download the checkpint under the repository.
you could change it by setting the path variable while running the script.

lude wrappers for these models. Run the model's .py file to download its checkpoint or view instructions for downloading. For example, if you want to download the `Resnet152`'s checkpoint, please run:

```shell
python example/resnet512.py path/to/download
```

Alternatively, click [here](https://1drv.ms/u/s!Al5BF1i8TRVbiQt2whUMwlzTn69R?e=qMr3XT) to download `Resnet512`'s checkpoint.

## Examples

### train model

For example, to train a `resnet512` with dataset under `data`:

```shell
python train.py data --arch resnet512 --batch-size 128 -j 0 --gpu 0 --pretrained --lr 1e-4 
```


### resume model to inference

For example, to inference the labels of images under `data/test` with model resume from `model_best.pth.tar`, it will print out the reusult and dump it to a `json` file as well.

```shell
python inference.py --data data/test -a resnet512 --gpu 0 -batch-size 128 -j 0 --resume model_best.pth.tar
```

## reference 
[*Meta Pseudo Labels*](https://github.com/kekmodel/MPL-pytorch)