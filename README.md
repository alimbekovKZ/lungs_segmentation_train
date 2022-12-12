# Lungs segmentation train pipeline
Lungs segmentation train pipeline

![https://habrastorage.org/webt/vk/jv/8r/vkjv8rjd04f1oicbczq5hyadhv0.png](https://habrastorage.org/webt/vk/jv/8r/vkjv8rjd04f1oicbczq5hyadhv0.png)

This repo is build on top of [https://github.com/sneddy/pneumothorax-segmentation/tree/master/unet_pipeline](https://github.com/sneddy/pneumothorax-segmentation/tree/master/unet_pipeline)

## Installation

`pip install -r requirements.txt`

The python package: https://github.com/alimbekovKZ/lungs_segmentation

### Example inference

Jupyter notebook with the example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_9MXcHg_iqTtycXrz4c8oLzPmryA_3EW?usp=sharing)

### WebApp

[lungs segmentation streamlit](https://alimbekovkz-lungs-segmentation-demo-app-r1t0f4.streamlit.app/)

## Data Preparation

Download the dataset from: [https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels](https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels)

Process the data using [script](https://github.com/alimbekovKZ/lungs_segmentation_train/blob/main/utils/prepare_division.sh)

The script will split masks into left and right lungs and store them in the directory `dataset/ml` and `dataset/mr`.

## Training

### Define the config.
Example at [resnet18/train_config](https://github.com/alimbekovKZ/lungs_segmentation_train/blob/main/experiments/resnet18/train_config.yaml)

You already have ready-made config files for networks at your disposal: resnet18, resnet34, resnet50, seresnext50 and densenet121

### Models results

| model | best dice | Mb |
|-------|-----------|----|
|  seresnext50     |  0.9669          | 165.4   |
|   resnet34    | 0.9657          |  103.4  |
|   densenet121    |  0.9655         |   62.8 |
|     resnet18  |   0.9623       |  73.4  |
|    resnet50   |     0.9534      |  591.7  |


You can choose model with greater score or maybe you need less weighted.

### Training

```
sh train.sh
```

or

```
python train.py <path to config>
```

### Inference

Jupyter notebook with the example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_9MXcHg_iqTtycXrz4c8oLzPmryA_3EW?usp=sharing)

Code example for resnet34:

```
import models.selim_zoo.unet as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.Resnet(seg_classes=2, backbone_arch='resnet34')
model = model.to(device)
model.load_state_dict(torch.load('experiments/resnet34/model_checkpoints/resnet34_epoch6.pth'))

def inference(img_path, thresh = 0.2):
    model.eval();
    image = cv2.imread(f'{img_path}')
    image = (image-image.min())/(image.max()-image.min())
    augs = aug_test(image = image)
    image = augs['image'].transpose((2,0,1))
    im = augs['image']
    image = np.expand_dims(image, axis = 0)
    image = torch.tensor(image)

    mask = torch.nn.Sigmoid()(model(image.float().cuda()))
    mask = mask[0,:,:,:].cpu().detach().numpy()
    mask = (mask>thresh).astype('uint8')
    return im, mask

image, mask = inference('test/1.jpeg', 0.2)
```


### Results on data from the Internet

#### seresnext50

![https://habrastorage.org/webt/vs/pl/sh/vsplshrczby2zuuxyhsboil54n0.png](https://habrastorage.org/webt/vs/pl/sh/vsplshrczby2zuuxyhsboil54n0.png)

#### resnet34

![https://habrastorage.org/webt/e3/mb/kc/e3mbkcxsmos6q4jlw5-tybudzji.png](https://habrastorage.org/webt/e3/mb/kc/e3mbkcxsmos6q4jlw5-tybudzji.png)

#### densenet121

![https://habrastorage.org/webt/ef/01/zo/ef01zo2g2qgsux8ses4keg4g8is.png](https://habrastorage.org/webt/ef/01/zo/ef01zo2g2qgsux8ses4keg4g8is.png)

#### resnet18

![https://habrastorage.org/webt/fz/-n/nz/fz-nnzgezbc_zwgmat6ztc3yxf0.png](https://habrastorage.org/webt/fz/-n/nz/fz-nnzgezbc_zwgmat6ztc3yxf0.png)

#### resnet50

![https://habrastorage.org/webt/eb/os/gk/ebosgkdotkrt7emke0btwyonp-e.png](https://habrastorage.org/webt/eb/os/gk/ebosgkdotkrt7emke0btwyonp-e.png)


### Authors

Renat Alimbekov, Ivan Vassilenko, Abylaikhan Turlassov
