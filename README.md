
 # tensorflow reimplementation of HiFill (CVPR 2020 Oral Paper)

<a href="https://arxiv.org/abs/2005.09704">Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting</a>


If the codes helps you in reserach, please cite the following paper:


```
@misc{yi2020contextual,
    title={Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting},
    author={Zili Yi and Qiang Tang and Shekoofeh Azizi and Daesik Jang and Zhan Xu},
    year={2020},
    eprint={2005.09704},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<a href="https://www.youtube.com/watch?v=Q7mX5Bstv7U">Youtube video</a>


## how to train

### get dependancies installed on the conda environment, see requirements.txt for details. The following package are required:

* tensorflow-gpu
* opencv
* scipy
* pyyaml
* neuralgym
* easydict

### get data prepared as in ./data/examples/

### specify the training data path in config.yaml

### run the following script to start training. 200000~300000 steps would be enough for good converngence

```
python train.py
```

## how to test

```
python test.py --image_dir='./data/test/images' --mask_dir='./data/test/masks' --output_dir='outputs' --checkpoint_dir='./model_logs/places2' --input_size=512 --times=8
```

## Exemplar Experimental results:

![HD](imgs/hd.jpg?raw=true)
![compare](imgs/compare.jpg?raw=true)

