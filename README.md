# SR_RiverSegmentation

Modified README of: code for Super-Resolution Deep Neural Networks for Water Classification from Free Multispectral Satellite Imagery

(original README is included in repo as old_README).
### Requirements

Inference only (a bit hacky stuff due to mmseg):

```bash
pip install -r requirements.txt
```

After this finishes, mim should be installed. Then instal mmcv:
```bash
mim instal mmengine
mim install mmcv==2.1.0
```
 
---

**Inference:**

```bash
python main.py
```

We provide three designs of the Sentinel-2 Super Resolution Segmentation model:
```
1: Baseline (dice_noSR): DeepLabV3+ with ResNet50 using DiceBCE loss without Super-resolution operations
2: SR-Dice (dice): DeepLabV3+ with ResNet50 using DiceBCE loss
3: SR-BCE (bce): DeepLabV3+ with ResNet50 using BCE loss
```

Our pretrained model checkpoints can be downloaded from [checkpoints](https://drive.google.com/drive/folders/1u3jlJdKWEbR0TaA9opYDEVvcF4YA5W6p?usp=sharing)


### Acknowledgement

The segmentation networks are implemented based on the [OpenMMLab](https://github.com/open-mmlab/mmsegmentation)
