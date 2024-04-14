# MLFMNet
Our results have been submitted to the official dataset under the username wxy07496, achieving fourth place. You can find more details at the following website.
 https://codalab.lisn.upsaclay.fr/competitions/7302#results 
We have also provided the scores as follows.
![UAVid Official](https://github.com/wxy16/FGFNet/assets/128227957/7413bf55-bc60-452d-a2bb-e957b2549109)

#The code will be open-sourced soon.

**2024.4.14**

MLFMNet was updated.


**UAVid**
```
python tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train_val" \
--output-img-dir "data/uavid/train_val/images" \
--output-mask-dir "data/uavid/train_val/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```
python tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train" \
--output-img-dir "data/uavid/train/images" \
--output-mask-dir "data/uavid/train/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```
python tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_val" \
--output-img-dir "data/uavid/val/images" \
--output-mask-dir "data/uavid/val/masks" \
--mode 'val' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

## Training

"-c" means the path of the config, use different **config** to train different models.

```
python train_supervision.py -c config/uavid/mlfmnet.py
```




**Training Code Reference **

 ([GeoSeg](https://github.com/WangLibo1995/GeoSeg))
 ([mmsegmentation](https://github.com/open-mmlab/mmsegmentation))
