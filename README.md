# AdaFreq for Wildlife Re‑Identification  


## Environment
```bibtex
cd AdaFreq
conda create -n adafreq python=3.10 -y
conda activate adafreq
pip install -r requirements.txt
```

## Training (single GPU)
```bibtex
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
  --nproc_per_node=1 --master_port 51119 --use_env \
  ./Adafreq/train.py \
  --config_file ./Adafreq/configs/ATRW/vit_transreid_stride.yml \
  MODEL.DIST_TRAIN True
```

## Evaluation
```bibtex
python ./Adafreq/test.py \
  --config_file ./Adafreq/configs/ATRW/vit_transreid_stride.yml \
  MODEL.DEVICE_ID "('0')"
```

## Multi‑GPU Training
```bibtex
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4 --master_port 51119 --use_env \
  ./Adafreq/train.py \
  --config_file ./Adafreq/configs/ATRW/vit_transreid_stride.yml \
  MODEL.DIST_TRAIN True
```

## Switching to Another Wildlife Dataset
> **Quick tip:**  
> **1.** Copy the ATRW `.yml` config **and** its matching datase Python files.  
> **2.** Edit the copy—change only `DATASETS.ROOT_DIR` (and `DATASETS.NAMES` if you wish).  
> **3.** Use the new config with `--config_file …` to train / test on *any* wildlife dataset.

## Citation
If you find this code useful for your research, please cite our paper

```bibtex
@inproceedings{li2024adaptive,
  title        = {Adaptive high-frequency transformer for diverse wildlife re-identification},
  author       = {Li, Chenyue and Chen, Shuoyi and Ye, Mang},
  booktitle    = {European Conference on Computer Vision},
  pages        = {296--313},
  year         = {2024},
  organization = {Springer}
}
```

## Citation
Our code is based on TransReID. Thanks for the great work!

```bibtex
@inproceedings{He_2021_ICCV,
    author    = {He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
    title     = {TransReID: Transformer-Based Object Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15013-15022}
}
```
