# CMU 11777 Multimodal Machine Learning Project: Multimodel VQA in Autonomous Driving

This repository contains all course materials and project code for CMU 11-777 Multimodal Machine Learning. The core project focuses on enhancing 
vision-language models (VLMs) using BEVFusion features (image + LiDAR) for the DriveLM-nuScenes dataset.

> 📄 [View Project Final Report](./doc/Final_Report.pdf)

## Project Highlights

- Integrates **Bird’s Eye View (BEV)** and multimodal fusion features from LiDAR and camera inputs.
- Fine-tunes a **T5-based vision-language model** (T5-Base / T5-Large) using LoRA and optionally frozen backbones.
- Evaluates language generation quality using standard metrics: BLEU-4, METEOR, ROUGE-L, CIDEr.

## Structure

- `EM-VLM4AD-patrick/`: Contains all code and resources for:
  - **Pretraining**
    - `train.py`: Main script for pretraining with various configurations.
    - `train_ddp.py`: Distributed training (DDP) version of `train.py`.
    - `run.sh`: Example training commands.
    - `run_ddp.sh`: Script to launch multi-GPU DDP training.
  - **Fine-tuning**
    - Shares training scripts above with different argument configurations.
    - Supports LoRA, frozen backbone, and checkpoint loading.
  - **Inference & Evaluation**
    - `eval.py`: Runs evaluation and generates predictions and metrics.
  - **Dataset Preparation**
    - `modules/`: Contains `multi_frame_model.py`, `multi_frame_dataset.py`, and `bevfusion_dataset.py`.
    - `add_lidar_to_multi_frame.py`: Tool to add LiDAR context to the multi-frame dataset.
    - `data/`: (Not tracked) Expected to contain:
      - `multi_frame/`: JSON files like `multi_frame_train.json`, `multi_frame_val.json`, `multi_frame_test.json`
      - `QA_dataset_nus/`, `nuscenes/`: Auxiliary DriveLM/nuScenes resources.
      - `bevfusion_feats/`: Directory for BEVFusion features  
        (default: `/data/Datasets/NuScenes-QA/bevfusion_feats/DriveLM/all/`specified in `EM-VLM4AD-patrick/modules/bevfusion_dataset.py`. Should be modified to your own location.)  
        ⚠️ Features are `.pt` files and will be **available soon**
  - `multi_frame_results/`: Stores pretrained checkpoints and logs.
  - `README.md`: Project introduction and usage.
  - `env.yml`: Conda environment configuration.

## How to Run

### Setup
All training and evaluation commands should be executed under:
```bash
cd ./EM-VLM4AD-patrick
```

---

### Pre-training

**T5-Base**
```bash
CUDA_VISIBLE_DEVICES=0 python ./train.py --batch-size 8 --epochs 8 --freeze-lm --num-workers 16 --load-checkpoint --checkpoint-file ./multi_frame_results/T5-Base/latest_model.pth --output-dir /data/patrick/mmml_saving/bev_Q_pretrained_T5-Base/ --lm T5-Base --load-orig-format
```

**T5-Large**
```bash
CUDA_VISIBLE_DEVICES=0 python ./train.py --batch-size 8 --epochs 8 --freeze-lm --num-workers 16 --load-checkpoint --checkpoint-file ./multi_frame_results/T5-Large-Q/latest_model.pth --output-dir /data/patrick/mmml_saving/bev_Q_pretrained_T5-Q-Large/ --lm T5-Large --load-orig-format
```

---

### Fine-tuning

**Without Pre-training**
```bash
CUDA_VISIBLE_DEVICES=4 python ./train.py --batch-size 8 --epochs 6 --lora --num-workers 16 --checkpoint-file /data/patrick/mmml_saving/bev_Q_pretrained_T5-Base/latest_model_saved.pth --load-checkpoint --output-dir /data/patrick/mml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr1e-5/ --learning-rate 1e-5 --feat bevfusion
```

**With Pre-training**
```bash
CUDA_VISIBLE_DEVICES=4 python ./train.py --batch-size 8 --epochs 6 --lora --num-workers 16 --checkpoint-file /data/patrick/mmml_saving/bev_Q_pretrained_T5-Q-Large/latest_model_saved.pth --load-checkpoint --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_T5-base_lr1e-4/ --learning-rate 1e-4 --feat bevfusion --restart
```

---

### Evaluation

**Without Pre-training**
```bash
CUDA_VISIBLE_DEVICES=3 python ./eval.py --batch-size 8 --lora --checkpoint-file /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr5e-4/latest_model_saved.pth --load-checkpoint --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_wo_pretrained_T5-base_lr5e-4/eval_result
```

**With Pre-training**
```bash
CUDA_VISIBLE_DEVICES=3 python ./eval.py --batch-size 8 --lora --checkpoint-file /data/patrick/mmml_saving/bev_Q_finetuned_T5-base_lr5e-4/latest_model_saved.pth --load-checkpoint --output-dir /data/patrick/mmml_saving/bev_Q_finetuned_T5-base_lr5e-4/eval_result
```

---

### Reference

You can also refer to the convenience script:
```bash
bash ./EM-VLM4AD-patrick/run.sh
```

---

## Metrics

Evaluation uses standard generation metrics:
- **BLEU-4**: Precision of 4-gram matches
- **METEOR**: Synonym-aware precision/recall
- **ROUGE-L**: Longest common subsequence overlap
- **CIDEr**: Consensus-based scoring over references

---

## Notes

- Dataset files (`.json`) should be downloaded manually and placed in `./data/multi_frame/`.
- Large files like model checkpoints are tracked using Git LFS.
