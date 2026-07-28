# CAST: Context-Aware Dynamic Latent Space Transformation for Interactive Text-to-Image Retrieval

## Datasets
Please download all datasets from their respective official websites.
- COCO 2017 Unlabeled Images
- VisDial

```
- VisDial
  ├── train
  │   ├── images
  │   └── visdial_1.0_train.json
  └── val
      ├── images
      └── visdial_1.0_val.json
```
After preparing the datasets above, please run the following script to generate the reconstructed dialogue captions:

```
python recon_dialog.py --split train --run_idx 0
```
## Environments

- **Ubuntu** 20.04 
- **CUDA** 12.6 
- **Python** 3.10 

Use the following instructions to create the corresponding conda environment. 
Please make sure to download the required pretrained models (e.g., BLIP) from their official sources before running the training or evaluation scripts.

```
conda create --name CAST python=3.10 -y
conda activate CAST
pip install -r requirements.txt
```
## Direct Training and Evaluation

- Before training, please make sure the reconstructed dialogue captions have been generated using `recon_dialog.py` and saved in the `dial_recon` folder.
- After preparing the datasets and reconstructed captions, you can start model training and evaluation using the provided shell scripts.

### Run the following script for multi-GPU finetuning on VisDial:

```
./train.sh $temp $recon_path $pretrained
```

**Argument meaning**

- `$temp` — contrastive loss temperature
- `$recon_path` — path to the reconstructed dialogue
- `$pretrained` — path to the pretrained BLIP checkpoint

**Example**

```
./train.sh 0.03 ./dial_recon pretrained/blip_base.pth
```

### Run the following script for evaluation with reconstructed dialogue captions:

```
./eval.sh $model $cache $data_dir $queries $recon $finetuned_weights
```

**Argument meaning**

- `$model` — name of the retrieval backbone (e.g., `blip`)
- `$cache` — path to the cached corpus feature file
- `$data_dir` — root directory of the dataset
- `$queries` — path to the VisDial query JSON file
- `$recon` — path to the reconstructed dialogue file
- `$finetuned_weights` — finetuned model checkpoint





