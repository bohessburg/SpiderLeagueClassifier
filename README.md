# SpiderLeague

A fine-grained image classifier that identifies 15 species of spiders using a ConvNeXt-Tiny backbone with a custom multi-layer classification head. 
This project is primarily for learning purposes.

## Species

Black Widow, Blue Tarantula, Bold Jumper, Brown Grass Spider, Brown Recluse Spider, Deinopis Spider, Golden Orb Weaver, Hobo Spider, Huntsman Spider, Ladybird Mimic Spider, Peacock Spider, Red Knee Tarantula, Spiny Backed Orb Weaver, White Knee Spider, Yellow Garden Spider

## Architecture

- **Backbone**: ConvNeXt-Tiny (pretrained on ImageNet) outputs a 768-dimensional feature vector
- **Classification head**: `Linear(768 -> 512) -> BatchNorm -> ReLU -> Dropout(0.4) -> Linear(512 -> 256) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(256 -> 15)`
- **Training strategy**: Two-phase fine-tuning
  - Phase 1 (10 epochs): Backbone frozen, only the classification head trains
  - Phase 2 (20 epochs): Full model fine-tuned with discriminative learning rates (backbone 1e-5, head 1e-4) and cosine LR decay
- **Data augmentation**: RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, ImageNet normalization

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download the dataset

1. Create a Kaggle account and generate an API token
2. Save your credentials to `~/.kaggle/kaggle.json`:
   ```bash
   mkdir -p ~/.kaggle
   echo '{"username":"YOUR_USERNAME","key":"YOUR_TOKEN"}' > ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Download the dataset:
   ```bash
   kaggle datasets download -d gpiosenka/yikes-spiders-15-species -p data --unzip
   ```

## Usage

### Train the model

```bash
python spiderleague.py
```

This trains the model in two phases and saves `best_spider_model.pth` with the best validation accuracy. After training, it evaluates on the test set and prints a per-class classification report with a confusion matrix.

### Classify a single image (inference only)

```bash
python spiderleague.py --infer path/to/spider_photo.jpg
```

If `best_spider_model.pth` exists locally it uses that. Otherwise it downloads the pretrained model from Hugging Face Hub.

### Upload the trained model to Hugging Face Hub

```bash
huggingface-cli login
python spiderleague.py --upload
```

Update `HF_REPO_ID` in `spiderleague.py` with your Hugging Face username first.

### Google Colab

The project also works in Colab. See the notebook for the Colab-specific setup (Kaggle credentials, `!pip install`, and optional Google Drive model saving).

## Dataset

[Yikes! Spiders! 15 Species](https://www.kaggle.com/datasets/gpiosenka/yikes-spiders-15-species) from Kaggle. ~2,185 training images, 75 validation images, and 75 test images across 15 spider species.

## To Do
- expand dataset to include more species
- create a UI to upload pictures for classification
