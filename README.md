# VisioMel Challenge: Predicting Melanoma Relapse

Tilted-Towers (lucas.robinet and Kheil-Z) solution


## Repo organization
```
.
├── README.md          <- You are here!
├── example_documentation_guide.pdf <- Solution Documentation
├── README_template.md <- Template that you can fill in to document your solution code ????
├── src                <-  Solution source code
└── models             <- Trained models

```
 
## Summary

Our adopted solution adopts a multi-task learning approach in order to accurately predict melanoma relapse.

The dataset at hand consists of multimodal data ([Detailed dataset description](https://www.drivendata.org/competitions/148/visiomel-melanoma/page/674/)):
- **Images**: *WSIs* are given, these correspond to pyramidal Tiff images contaning recursively downsampled versions of the original full resolution slice scan. 
- **Tabular data**: an array of *clinical data* (tabular data), such as age, sex and body site location are given.
- **Labels** : *Relapse* bool, *Ulceration* bool and *Breslow* categories.


In order to make the most of all available data our submitted solution encodes the *image data** as a *512 dimentional vector* using a ResNet18 finedtuned on predicting the various labels. Meanwhile *tabular data* are mapped using well-designed dictionnaries to various values. A shallow fully connected network is then applied to the tabular tensor data as well as the image projection in the ResNet's latent space. Finally three classification layers allow us to train our network to predict the labels described above.

Our final model is thus composed as follows: 

TODO : inser drawio figure?

Models are trained with various loss functions (BCE and FocalLoss), and the three loss terms are averaged using a weighted sum before backpropating the resulting loss.

TODO : more?

## Setup
1. Install the prerequisities:
    - Python version 3.11.2 (TODO)
    - AWS CLI ([installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))
    - `git clone https://github.com/Kheil-Z/VisioMelSolution.git `
    - `cd VisioMelSolution`
2. Install the required python packages:
    - `conda create --name visioMel `
    - `conda install pip `
    - `pip install -r requirements.txt `

# Hardware
The solution was run on on a desktop machine (AMD EPYC 7502P 32-Core Processor, 64 GB RAM) with 4 NVIDIA GeForce 2080 Ti.

Training time: ~ 30 minutes

Inference time: ~ 5 minutes

Machine specs you used for inference/training, and rough estimates of how long each step took.

# Run training (Skip if want to infer using pretrained models)

1. `python src/prepare_data.py ` or `python src/prepare_data.py --data_path PATH/TO/TIFFS/` 
   - Splits the given data into the stratified train and validation datasets which we used. (saved as .csv files in data/)
   - Either : 
      - Iteratively downloads the Tiff images from the competions AWS bucket, saves a downsampled version of the image and deletes the heavy Tiff file. (saved as .png files in data/images/)
      - Or if Tiff data is already available, use --data_path flag to signal a path to a folder which should be structured as: 
      
            ```
            .
            ├── train_metadata.csv          <- Data description csv (see challenge for expected format)
            ├── train_labels.csv    <- Labels (see challenge for expected format)
            └── images/    <- Folder containing all TIFF format images
            ```
2. `python src/tritrain.py `
   - Train and save the ResNet on the downloaded images. (You should now have a "tritrain.pth" file in the "models/" directory.)
3. `python src/multimodal_tritrain.py `
   - Train and save the Multi-modal deep classifier on the image embeddings and tabular data. (You should now have a "after_finetune.pth" file in the "models/" directory.)

# Run inference
#### Using pretrained models (skip previous steps):
` python src/run_infer.py --model_path ../models/pretrained_after_finetune.pth --tritrain_path ../models/pretrained_tritrain.pth `
#### Using new model (if you followed the previous training steps):
` python src/run_infer.py --model_path ../models/after_finetune.pth --tritrain_path ../models/tritrain.pth `

| Arg              | Default                   | Help                                                            |
|------------------|---------------------------|-----------------------------------------------------------------|
| --model_path     | models/after_finetune.pth | Path to trained model used as classifier.                       |
| --tritrain_path  | models/tritrain.pth       | Path to trained resNet model used as feature extractor.         |
| --model_type     | FC                        | Tpe of model ("FC" or "CNN", see model class)                   |
| --relapse_only   | False                     | Describes model outputs (see model class)                       |
| --data_path      | data/                     | Path to data, must contain 'metada.csv' and tiff format images. |
| --age_trick_bool | True                      | use age trick at inference                                      |

# Disclaimer
