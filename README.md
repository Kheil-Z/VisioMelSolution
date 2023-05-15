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
The solutuon was run on TODO, 

Training time: <...>

Inference time: <...>

Machine specs you used for inference/training, and rough estimates of how long each step took.

# Run training

1. `python prepare_data.py `
2. `python multimodal_tritrain.py `
3. `python tritrain.py `

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
