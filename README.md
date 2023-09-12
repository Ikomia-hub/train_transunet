<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/train_transunet/main/icons/transunet.png" alt="Algorithm icon">
  <h1 align="center">train_transunet</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_transunet">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_transunet">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_transunet/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_transunet.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Training process for TransUNet model. This algorithm can train TransUNet model for semantic segmentation. 

![Medical TranUnet illustration](https://149695847.v2.pressablecdn.com/wp-content/uploads/2021/03/pasted-image-0-11.png)


## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "semantic_segmentation",
}) 

# Add training algorithm
train = wf.add_task(name="train_transunet", auto_connect=True)

# Launch your training on your data
wf.run()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters


- **input_size** (int) - default '256': Size of the input image.
- **epochs** (int) - default '15': Number of complete passes through the training dataset.
- **batch_size** (int) - default '1': Number of samples processed before the model is updated.
- **learning_rate** (float) - default '0.01': Step size at which the model's parameters are updated during training.
- **output_folder** (str, *optional*): path to where the model will be saved. 
- **num_workers** (int) - default '0': How many parallel subprocesses you want to activate when you are loading all your data during your training or validation. 
- **weight_decay** (float) - default '1e-4': Amount of weight decay, regularization method.
- **eval_period** (int) - default '100: Interval between evaluations.  
- **max_iter** (int) - default '1000': Maximum number of iterations. 
- **early_stopping** (bool) - default 'False': Activate early stopping callback to avoid over fitting.
- **dataset_split_ratio** (int) – default '90' ]0, 100[: Divide the dataset into train and evaluation sets.
- **patch_size** (int) - default '16':  Path size of the ViT model.


**Parameters** should be in **strings format**  when added to the dictionary.


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "semantic_segmentation",
}) 

# Add training algorithm
train = wf.add_task(name="train_transunet", auto_connect=True)
train.set_parameters({
    "batch_size": "1",
    "max_iter": "1000",
    "input_size": "256",
    "patch_size": "16",
    "dataset_split_ratio": "5",
    "eval_period": "50",
    "learning_rate": "0.01",
    "early_stopping": "False"
}) 

# Launch your training on your data
wf.run()
```

