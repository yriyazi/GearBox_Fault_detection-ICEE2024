# Deep Learning Module

The **deeplearning** folder within this repository holds the core components related to the deep learning aspects of the project. This module includes functionalities for training and evaluating neural network models tailored for remote sensing image captioning. The main script in this folder, `base.py`, encompasses various functions and classes necessary for training, saving and loading models, and conducting evaluations. Below is an overview of the components and their functionalities:

## `base.py` Overview

The `base.py` script includes the following components:

### `AverageMeter` Class

This class calculates and stores the average and current values, which are commonly used in training and validation loops.

### `save_model` Function

This function is responsible for saving a trained model checkpoint along with its optimizer state, if available. It accepts the model, optimizer, and paths for saving.

### `load_model` Function

This function loads a saved model checkpoint and its corresponding optimizer state, if available. It returns the loaded model and optimizer.

### `normal_accuracy` Function

This function computes the accuracy of predictions against ground truth labels, considering only non-padding tokens.

### `teacher_forcing_decay` Function

This function calculates the teacher forcing ratio decay based on the current epoch and the total number of epochs. Teacher forcing is used during training to guide the decoder with ground truth tokens.

### `train` Function

The `train` function encapsulates the complete training process. It iterates over epochs, performs training and validation loops, and saves model checkpoints at specified intervals. The function accepts various parameters including data loaders, model architecture, training settings, loss criterion, optimizer, learning rate scheduler, and more.

In the training loop, the function updates the model's weights based on the loss and performs various logging tasks. It also applies teacher forcing decay during training if specified.

The function additionally handles validation and testing evaluations if desired, recording accuracy and loss values.

## Usage

To utilize the deep learning functionalities provided in `base.py`, follow these steps:

1. Import the required functions and classes from `base.py` in your main script.
2. Define your neural network model architecture and set up data loaders.
3. Configure training settings, loss criterion, optimizer, and learning rate scheduler.
4. Call the `train` function, passing the necessary parameters.

Example code snippet:

```python
from deeplearning.base import train

# Define your model, data loaders, and other parameters

# Configure loss criterion, optimizer, and learning rate scheduler

# Call the train function
train(
    train_loader,
    val_loader,
    model,
    model_name,
    epochs,
    load_saved_model,
    ckpt_save_freq,
    ckpt_save_path,
    ckpt_path,
    report_path,
    criterion,
    optimizer,
    lr_scheduler,
    sleep_time,
    Validation_save_threshold,
    test_loader,
    test_evaluate,
    device,
    Teacher_forcing_train,
    Teacher_forcing_num_epochs
)
```

## Acknowledgments

The deep learning functionalities and components presented in `base.py` have been developed by the project contributors. We acknowledge the insights gained from the fields of deep learning, image captioning, and remote sensing.

For more in-depth understanding and context, refer to the code within the `deeplearning` directory and the associated documentation.

## Contact

If you have any inquiries, suggestions, or contributions related to the deep learning module or any other aspects of the project, feel free to contact us at [project_email@example.com](mailto:project_email@example.com). Your engagement and input are highly valued!
