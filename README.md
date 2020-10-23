# 42-Multilayer_Perceptron
In this project I explored the creation of a MLP network from scratch, to attempt to classify breast data into healthy and cancerious, with the best performance.
Familiarizing and exploring concepts in ML such as feedforward, backpropagation and gradient descent, along with procedures like data manipulation and score evaluation.

I had the idea to make the code as modular and customizable as possible, to be able to test different architectures and parameters for the models. With that in mind, different classes were created, notably one Model class and various Layer classes. For the executable file (multilayer_perceptron.py), a model was already created with specific parameters, but feel free to pick and try different architectures and values if you want to create a custom model.

## Installation
To execute this code, you will need python 3.7.* and install the modules in requirements.txt.

## Usage
The executable file that you should be running is multilayer_perceptron.py. You can always display help by using the -h or --help flags:
> python multilayer_perceptron.py --help

> python multilayer_perceptron.py predict --help

You now have to select one of the four processes that the program will run: train, predict, split, info.

### Train
Typically the first part of the program to be executed, and will use provided data to train a neural network classifier, that will be outputted in the networks folder with .model format.
It requires a data file (to be found in resources/data.csv) and either file containing some more data for validation, or a percentage to be reserved from input data that will be used as validation. Optional parameters allow name selection for the output file and statistic plotting of the scores during training.
One valid command would be:
> python multilayer_perceptron.py train resources/data.csv --val_split 20 --out BreastCancerClassifier --plot

A progress bar will be displayed with the ETA of training and validation scores will be printed when training is finished.

### Predict
Used to infer a predictions on data, using a .model trained network. Needs both the path to a model and path to data, and accepts parameters to save predictions to a .csv file, and a flag that indicated whther data is labelled and we should be running a validation.
Example command:
> python multilayer_perceptron.py predict networks/network.model --data data/fully_tagged_data.csv --validation

Scores will be printed in the case that a validation is run, and, if --save flag is used, predictions will be saved in a .csv inside predictions folder.

### Split
Used to separate some data into training and validation, given the percentage that should conform the validation set. Also accepts an optional parameter to name the folder in which the resulting splits will be stored.
For example:
> python multilayer_perceptron.py split resources/data.csv --val_split 35 --output_dir separated_data

Will output a folder with two files inside, train.csv and validation.csv.

### Info
Will give you information about a model stored at a .model file, including its paramaters during training, or its structure.
> python multilayer_perceptron.py info network/BreastCancerClassifier.model

Learning stats will be plotted as well as all model information available.
