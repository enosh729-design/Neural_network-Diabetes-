# Neural Network for Diabetes Progression Prediction

This repository contains a Neural Network model built with TensorFlow/Keras to predict diabetes progression based on the "Diabetes dataset" from Scikit-Learn. It was originally developed using Google Colab.

## Dataset
The dataset involves 10 baseline variables (age, sex, body mass index, average blood pressure, and six blood serum measurements) corresponding to 442 diabetes patients. The target is a continuous quantitative measure of disease progression one year after the baseline.

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow

Install dependencies by running:
```bash
pip install -r requirements.txt
```

## Model Architecture

The model uses a simple Multilayer Perceptron (MLP) architecture configured for regression:
- **Input layer**: Receives 10 scaled features.
- **Hidden Layer 1**: 64 neurons, ReLU activation, 20% Dropout.
- **Hidden Layer 2**: 32 neurons, ReLU activation, 20% Dropout.
- **Output Layer**: 1 neuron, Linear activation.

The model is optimized using the Adam optimizer with a Mean Squared Error (MSE) loss function. It uses Early Stopping with a patience of 20 epochs to prevent overfitting and restore the best weights based on validation loss.

## How to Run

Run the generated python script directly from your terminal:

```bash
python "neural_network(diabetes)).py"
```

Or you can open the Jupyter Notebook version:
```bash
jupyter notebook "Neural_network(Diabetes)).ipynb"
```

## Visualizations
After training, the script evaluates its performance mapped across the test dataset and outputs:
- **Learning Curve**: Model Loss (MSE) showing Training vs Validation Loss over epochs.
- **True vs Predicted**: A scatter plot matching predicted values against real ones, with a perfect prediction reference line.
