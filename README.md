# S-VGAE
## Description
This is a representation learning method for predicting protein-protein interactions (PPI). Our paper about this method is currently under review.
The auto-encoder part of our model is improved based on the implementation by T. N. Kifp. You can find the source code here [Graph Auto-Encoders](https://github.com/tkipf/gae)

## Usage
### Requirements
- Python 2.7
- TensorFlow
- Keras
- networkx
- scipy
- numpy

### Run
> cd src

> python main.py

You can specify the dataset by using "--dataset E.coli", for example, you can also specify the parameter weight rate by using "--wr 10.0"


