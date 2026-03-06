"""
Inference Script
Evaluate trained models on test sets
"""
import argparse
import json
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.data_loader import load_dataset,one_hot_encode
from src.ann.neural_network import NeuralNetwork


def parse_arguments():
    """
    Parse command-line arguments for inference.

    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('-m','--model_path',type=str,default='src/best_model.npy')
    parser.add_argument('-d','--dataset',choices=['mnist','fashion_mnist'],default='mnist')
    parser.add_argument('-b','--batch_size',type=int,default=64)
    parser.add_argument('-sz','--hidden_layers',type=int,nargs='+',default=[128])
    parser.add_argument('-n','--num_neurons',type=int,default=128)
    parser.add_argument('-a','--activation',choices=['relu','sigmoid','tanh'],default='relu')
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model,X_test,y_test):
    """
    Evaluate model on test data.

    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    y_hat = model.forward(X_test)
    y_hat_labels = np.argmax(y_hat,axis=1)
    loss = model.loss_fn.forward_pass(y_hat.T,one_hot_encode(y_test).T)

    accuracy = accuracy_score(y_test,y_hat_labels)
    precision = precision_score(y_test,y_hat_labels,average='macro',zero_division=0)
    recall = recall_score(y_test,y_hat_labels,average='macro',zero_division=0)
    f1 = f1_score(y_test,y_hat_labels,average='macro',zero_division=0)

    return {
        "logits":y_hat,
        "loss":loss,
        "accuracy":accuracy,
        "f1":f1,
        "precision":precision,
        "recall":recall
    }


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()

    model_path = args.model_path
    config_path = args.model_path.replace(".npy","_config.json")

    with open(config_path,'r') as f:
        config = json.load(f)
    nn_args = argparse.Namespace(**config)

    model = NeuralNetwork(nn_args)
    weights = load_model(model_path)
    model.set_weights(weights)

    X_train,y_train,X_test,y_test = load_dataset(args.dataset)
    result = evaluate_model(model,X_test,y_test)

    print("Accuracy:",result["accuracy"])
    print("Precision:",result["precision"])
    print("Recall:",result["recall"])
    print("F1-score:",result["f1"])
    print("Loss:",result["loss"])
    print("Evaluation complete!")

    return result



if __name__ == '__main__':
    main()
