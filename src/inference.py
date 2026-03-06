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

def _get_nn_class():
    try:
        from src.ann.neural_network import NeuralNetwork
    except ImportError:
        from ann.neural_network import NeuralNetwork
    return NeuralNetwork

NeuralNetwork = _get_nn_class()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('-m','--model_path',type=str,default='src/best_model.npy')
    parser.add_argument('-d','--dataset',choices=['mnist','fashion_mnist'],default='mnist')
    parser.add_argument('-e','--epochs',type=int,default=20)
    parser.add_argument('-b','--batch_size',type=int,default=128)
    parser.add_argument('-lr','--learning_rate',type=float,default=0.001)
    parser.add_argument('-o','--optimizer',choices=['sgd','momentum','rmsprop'],default='rmsprop')
    parser.add_argument('-nhl','--num_layers',type=int,default=2)
    parser.add_argument('-sz','--hidden_size',type=int,nargs='+',default=[128,128])
    parser.add_argument('-a','--activation',choices=['relu','sigmoid','tanh'],default='relu')
    parser.add_argument('-l','--loss',choices=['cross_entropy','mse'],default='cross_entropy')
    parser.add_argument('-w_i','--weight_init',choices=['random','xavier','zeros'],default='xavier')
    parser.add_argument('-wd','--weight_decay',type=float,default=0.0)
    parser.add_argument('-w_p','--wandb_project',type=str,default='da6401_assignment1')
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model,X_test,y_test):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    y_hat = model.forward(X_test)
    y_hat_labels = np.argmax(y_hat,axis=1)

    # inline one_hot_encode to avoid any import dependency
    n_classes = y_hat.shape[1]
    Y_oh = np.zeros((len(y_test), n_classes))
    Y_oh[np.arange(len(y_test)), y_test.astype(int)] = 1
    loss = model.loss_fn.forward_pass(y_hat.T, Y_oh.T)

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
    args = parse_arguments()
    args.hidden_layers = args.hidden_size

    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    try:
        from src.utils.data_loader import load_dataset
    except ImportError:
        from utils.data_loader import load_dataset
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
