"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.data_loader import load_dataset, one_hot_encode
from src.ann.neural_network import NeuralNetwork


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset',choices=['mnist','fashion_mnist'],default='mnist')
    parser.add_argument('-e','--epochs',type=int,default=20)
    parser.add_argument('-b','--batch_size',type=int,default=64)
    parser.add_argument('-lr','--learning_rate',type=float,default=0.01)
    parser.add_argument('-o','--optimizer',choices=['sgd','momentum','rmsprop'],default='sgd')
    parser.add_argument('-nhl','--num_layers',type=int,default=1)
    parser.add_argument('-sz','--hidden_size',type=int,nargs='+',default=[128])
    parser.add_argument('-a','--activation',choices=['relu','sigmoid','tanh'],default='relu')
    parser.add_argument('-l','--loss',choices=['cross_entropy','mse'],default='cross_entropy')
    parser.add_argument('-w_i','--weight_init',choices=['random','xavier','zeros'],default='xavier')
    parser.add_argument('-wd','--weight_decay',type=float,default=0.0)
    parser.add_argument('-w_p','--wandb_project',type=str,default='da6401_assignment1')
    parser.add_argument('-m','--model_save_path',type=str,default='src/best_model.npy')
    return parser.parse_args()


def save_best_model(weights,config,model_save_path):
    np.save(model_save_path,weights)
    config_path = model_save_path.replace(".npy","_config.json")
    with open(config_path,"w") as f:
        json.dump(config,f,indent=2)


def main():
    args = parse_arguments()
    args.hidden_layers = args.hidden_size

    import wandb
    wandb.init(project=args.wandb_project,config=vars(args))

    X_train,y_train,X_test,y_test = load_dataset(args.dataset)
    model = NeuralNetwork(args)

    best_f1 = 0.0
    best_weights = None
    best_config = None
    for ep in range(args.epochs):
        avg_loss = model.train(X_train,y_train,epochs=1,batch_size=args.batch_size)
        result = model.evaluate(X_test,y_test)
        print(f"Epoch {ep+1}/{args.epochs},loss: {avg_loss:.4f}, accuracy:{result['accuracy']:.4f},f1:{result['f1']:.4f}")
        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_weights = model.get_weights()
            best_config = {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "optimizer": args.optimizer,
                "num_layers": args.num_layers,
                "hidden_size": list(args.hidden_size),
                "hidden_layers": list(args.hidden_layers),
                "activation": args.activation,
                "loss": args.loss,
                "weight_init": args.weight_init,
                "weight_decay": args.weight_decay,
                "wandb_project": args.wandb_project,
                "model_save_path": args.model_save_path,
            }

    model.set_weights(best_weights)
    result = model.evaluate(X_test,y_test)

    save_best_model(best_weights,best_config,args.model_save_path)

    wandb.log({"test_accuracy": result["accuracy"],"test_precision":result["precision"],
               "test_recall":result["recall"],"test_f1":result["f1"]})
    wandb.finish()

    print("\n--- Final metrics ---")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision:{result['precision']:.4f}")
    print(f"Recall:{result['recall']:.4f}")
    print(f"F1-score:{result['f1']:.4f}")
    print(f"Best model saved to {args.model_save_path}")


if __name__ == '__main__':
    main()
