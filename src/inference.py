import argparse
import numpy as np
import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork

def parse_arguments():
    src = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('-m','--model_path',type=str,default=os.path.join(src,'best_model.npy'))
    p.add_argument('-d','--dataset',choices=['mnist','fashion_mnist'],default='mnist')
    p.add_argument('-e','--epochs',type=int,default=20)
    p.add_argument('-b','--batch_size',type=int,default=128)
    p.add_argument('-lr','--learning_rate',type=float,default=0.001)
    p.add_argument('-o','--optimizer',choices=['sgd','momentum','rmsprop'],default='rmsprop')
    p.add_argument('-nhl','--num_layers',type=int,default=2)
    p.add_argument('-sz','--hidden_size',type=int,nargs='+',default=[128,128])
    p.add_argument('-a','--activation',choices=['relu','sigmoid','tanh'],default='relu')
    p.add_argument('-l','--loss',choices=['cross_entropy','mse'],default='cross_entropy')
    p.add_argument('-w_i','--weight_init',choices=['random','xavier','zeros'],default='xavier')
    p.add_argument('-wd','--weight_decay',type=float,default=0.0)
    p.add_argument('-w_p','--wandb_project',type=str,default='da6401_assignment1')
    args,_ = p.parse_known_args()
    return args

def main():
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    from utils.data_loader import load_dataset

    src = os.path.dirname(os.path.abspath(__file__))
    args = parse_arguments()

    loaded = False
    for mp in [os.path.join(os.getcwd(),'best_model.npy'),os.path.join(src,'best_model.npy')]:
        if os.path.exists(mp):
            weights = np.load(mp,allow_pickle=True).item()
            loaded = True
            break
    if not loaded:
        weights = np.load(args.model_path,allow_pickle=True).item()

    wk = sorted([k for k in weights if k.startswith('W') and k[1:].isdigit()],key=lambda k:int(k[1:]))
    hidden = [np.array(weights[k]).shape[0] for k in wk[:-1]]
    args.hidden_size = hidden
    args.hidden_layers = hidden

    model = NeuralNetwork(args)
    model.set_weights(weights)

    _,_,x_te,y_te = load_dataset(args.dataset)
    yhat = model.forward(np.array(x_te,dtype=float))
    preds = np.argmax(yhat,axis=1)
    y_te = np.array(y_te,dtype=int).flatten()

    nc = yhat.shape[1]
    yoh = np.zeros((len(y_te),nc))
    yoh[np.arange(len(y_te)),y_te] = 1
    loss = float(model.loss_fn.forward_pass(yhat.T,yoh.T))

    res = {
        "logits":yhat,
        "loss":loss,
        "accuracy":accuracy_score(y_te,preds),
        "precision":precision_score(y_te,preds,average='macro',zero_division=0),
        "recall":recall_score(y_te,preds,average='macro',zero_division=0),
        "f1":f1_score(y_te,preds,average='macro',zero_division=0),
    }
    print("Accuracy:",res["accuracy"])
    print("Precision:",res["precision"])
    print("Recall:",res["recall"])
    print("F1-score:",res["f1"])
    print("Loss:",res["loss"])
    print("Evaluation complete!")
    return res

if __name__=='__main__':
    main()
