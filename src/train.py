import argparse
import numpy as np
import os
import sys

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ann.neural_network import NeuralNetwork

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-d','--dataset',choices=['mnist','fashion_mnist'],default='mnist')
    p.add_argument('-e','--epochs',type=int,default=20)
    p.add_argument('-b','--batch_size',type=int,default=64)
    p.add_argument('-lr','--learning_rate',type=float,default=0.01)
    p.add_argument('-o','--optimizer',choices=['sgd','momentum','rmsprop'],default='sgd')
    p.add_argument('-nhl','--num_layers',type=int,default=1)
    p.add_argument('-sz','--hidden_size',type=int,nargs='+',default=[128])
    p.add_argument('-a','--activation',choices=['relu','sigmoid','tanh'],default='relu')
    p.add_argument('-l','--loss',choices=['cross_entropy','mse'],default='cross_entropy')
    p.add_argument('-w_i','--weight_init',choices=['random','xavier','zeros'],default='xavier')
    p.add_argument('-wd','--weight_decay',type=float,default=0.0)
    p.add_argument('-w_p','--wandb_project',type=str,default='da6401_assignment1')
    src = os.path.dirname(os.path.abspath(__file__))
    p.add_argument('-m','--model_save_path',type=str,default=os.path.join(src,'trained_model.npy'))
    args,_ = p.parse_known_args()
    args.hidden_layers = args.hidden_size # set hidden layers

    use_wandb = bool(os.environ.get("WANDB_API_KEY")) # only use wandb if key exists
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project,config=vars(args),
                   settings=wandb.Settings(start_method="thread"))

    from utils.data_loader import load_dataset
    x_tr,y_tr,x_te,y_te = load_dataset(args.dataset) # load data
    model = NeuralNetwork(args)

    for ep in range(args.epochs): # train loop
        loss = model.train(x_tr,y_tr,epochs=1,batch_size=args.batch_size)
        res = model.evaluate(x_te,y_te)
        print("epoch "+str(ep+1)+"/"+str(args.epochs)+" loss="+str(round(loss,4))+" acc="+str(round(res['accuracy'],4))+" f1="+str(round(res['f1'],4)))
        if use_wandb:
            wandb.log({"epoch":ep+1,"loss":loss,"accuracy":res['accuracy'],"f1":res['f1']})

    np.save(args.model_save_path,model.get_weights()) # save model
    if use_wandb:
        wandb.log({"test_accuracy":res['accuracy'],"test_f1":res['f1']})
        wandb.finish(quiet=True)
    print("acc="+str(res['accuracy'])+" f1="+str(res['f1']))
    print("saved to "+str(args.model_save_path))

if __name__=='__main__':
    main()
