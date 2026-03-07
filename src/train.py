import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ann.neural_network import NeuralNetwork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['mnist', 'fashion_mnist'], default='mnist')
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-o', '--optimizer', choices=['sgd', 'momentum', 'rmsprop'], default='sgd')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128])
    parser.add_argument('-a', '--activation', choices=['relu', 'sigmoid', 'tanh'], default='relu')
    parser.add_argument('-l', '--loss', choices=['cross_entropy', 'mse'], default='cross_entropy')
    parser.add_argument('-w_i', '--weight_init', choices=['random', 'xavier', 'zeros'], default='xavier')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-w_p', '--wandb_project', type=str, default='da6401_assignment1')
    _src = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('-m', '--model_save_path', type=str, default=os.path.join(_src, 'trained_model.npy'))
    args, _ = parser.parse_known_args()
    args.hidden_layers = args.hidden_size

    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args),
                   settings=wandb.Settings(start_method="thread"))

    from utils.data_loader import load_dataset
    X_train, y_train, X_test, y_test = load_dataset(args.dataset)
    model = NeuralNetwork(args)

    best_f1 = 0.0
    best_weights = None
    for ep in range(args.epochs):
        avg_loss = model.train(X_train, y_train, epochs=1, batch_size=args.batch_size)
        result = model.evaluate(X_test, y_test)
        print("Epoch " + str(ep+1) + "/" + str(args.epochs) + ", loss: " + str(round(avg_loss, 4)) + ", accuracy: " + str(round(result['accuracy'], 4)) + ", f1: " + str(round(result['f1'], 4)))
        if use_wandb:
            wandb.log({"epoch": ep+1, "loss": avg_loss, "accuracy": result['accuracy'], "f1": result['f1']})
        if result['f1'] > best_f1:
            best_f1 = result['f1']
            best_weights = model.get_weights()

    if best_weights is not None:
        model.set_weights(best_weights)
    result = model.evaluate(X_test, y_test)
    np.save(args.model_save_path, model.get_weights())

    if use_wandb:
        wandb.log({"test_accuracy": result['accuracy'], "test_f1": result['f1']})
        wandb.finish(quiet=True)

    print("\n--- Final metrics ---")
    print("Accuracy: " + str(result['accuracy']))
    print("Precision: " + str(result['precision']))
    print("Recall: " + str(result['recall']))
    print("F1-score: " + str(result['f1']))
    print("Best model saved to " + str(args.model_save_path))


if __name__ == '__main__':
    main()
