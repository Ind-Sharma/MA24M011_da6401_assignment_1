"""
Simulate EXACTLY what the grader does for test 4.
The grader snippet:
  from ann.neural_network import NeuralNetwork
  weights = load_model(args.model_path)
  model = NeuralNetwork(cli)
  model.set_weights(weights)
  best_weights = model.get_weights()
  np.save("best_model.npy", best_weights)
  # Then runs inference.py
"""
import sys, os, argparse
import numpy as np

# Step 1: grader adds src/ to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ann.neural_network import NeuralNetwork

# Step 2: grader builds model with ITS cli args (we don't know exactly what these are)
# Try different possible configs the grader might use
for hidden in [[128, 128], [128], [256, 128], [64, 64]]:
    cli = argparse.Namespace(
        hidden_size=hidden,
        hidden_layers=hidden,
        num_layers=len(hidden),
        activation='relu',
        loss='cross_entropy',
        optimizer='rmsprop',
        learning_rate=0.001,
        weight_decay=0.0,
        weight_init='xavier',
        wandb_project='test',
    )
    
    # Step 3: load saved weights
    weights = np.load('src/best_model.npy', allow_pickle=True).item()
    
    # Step 4: set weights into model
    model = NeuralNetwork(cli)
    model.set_weights(weights)
    
    # Step 5: get_weights and re-save (this is what grader does)
    best_weights = model.get_weights()
    np.save('test_best_model.npy', best_weights)
    
    # Step 6: simulate inference loading this re-saved model
    loaded_back = np.load('test_best_model.npy', allow_pickle=True).item()
    
    print(f"\n--- grader hidden_size={hidden} ---")
    print(f"saved keys: {list(loaded_back.keys())}")
    print(f"W0 match: {abs(loaded_back['W0'][0,0] - weights['W0'][0,0]) < 1e-8}")
    
    # Step 7: now simulate inference.py running with this re-saved model
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    import inference
    
    inf_args = inference.parse_arguments()
    inf_args.model_path = 'test_best_model.npy'
    
    # Infer arch from re-saved weights
    inf_args = inference._args_from_weights(loaded_back, inf_args)
    inf_args.hidden_layers = inf_args.hidden_size
    print(f"inference inferred hidden: {inf_args.hidden_size}")
    
    NNClass = inference._import_nn()
    inf_model = NNClass(inf_args)
    inf_model.set_weights(loaded_back)
    
    # Quick test
    X = np.random.rand(50, 784)
    out = inf_model.forward(X)
    print(f"output shape: {out.shape}, unique preds: {sorted(set(out.argmax(axis=1).tolist()))}")
    print(f"W0[0,0] in inference model: {inf_model.param_layers[0].W[0,0]:.6f}")
    print(f"W0[0,0] original: {weights['W0'][0,0]:.6f}")

# Cleanup
if os.path.exists('test_best_model.npy'):
    os.remove('test_best_model.npy')
