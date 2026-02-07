import argparse
from model import Model


def main(hyper_params):
    """
    Calls the training loop

    Args:
        hyper_params: a dict object with the hyperparameters of the model and the training loop
    """
    model = Model(hyper_params)
    model.train_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    hyperparameters = {
        "layers": args.layers,
        "hidden": args.hidden_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size
    }
    main(hyperparameters)
