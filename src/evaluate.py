import argparse
from model import Model

def evaluate(dir):
    model = Model(hyperparameters=None, load_existing=True)
    model.load_model(dir)
    model.test_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    args = parser.parse_args()
    evaluate(args.model_dir)