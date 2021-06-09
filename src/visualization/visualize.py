import sys
import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
sys.path.append("../")
from models.model import MyAwesomeModel


class Visualize(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def visualize(self):
        # Change directory
        dir_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(dir_path)

        model = MyAwesomeModel()

        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--load_model_from", default="")
        parser.add_argument("--load_data_from", default="")

        args = parser.parse_args(sys.argv[2:])

        # Load model in to system
        if args.load_model_from:
            state_dict = torch.load(args.load_model_from)
            model.load_state_dict(state_dict)
        else:
            state_dict = torch.load("../../models/model.pth")
            model.load_state_dict(state_dict)

        # Load data in to system
        if args.load_data_from:
            test_set = torch.load(args.load_data_from)

        else:
            test_set = torch.load("../../data/processed/test.pt")

        predictions = torch.empty(64, 64)
        pred_labels = [0] * 64

        with torch.no_grad():
            model.eval()
            for images, labels in test_set:
                log_ps = model(images, return_f=True)
                s = log_ps.shape
                if s[0] == 64:
                    predictions = torch.cat([predictions, log_ps], 0)
                    pred_labels += labels.tolist()

        print(pred_labels)
        x = np.asarray(TSNE(n_components=2).fit_transform(predictions))

        plt.scatter(
            x=x[:, 0], y=x[:, 1], c=pred_labels,
        )
        plt.show()


if __name__ == "__main__":
    Visualize()
