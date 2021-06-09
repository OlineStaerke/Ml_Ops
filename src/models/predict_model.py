import sys, os
import argparse

import torch
from model import MyAwesomeModel
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import helper


class Predict(object):
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

    def predict(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
        )

        # Change directory
        dir_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(dir_path)

        model = MyAwesomeModel()
        print("Predict until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--load_model_from", default="")
        parser.add_argument("--load_data_from", default="")

        args = parser.parse_args(sys.argv[2:])
        print(args)

        # Load model in to system
        if args.load_model_from:
            state_dict = torch.load(args.load_model_from)
            model.load_state_dict(state_dict)

        # Load data in to system
        if args.load_data_from:

            img = Image.open(args.load_data_from)

        else:
            img = Image.open("../../data/raw/00000.jpg")

        # Transform
        img = transform(img)
        model.eval()

        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = model(img)

        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp(logps)
        helper.view_classify(img.view(1, 28, 28), ps)
        plt.show()


if __name__ == "__main__":
    Predict()
