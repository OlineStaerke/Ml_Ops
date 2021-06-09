# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
from torchvision import datasets, transforms
import os


@click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    # Define a transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
    )

    trainset = datasets.MNIST(".", download=True, train=True, transform=transform)
    train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    torch.save(train, "../../data/processed/train.pt")

    testset = datasets.MNIST(".", download=True, train=False, transform=transform)
    test = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    torch.save(test, "../../data/processed/test.pt")

    # Creating predict dataset

    dataset = datasets.MNIST(root=dir_path)

    for idx, (img, _) in enumerate(dataset):
        if idx <= 10:
            img.save("../../data/raw/{:05d}.jpg".format(idx))

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
