import sys
import argparse

import torch
from data import mnist
from model import MyAwesomeModel
from torch import nn, optim
import matplotlib.pyplot as plt

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    


    def train(self):

        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
      
        args = parser.parse_args(sys.argv[2:])
        print(args)

        model = MyAwesomeModel()
        train_set, test_set = mnist()

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        epochs = 10
        train_losses, test_losses = [], []

        for e in range(epochs):
            running_loss = 0

            for images, labels in train_set:
                
                optimizer.zero_grad()
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
         
            print(f'Epoch {e+1} \t\t Training Loss: {running_loss / len(train_set)} \t\t Accuracy: {accuracy.item()*100}')
            train_losses.append(running_loss / len(train_set))

            val_loss = 0
            for images, labels in test_set:
                
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                val_loss += loss.item()
            
            test_losses.append(val_loss / len(test_set))

            model.train()


        plt.plot(train_losses)
        plt.plot(test_losses)
        plt.show()
        torch.save(model.state_dict(), 'checkpoint.pth')
        
    def evaluate(self):
        model = MyAwesomeModel()
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        if args.load_model_from:
        
            state_dict = torch.load('checkpoint.pth')
            model.load_state_dict(state_dict)
        _, test_set = mnist()

        criterion = nn.NLLLoss()
        valid_loss = 0.0
        accList = []
        with torch.no_grad():
        
            model.eval()
            for images, labels in test_set:
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                valid_loss+=loss

                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                accList.append(accuracy.item()*100)
            
            
            print(f'Accuracy: {sum(accList)/len(accList)}%')
            print(f'Validation Loss: {valid_loss / len(test_set)}')


if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    