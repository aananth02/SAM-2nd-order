import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import the necessary modules for SAM and SecondOrderSAM
from sam import SAM
from second_order_sam import SecondOrderSAM
from model.wide_res_net import WideResNet
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR

# Define function to load MNIST dataset
def get_data_loader(batch_size, threads):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)
    return train_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add all arguments here
    parser.add_argument("--adaptive", action='store_true', help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loader(args.batch_size, args.threads)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=1, labels=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Assuming SAM and SecondOrderSAM optimizers are defined elsewhere
    sam_optimizer = SAM(model.parameters(), torch.optim.SGD, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, rho=args.rho, adaptive=args.adaptive)
    second_order_optimizer = SecondOrderSAM(model.parameters(), torch.optim.SGD, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, rho=args.rho, adaptive=args.adaptive)

    # Log files for SAM and SecondOrderSAM
    sam_log = Log('sam_training.log')
    second_order_sam_log = Log('second_order_sam_training.log')

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Training steps for SAM
            optimizer = sam_optimizer
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Additional steps specific to SAM or SecondOrderSAM if necessary

            # Log the training process
            sam_log.write(f'Epoch {epoch}, Loss: {loss.item()}')

        # Repeat for SecondOrderSAM with second_order_sam_log
        # Ensure to reset model to initial state or use a separate model instance for a fair comparison
            
    # Second training loop for SecondOrderSAM
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=1, labels=10).to(device)

    for epoch in range(args.epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Training steps for SAM
            optimizer = second_order_optimizer
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Additional steps specific to SAM or SecondOrderSAM if necessary

            # Log the training process
            second_order_sam_log.write(f'Epoch {epoch}, Loss: {loss.item()}') 

    # Ensure to flush the logs at the end
    sam_log.flush()
    second_order_sam_log.flush()
