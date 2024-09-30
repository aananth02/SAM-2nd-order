import argparse
import torch

#from models.model_factory import MODEL_GETTERS
from models.wrn import WideResNet
from models.smooth_cross_entropy import smooth_crossentropy
# from data.cifar10 import Cifar10
from data.cifar100 import Cifar100
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import sys; sys.path.append("..")
from sam_ga_implementation import SAM
# from sam import SAM
# from ssam import SSAM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--optim", default="sam", type=str, help="If True then use SAM/ ASAM, else use SGD alone")
    parser.add_argument("--model", default="resnet18", type=str, help="Default model for training")
    parser.add_argument("--num_classes", default=10, type=int, help="Cifar10 has 10 classes")
    parser.add_argument("--steps", default=1, type=int, help="n in n-step SAM")
    # TODO: Need to add an argument for cifar10
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar100(args.batch_size, args.threads)
    log = Log(log_each=10)
    # model = MODEL_GETTERS[args.model](
    #     num_classes=args.num_classes, pretrained=False).to(device)
    # model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=100).to(device)
    base_optimizer = torch.optim.SGD
    # optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # print(model)
    # print(dataset)
    print(args.optim)
    
    if args.optim == "sam":
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == "ssam":
        optimizer = SSAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = base_optimizer(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    if args.optim == 'sgd':
        for epoch in range(args.epochs):
            model.train()
            log.train(len_dataset=len(dataset.train))

            for batch in dataset.train:
                inputs, targets = (b.to(device) for b in batch)

                # forward step
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                
                # backward step
                loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    correct = torch.argmax(predictions.data, 1) == targets
                    log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                    scheduler(epoch)

            model.eval()
            log.eval(len_dataset=len(dataset.test))

            with torch.no_grad():
                for batch in dataset.test:
                    inputs, targets = (b.to(device) for b in batch)

                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    log(model, loss.cpu(), correct.cpu())

            log.flush()
    else:
        loss_epsilon = None
        for epoch in range(args.epochs):
            model.train()
            log.train(len_dataset=len(dataset.train))


            for batch in dataset.train:
                inputs, targets = (b.to(device) for b in batch)

                # first forward-backward step
                enable_running_stats(model)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                loss.mean().backward()
                if (epoch == args.epochs - 1):
                    optimizer.dummy_step(zero_grad=False)
                    loss_epsilon = loss.sum().item() / loss.size(0)
                    optimizer.undo_dummy(zero_grad=False)
                optimizer.first_step(n=args.steps, zero_grad=True)

                # second forward-backward step
                disable_running_stats(model)
                smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
                optimizer.second_step(zero_grad=True)

                with torch.no_grad():
                    correct = torch.argmax(predictions.data, 1) == targets
                    log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                    scheduler(epoch)

            model.eval()
            log.eval(len_dataset=len(dataset.test))

            with torch.no_grad():
                for batch in dataset.test:
                    inputs, targets = (b.to(device) for b in batch)

                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    log(model, loss.cpu(), correct.cpu())

        log.flush()
        print(f"Sharpness in {args.steps} step SAM")
        print(loss_epsilon)
