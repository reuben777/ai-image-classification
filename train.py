import argparse
from helper_functions import go_go_data_loaders, go_go_model, go_go_network, go_go_save_checkpoint, go_go_load_checkpoint, go_go_print, go_go_check_accuracy

arg_parse = argparse.ArgumentParser(description='train.py')

arg_parse.add_argument('--base_data_directory', nargs='*', default="./flowers")
arg_parse.add_argument('--arch', default='densenet121', nargs='*', action="store", type = str)
arg_parse.add_argument('--checkpoint', default='./checkpoint.pth', nargs='*', action="store", type = str)
arg_parse.add_argument('--use_checkpoint',  action='store_true', default=False)
arg_parse.add_argument('--learning_rate', action="store", type=int, default=0.001)
arg_parse.add_argument('--dropout', action="store", type=int, default = 0.5)
arg_parse.add_argument('--training_iterations', dest="epochs", action="store", type=int, default=15)
arg_parse.add_argument('--hidden_layer', default='500', type = int)
arg_parse.add_argument('--check_accuracy', action='store_true', default=False)
arg_parse.add_argument('--device', type = str, default="cuda")

args = arg_parse.parse_args()

dataloaders, datasets = go_go_data_loaders(args.base_data_directory)

if args.use_checkpoint:
    try:
        go_go_print('Using Checkpoint')
        model, criterion, optimizer, model_settings = go_go_load_checkpoint(args.checkpoint)
    except:
        print('No checkpoint found. One will be generated after first training.')
else:
    model, criterion, optimizer, model_settings = go_go_model(args.arch, args.hidden_layer, 102, args.learning_rate, args.dropout, args.device)
    # do long-running work here
go_go_print('Args: Architecture: {}, Use Checkpoint: {}, Learning Rate: {}, Dropout: {}, Iterations: {}, Hidden Layer: {}, Device: {}'
            .format(args.arch,
                    args.use_checkpoint,
                    args.learning_rate,
                    args.dropout,
                    args.epochs,
                    args.hidden_layer,
                    args.device
                   ))

model = go_go_network(model, criterion, optimizer, dataloaders, datasets, args.epochs, args.device)

go_go_save_checkpoint(args.checkpoint, model, datasets['training'], model_settings)

if args.check_accuracy:
    go_go_check_accuracy(dataloaders['testing'], model)
