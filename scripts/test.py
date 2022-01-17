"""
Usage:
    main.py [options] 
Options:
    -h --help                         show this screen
    --model-path=<str>                path of the trained model
    --max-length=<int>                text length [default: 128]
    --seed=<int>                      seed [default: 0]
    --test-batch-size=<int>           batch size [default: 32]
    --lang=<str>                      language choice [default: English]
    --test-path=<str>                 file path of the test set [default: ]
"""
from learner import EvaluateOnTest
from model import SpanEmo
from data_loader import DataClass
from torch.utils.data import DataLoader
import torch
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--model-path', type=str)
ap.add_argument('--max-length', type=int, default=128)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--test-batch-size', type=int, default=32)
ap.add_argument('--lang', type=str, default='English')
ap.add_argument('--test-path', type=str)

args = ap.parse_args()
args = {
    '--model-path': args.model_path,
    '--max-length': args.max_length,
    '--seed': args.seed,
    '--test-batch-size': args.test_batch_size,
    '--lang': args.lang,
    '--test-path': args.test_path
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda:0':
    print("Currently using GPU: {}".format(device))
    np.random.seed(int(args['--seed']))
    torch.cuda.manual_seed_all(int(args['--seed']))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("Currently using CPU")
#####################################################################
# Define Dataloaders
#####################################################################
test_dataset = DataClass(args, args['--test-path'])
test_data_loader = DataLoader(test_dataset,
                              batch_size=int(args['--test-batch-size']),
                              shuffle=False)
print('The number of Test batches: ', len(test_data_loader))
#############################################################################
# Run the model on a Test set
#############################################################################
model = SpanEmo(lang=args['--lang'])
learn = EvaluateOnTest(model, test_data_loader, model_path=args['--model-path'])
learn.predict(device=device)
