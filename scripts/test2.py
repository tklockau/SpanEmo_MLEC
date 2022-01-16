from learner import EvaluateOnTest
from model import SpanEmo
from data_loader import DataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import numpy as np


args = docopt(__doc__)
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
learn = EvaluateOnTest(model, test_data_loader, model_path='models/' + args['--model-path'])
learn.predict(device=device)


