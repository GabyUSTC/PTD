from dataloader import LP_Loader
from model import PureMF
from torch import optim
from procedure import Test, Softmax_train_teacher
import time
import torch
from os.path import join
from tensorboardX import SummaryWriter
from parse import parse_args

args = parse_args()

config = {}
config['dataset'] = args.dataset
config['drop_ratio'] = args.drop_ratio
config['k'] = args.k
config['lr'] = args.lr
config['neg_k'] = args.neg_k
config['latent_dim_rec'] = args.recdim
config['t'] = args.t
config['sim_type'] = args.sim_type
config['bpr_sim_type'] = args.bpr_sim_type
config['test_sim_type'] = args.test_sim_type
config['epochs'] = args.epochs
config['multicore'] = args.multicore
config['test_u_batch_size'] = args.testbatch
config['batch_size'] = args.batch
config['topks'] = eval(args.topks)
config['tensorboard'] = bool(args.tensorboard)
config['seed'] = args.seed
config['decay'] = args.decay
config['teacher_k'] = args.teacher_k
config['alpha'] = args.alpha
config['beta'] = args.beta
config['a'] = args.a
config['norm_type'] = args.norm_type
config['epsilon'] = args.epsilon
config['noise'] = args.noise

if args.device != -1:
    config['device'] = torch.device(args.device)
else:
    config['device'] = torch.device("cpu")
    
print("the settings of our experiment:")

for key, value in config.items():
    print(f"{key} : {value}")

dataset = LP_Loader(config['dataset'], config)
Recmodel = PureMF(config, dataset).to(config['device'])

opt = optim.Adam(Recmodel.parameters(), lr=config['lr'])
train_procedure = Softmax_train_teacher

Neg_k = config['neg_k']

ROOT_PATH = "./"
BOARD_PATH = join(ROOT_PATH, 'runs')
comment = 'mf'
epochs = config['epochs']

w = SummaryWriter(join(BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + comment))

best_recall = 0.
best_results = None
for epoch in range(config['epochs']):
    start = time.time()
    if epoch % 10 == 0:
        print("[TEST]")
        results = Test(dataset, Recmodel, epoch, config, w, config['multicore'])
        if results['recall'][0] > best_recall:
            best_recall = results['recall'][0]
            best_results = results
    output_information = train_procedure(dataset, Recmodel, epoch, config, opt, neg_k=Neg_k, w=w)
    Recmodel.epoch += 1
    print(f'EPOCH[{epoch}/{epochs}] {output_information}')
print('----------------------OPTIMIZATION FINISHED!---------------------------')
print(f"best result:")
print(best_results)