import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="PTARS")

    parser.add_argument('--dataset', type=str, default='amazon-book',
                        help="choose from ['gowalla', 'yelp2018', 'amazon-book']")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of the model")
    parser.add_argument('--epochs', type=int,default=301,
                        help="the total epochs of training the model")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="whether we use tensorboard") 
    parser.add_argument('--seed', type=int,default=2022,
                        help="the random seed for the experiment")   
    parser.add_argument('--batch', type=int,default=1024,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--topks', nargs='?',default="[20, 50]",
                        help="@k test list")    
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")           
    parser.add_argument('--decay', type=float,default=0,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--t', type=float,default=0.11,
                        help="temperature coeffient in softmax")
    parser.add_argument('--a', type=float,default=20,
                        help="self loop strength")
    parser.add_argument('--norm_type', type=str,default='0.6',
                        help="D^{-p}AD^{-(1-p)}")
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--loss_type', type=str, default='softmax',
                        help="choose from ['log', 'bpr', 'softmax']")
    parser.add_argument('--sim_type', type=str, default='cos',
                        help="choose from ['ip', 'cos', 'non_sym'], where 'ip' denotes inner product")
    parser.add_argument('--bpr_sim_type', type=str, default='cos',
                        help="choose from ['ip', 'cos', 'non_sym'], where 'ip' denotes inner product")
    parser.add_argument('--test_sim_type', type=str, default='cos',
                        help="choose from ['ip', 'cos'], where 'ip' denotes inner product")
    parser.add_argument('--neg_k', type=int,default=1024,
                        help="the number of negative samples for softmax loss")
    parser.add_argument('--device', type=int,default=1,
                        help="choose which gpu to use")
    parser.add_argument('--alpha', type=float,default=1.,
                        help="trade off between ranking loss and teacher loss")
    parser.add_argument('--beta', type=float,default=0.2,
                        help="trade off between ranking loss and teacher loss")
    parser.add_argument('--teacher_k', type=int,default=50,
                        help="items per user that provided by label propagation")
    parser.add_argument('--drop_ratio', type=float,default=0.05,
                        help="dropout ratio on label propagation computing")
    parser.add_argument('--k', type=int,default=20,
                        help="test topk for propagation matrix")
    parser.add_argument('--epsilon', type=int,default=100,
                        help="self-adjust factor")
    parser.add_argument('--noise', type=float,default=0.,
                        help="noise ratio of the original dataset")

    
    return parser.parse_args()