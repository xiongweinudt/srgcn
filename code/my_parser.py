import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')     # 1000
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Initial learning rate.')  # 2e-3
    parser.add_argument('-reg', '--weight_decay', type=float, default=1e-6,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=300,  # cmu 300  na 600
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--early_stopping_rounds', type=int, default=30,  # rounds=10,more rounds maybe lead higher acc
                        help='early_stopping_rounds  ')
    parser.add_argument('-d', '--dir', metavar='str', type=str, default='../data/cmu/',
                        help='home directory')
    parser.add_argument('-enc', '--encoding', metavar='str', type=str, default='latin1',
                        help='Data Encoding (e.g. latin1, utf-8)')
    parser.add_argument('-cel', '--celebrity', metavar='int', type=int, default=5,  # cmu 5    na 15
                        help='celebrity threshold')
    parser.add_argument('-bucket', '--bucket', metavar='int', type=int, default=50,  # cmu 50   na 2400
                        help='discretisation bucket size')
    parser.add_argument('-mindf', '--mindf', metavar='int', type=int, default=10,
                        help='minimum document frequency in BoW')
    parser.add_argument('-maxdf', '--maxdf', metavar='int', type=int, default=0.2,
                        help='minimum document frequency in BoW')
    parser.add_argument('-builddata', action='store_true',
                        help='if exists do not reload dumped data, build it from scratch')

    args = parser.parse_args()
    return args

