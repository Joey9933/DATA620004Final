import argparse
import os

parser = argparse.ArgumentParser(description="PyTorch SimCLR")

def parse_common_args(parser):
    parser.add_argument('-m','--model-name', type=str, default='ResNet', help='ResNet for supervised,SimCLR for selfsupervised',choices=['ResNet','SimCLR'])
    parser.add_argument('-dataset-name', default='imagenet200', help='dataset name',choices = ['cifar100','imagenet200'])
    # parser.add_argument('--save_prefix', type=str, default='pref', help='some comment for model or test result dir')
    # parser.add_argument('--load_model_path', type=str, default='checkpoints/base_model_pref/0.pth',
                        # help='model path for pretrain or test')
    # parser.add_argument('--load_not_strict', action='store_true', help='allow to load only common state dicts')
    # parser.add_argument('--val_list', type=str, default='/data/dataset1/list/base/val.txt',
                        # help='val list in train, test list path in test')
    # parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--out_dim',default=100, type=int, help="feature dimension (100 for CIFAR100 or 200 for Imagenet200)")
    parser.add_argument("--gpu-index", default=0, type=int, help="Gpu index.")
    parser.add_argument("--arg-dir",type=str,default='args',help='dir to save the args')
    return parser


def parse_train_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--lr','--learning-rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay','--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--model_dir', type=str, default='ckp/', help='dir to save the model ckps')
    parser.add_argument('--trainer', type=str, default='SelfSupervised',choices=['SelfSupervised','Supervised'])
    parser.add_argument('-b','--batch_size', type=int, default=2560)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.1,help='softmax temperature')
    parser.add_argument('--n-views',default=2,type=int,help='Number of views for contrastive learning training')
    parser.add_argument('--data_proportion',default=1,type=int,help='Number of data used for training. eg:10 means 1/10')
    return parser


def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--save_viz', action='store_true', help='save viz result in eval or not')
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')
    return parser


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    print(args)
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args


def get_train_model_dir(args):
    model_dir = os.path.join('checkpoints', args.model_type + '_' + args.save_prefix)
    if not os.path.exists(model_dir):
        os.system('mkdir -p ' + model_dir)
    args.model_dir = model_dir


# def get_test_result_dir(args):
#     ext = os.path.basename(args.load_model_path).split('.')[-1]
#     model_dir = args.load_model_path.replace(ext, '')
#     val_info = os.path.basename(os.path.dirname(args.val_list)) + '_' + os.path.basename(args.val_list.replace('.txt', ''))
#     result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
#     if not os.path.exists(result_dir):
#         os.system('mkdir -p ' + result_dir)
#     args.result_dir = result_dir


def save_args(args, save_dir):
    args_path = os.path.join(save_dir, '{}_args.txt'.format(args.model_name))
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def prepare_train_args():
    args = get_train_args()
    # get_train_model_dir(args)
    save_args(args, args.arg_dir)
    return args


# def prepare_test_args():
#     args = get_test_args()
#     get_test_result_dir(args)
#     save_args(args, args.result_dir)
#     return args


if __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args()