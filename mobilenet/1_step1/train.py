import argparse

parser = argparse.ArgumentParser(description='ReActNet')
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=256, help='number of training epochs')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

args = parser.parse_args()
print("args.data: {}".format(args.data))
print("args.batch_size: {}".format(args.batch_size))
print("args.learning_rate: {}".format(args.learning_rate))
print("args.epochs: {}".format(args.epochs))
print("args.weight_decay: {}".format(args.weight_decay))

# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
# parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
# parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
# parser.add_argument('--teacher', type=str, default='resnet34', help='path of ImageNet')
# parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')