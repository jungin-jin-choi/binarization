import argparse
import torchvision.models as models
import torch.nn as nn

from reactnet import reactnet

parser = argparse.ArgumentParser(description='ReActNet')
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=256, help='number of training epochs')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--teacher', type=str, default='resnet18', help='teacher model')
args = parser.parse_args()

# tee: log/training.text: No such file or directory
# [ TO-DO ] create log directory if not exists

def main():
    print("args.data: {}".format(args.data))
    print("args.batch_size: {}".format(args.batch_size))
    print("args.learning_rate: {}".format(args.learning_rate))
    print("args.epochs: {}".format(args.epochs))
    print("args.weight_decay: {}".format(args.weight_decay))
    print("args.teacher: {}".format(args.teacher))

    model_teacher = models.__dict__[args.teacher](pretrained=True)
    model_teacher = nn.DataParallel(model_teacher).cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.eval()

    model_student = reactnet()
    model_student = nn.DataParallel(model_student).cuda()

    # define loss function
    # define optimizer
    # define scheduler
    
    # [ 2 ] load checkpoint
    # if checkpoint exists, load checkpoint,
    #                       set start_epoch to the startpoint of checkpoint,
    #                       step scheduler - adjust the learning rate according to the checkpoint

    # [ 3 ] load training data, validation data
    # define normalization 
    # define transformation
    # load imageset data from train folder and valid folder, apply transformation

    # [ 4 ] train the model
    # until args.epochs
    #       train(), validate()
    #       if val_acc > best_acc: update best_acc
    #       save checkpoint, record whether it is best_acc (set flag)
    # measure training time

if __name__ == '__main__':
    main()