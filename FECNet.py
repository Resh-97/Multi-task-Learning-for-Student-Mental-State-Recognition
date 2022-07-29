### imports
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from models.FECNet import FECNet
from utils.pytorchtools import EarlyStopping
from utils.data_prep import DATALoader , AffDataLoader
#from torch.utils.data import  DataLoader
from data.export_train_label import creat_label
from data.image_downloader import download_img
from facenet_pytorch import InceptionResnetV1
#from models.inception_resnet_v1 import InceptionResnetV1

### functions
def triplet_loss(y_pred):
    ref = y_pred[0::3, :]
    pos = y_pred[1::3, :]
    neg = y_pred[2::3, :]
    L12 = (ref - pos).pow(2).sum(1)
    L13 = (ref - neg).pow(2).sum(1)
    L23 = (pos - neg).pow(2).sum(1)
    correct = (L12 < L13) * (L12 < L23)

    delta = 0.2
    d1 = F.relu((L12 - L13) + delta)
    d2 = F.relu((L12 - L23) + delta)
    d = torch.mean(d1 + d2)
    return d, torch.sum(correct)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='PyTorch FECNet')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 240)')
    parser.add_argument('--epochs', type=int, default=13,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument('--nesterov', type=float, default=True,
                        help='nesterov (default: True)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='alpha (default: 0.1)')
    parser.add_argument('--val_ratio', type=float, default=0.01,
                        help='Ratio of number of Validation data.')
    parser.add_argument('--tr_ratio', type=float, default=1,
                        help='Ratio of number of train data.Default is 1.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.',default=4)
    parser.add_argument('--pretrained', dest='pretrained', type=bool,
                        help='Use pretrained weightts of FECNet.', default=False)
    args = parser.parse_args()


    # loading data
    if not os.path.exists('data/train'):
        os.makedirs('data/train', exist_ok=True)
        creat_label()
        download_img()

    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)



    #model = FECNet(pretrained=args.pretrained)

    #model = FECNet()

    ftr_extractor = InceptionResnetV1(pretrained='vggface2',classify=True)
    del ftr_extractor.last_bn
    del ftr_extractor.logits
    del ftr_extractor.repeat_3
    del ftr_extractor.block8
    del ftr_extractor.avgpool_1a
    del ftr_extractor.dropout
    del ftr_extractor.last_linear
    model = FECNet(ftr_extractor)
    model.cuda()
    Num_Param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of Trainable Parameters= %d" % (Num_Param))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum =args.momentum, nesterov=args.nesterov )
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    early_stopping = EarlyStopping(patience=50, verbose=True)

    running_loss = 0
    print_per_epoch = 1
    correct = 0
    Len = 0
    Len_aff = 0
    print("Beginning data loading.....")
    tr_dataloader, val_dataloader = DATALoader(csv_file='data/labels.csv', args=args)
    tr_loader, val_loader = AffDataLoader(csv_file='data/train_set/affectnet_train.csv', args=args)

    print("Data Loading is complete....")
    for epoch in range(args.epochs):
        # scheduler.step()

        # Training
        print("Total datapoints...{}".format(len(tr_dataloader)))
        count = 0
        tr_loader_iterator = iter(tr_loader)

        print("Outside")


        for i_batch, sample_batched in enumerate(tr_dataloader):

            try:
                sample_batched_aff = next(tr_loader_iterator)

            except StopIteration:

                tr_loader_iterator = iter(tr_loader)
                sample_batched_aff = next(tr_loader_iterator)

            #print("i_batch"+str(i_batch))
            #print(sample_batched.shape)
            #print(sample_batched)
            #print(sample_batched_aff[0].shape)
            #print(sample_batched_aff[0])

            count+=1
            model.zero_grad()

            e_fec = model(torch.FloatTensor(sample_batched).view(sample_batched.shape[0] * 3, 3, 140, 140).cuda())
            linear_fec = torch.nn.Linear(128, 32).cuda()
            targets = linear_fec(e_fec)
            #print("Targets "+ str(targets.shape))
            #print(targets)

            e_aff = model(torch.FloatTensor(sample_batched_aff[0]).view(sample_batched_aff[0].shape[0] * 1, 3, 140, 140).cuda())
            logits_aff = torch.nn.Linear(128, 8).cuda()
            class_pred = logits_aff(e_aff)
            #print("e_aff success...")
            #print("Class "+ str(class_pred.shape))
            #print(class_pred)

            print("Fetching targets.....{}".format(count))
            loss_fec, cor = triplet_loss(targets)

            Len += sample_batched.shape[0]
            Len_aff += sample_batched_aff[1].shape[0]
            correct += cor.detach().cpu().numpy()

            cross_entropy = torch.nn.CrossEntropyLoss().cuda()
            loss_aff = cross_entropy(class_pred, sample_batched_aff[1].view(sample_batched_aff[1].shape[0]).cuda())
            print("Class pred")
            print(torch.argmax(class_pred, dim=1))
            print("target value")
            print(sample_batched_aff[1].view(sample_batched_aff[1].shape[0]))
            acc_aff = (torch.argmax(class_pred, dim=1) - sample_batched_aff[1].view(sample_batched_aff[1].shape[0]).cuda())
            loss = loss_fec + args.alpha * loss_aff

            loss.backward()
            optimizer.step()

            running_loss += loss.detach().cpu().numpy()

        if epoch % print_per_epoch == print_per_epoch - 1:  # print every 1 mini-batches

            # Validation
            Len_val = 0
            correct_val = 0

            with torch.no_grad():
                running_loss_Valid = 0
                val_loader_iterator = iter(val_loader)
                for i_batch, sample_batched in enumerate(val_dataloader):

                    try:
                        sample_batched_aff = next(val_loader_iterator)

                    except StopIteration:
                        val_loader_iterator = iter(val_loader)
                        sample_batched_aff = next(val_loader_iterator)

                    e_fec = model(torch.FloatTensor(sample_batched).view(sample_batched.shape[0] * 3, 3, 140, 140).cuda())
                    linear_fec = torch.nn.Linear(128, 32).cuda()
                    targets = linear_fec(e_fec)

                    e_aff = model(torch.FloatTensor(sample_batched_aff[0]).view(sample_batched_aff[0].shape[0] * 1, 3, 140, 140).cuda())
                    logits_aff = torch.nn.Linear(128, 8).cuda()
                    class_pred = logits_aff(e_aff)
                    acc_aff_val = (torch.argmax(class_pred, dim=1) - sample_batched_aff[1].view(sample_batched_aff[1].shape[0]).cuda())

                    loss_fec, cor = triplet_loss(targets)
                    cross_entropy = torch.nn.CrossEntropyLoss().cuda()
                    loss_aff = cross_entropy(class_pred, sample_batched_aff[1].view(sample_batched_aff[1].shape[0]).cuda())
                    loss = loss_fec + args.alpha * loss_aff

                    Len_val += sample_batched.shape[0]
                    correct_val += cor.detach().cpu().numpy()
                    running_loss_Valid += loss.detach().cpu().numpy()
            print('['+str(epoch + 1)+', '+str(Len)+', '+str(Len_aff)+'] loss: '+str(running_loss / print_per_epoch)+'      Val_acc_fec: '+str(correct_val / Len_val)+'    Train_acc_fec: '+str(correct / Len)+'    Val_acc_aff: '+str(acc_aff)+'    Train_acc_aff: '+str(acc_aff_val))

            running_loss = 0
            Len = 0
            correct = 0

            ### Check early stopping
            early_stopping(float(running_loss_Valid), model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
