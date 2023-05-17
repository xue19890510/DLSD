from curses.panel import top_panel
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import argparse
from methods.stl_deepbdc import STLDeepBDC

from data.datamgrnoise import SimpleDataManagerSsl, SetDataManager
from methods.templatemarginbias import BaselineTrain
from utils import *
from data.mini_dataloader import DataManager
import tqdm as tqdm
global global_step
global_step = 0


model_dict = dict(
    ResNet10=resnet.ResNet10,
    ResNet12=resnet.ResNet12,
    ResNet18=resnet.ResNet18,
    ResNet34=resnet.ResNet34,
    ResNet34s=resnet.ResNet34s,
    ResNet50=resnet.ResNet50,
    ResNet101=resnet.ResNet101)
def update_ema_variables(model, ema_model, alpha=0.999, global_step=-1):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def update_new_variables(model, ema_model ):
    # Use the true average until the exponential average is more correct
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(0).add_(1, param.data)



class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


def train_distill(params, base_loader, val_loader, test_loader,model, model_t, model_ema,stop_epoch):   
    global global_step
    trlog = {}
    trlog['args'] = vars(params)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    if params.method in ['stl_deepbdc']:
        bas_params = filter(lambda p: id(p) != id(model.dcov.temperature), model.parameters())
        optimizer = torch.optim.SGD([
            {'params': bas_params}, 
            {'params': model.dcov.temperature, 'lr': params.t_lr}], lr=params.lr, weight_decay=5e-4, nesterov=True, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, weight_decay=5e-4, nesterov=True, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=params.milestones, gamma=params.gamma)

    # loss_fn = SmoothCrossEntropy(epsilon=0.1).cuda()
    loss_fn =  nn.CrossEntropyLoss()

    loss_div_fn = DistillKL(4)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    # novel_few_shot_params = dict(n_way=5, n_support=params.n_shot)
    # model_test  = STLDeepBDC(params, model_dict['ResNet12'], **novel_few_shot_params).cuda()
    # model_test_ema  = STLDeepBDC(params, model_dict['ResNet12'], **novel_few_shot_params).cuda()
    for epoch in range(0, stop_epoch):
        epoch_time = time.time()
        
        model.train()
        
        print_freq = 200
        avg_loss = 0
        avg_acc = 0
        start_time = time.time()
        for i, (x, x_noise,target) in enumerate(base_loader):           
            x = Variable(x.cuda())
            x_noise = Variable(x_noise.cuda())
            y,rot_y=target
            rot_y = Variable(rot_y.cuda())
            y = Variable(y.cuda())
            with torch.no_grad():
                scores_t= model_t(x,y)
                scores_ema = model_ema(x_noise,y)
                scores_sum = (scores_t.detach()+scores_ema.detach())/2
                # scores_sum_rot = (scores_t_rot.detach()+scores_ema_rot.detach())/2
                
            scores = model(x,y)
            loss_cls = loss_fn(scores, y)
            # loss_rot = loss_fn(scores_rot,rot_y)

            pred = scores.data.max(1)[1]
            train_acc = pred.eq(y.data.view_as(pred)).sum()
 
            loss_div = loss_div_fn(scores, scores_sum)
            # loss_div_rot = loss_div_fn(scores_rot, scores_sum_rot)
            # loss = (loss_cls+loss_rot) * 0.5 + (loss_div+loss_div_rot) * 0.5
            # loss = (loss_cls) * 0.5 + (loss_div) * 0.5     # born1 69.17 85.89  born2 68.67  85.54
            loss = (loss_cls) * 0.5 + (loss_div) * 0.5     # 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            avg_acc = avg_acc + train_acc.item()

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Time {:.2f}'.format(epoch, i, len(base_loader), avg_loss / float(i + 1), time.time()-start_time))
            start_time = time.time()
            update_ema_variables(model, model_ema, alpha=0.998,  global_step=global_step)
            global_step+=1

       
        # if (epoch %20==0 and epoch>90) or (epoch %40==0 and epoch>15) or (epoch>135):
        #     model_ema_dict_pa =  model_ema.state_dict()
        #     model_dict_pa =  model.state_dict()


        #     model_test_ema_dict_pa = model_test_ema.state_dict()
        #     model_test_dict_pa = model_test.state_dict()


        #     filter_dict_test_ema = {k: v for k, v in model_ema_dict_pa.items() if k in model_test_ema_dict_pa}
        #     filter_dict_test = {k: v for k, v in model_dict_pa.items() if k in model_test_dict_pa}

        #     model_test_ema_dict_pa.update(filter_dict_test_ema) 
        #     model_test_dict_pa.update(filter_dict_test) 

        #     model_test_ema.load_state_dict(model_test_ema_dict_pa)
        #     model_test.load_state_dict(model_test_dict_pa)

        #     # update_new_variables(model_ema,model_test_ema)
        #     # update_new_variables(model,model_test)
        #     model_test_ema.eval()
        #     model_test.eval()
        #     acc_all = []
        #     acc_all_ema = []
        #     test_start_time = time.time()
        #     iter_num = 2000#params.test_n_episode
        #     tqdm_gen = tqdm.tqdm(test_loader)
        #     for i , (images_train, labels_train, images_test, labels_test) in enumerate(tqdm_gen):
        #         with torch.no_grad():
        #             way=5
        #             x_train=images_train.squeeze().reshape(way,params.n_shot,3,params.image_size,params.image_size)
        #             x_test = images_test.squeeze().reshape(way,params.n_query,3,params.image_size,params.image_size)
        #             x = torch.cat((x_train,x_test),1)
        #             y=torch.cat((labels_train,labels_test),1)
        #             model.n_query = params.n_query
        #             scores1 = model_test.set_forward(x, False)
        #             scores_ema = model_test_ema.set_forward(x, False)
        #         pred1 = scores1
        #         pred_ema = scores_ema
        #         y = np.repeat(range(params.test_n_way), params.n_query)
        #         acc = np.mean(pred1 == y) * 100
        #         acc_ema = np.mean(pred_ema == y) * 100

        #         acc_all.append(acc)
        #         acc_all_ema.append(acc_ema)
        #         # tqdm_gen.set_description(f'avg.acc:{(np.mean(acc_all)):.2f} (curr:{acc:.2f})')
        #         # tqdm_gen.set_description(f'avg.acc_ema:{(np.mean(acc_all_ema)):.2f} (curr:{acc_ema:.2f})')

        #     acc_all = np.asarray(acc_all)
        #     acc_mean = np.mean(acc_all)
        #     acc_all_ema = np.asarray(acc_all_ema)
        #     acc_mean_ema = np.mean(acc_all_ema)
        #     acc_std = np.std(acc_all)
        #     print('%d Test Acc = %4.2f%% +- %4.2f%% (Time uses %.2f minutes)'
        #         % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num), (time.time() - test_start_time) / 60))
        #     print('%d Test ema Acc = %4.2f%% +- %4.2f%% (Time uses %.2f minutes)'
        #         % (iter_num, acc_mean_ema, 1.96 * acc_std / np.sqrt(iter_num), (time.time() - test_start_time) / 60))

        #     if acc_mean_ema>trlog['max_acc']:
        #         trlog['max_acc'] = acc_mean_ema
        #         trlog['max_acc_epoch'] = epoch
        #         outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
        #         torch.save({'epoch': epoch, 'state': model_ema.state_dict()}, outfile)



        if epoch == params.save_freq or (epoch%10==0 and epoch>115) or epoch==0:
            outfile_ema = os.path.join(params.checkpoint_dir,'ema_{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model_ema.state_dict()}, outfile_ema)
            # outfile = os.path.join(params.checkpoint_dir,"ema_smooth_ssl" ,'org_{:d}.tar'.format(epoch))
            # torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        
        if epoch == stop_epoch - 1:
            outfile_ema = os.path.join(params.checkpoint_dir, 'ema_last_model.tar')
            torch.save({'epoch': epoch, 'state': model_ema.state_dict()}, outfile_ema)
            # outfile = os.path.join(params.checkpoint_dir, "ema_smooth_ssl" ,'ori_last_model.tar')
            # torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)


        
        # print("best acc_ema is {:.2f}, best epoch is  {:.2f}".format(trlog['max_acc'],  trlog['max_acc_epoch']))
        

        avg_acc = avg_acc / (len(base_loader)*params.batch_size) * 100
        avg_loss = avg_loss / len(base_loader)

        trlog['train_loss'].append(avg_loss)
        trlog['train_acc'].append(avg_acc)

        torch.save(trlog, os.path.join(params.checkpoint_dir, 'trlog'))

        lr_scheduler.step()  # lr decreased
    
        print('1 epoch use {:.2f}mins '.format((time.time() - epoch_time)/60))
        print("train loss is {:.2f}, train acc is {:.2f}".format(avg_loss, avg_acc))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', default=84, type=int, choices=[84, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
    parser.add_argument('--batch_size', default=64, type=int, help='pre-training batch size')
    parser.add_argument('--lr', type=float, default=0.05, help='initial learning rate of the backbone')
    parser.add_argument('--t_lr', type=float, default=0.05, help='initial learning rate uesd for the temperature of bdc module')

    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--milestones', nargs='+', type=int, default=[80, 120], help='milestones for MultiStepLR')
    parser.add_argument('--epoch', default=180, type=int, help='stopping epoch')
    parser.add_argument('--gpu', default='0', help='gpu id')

    parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet','tiered_imagenet','cub'])
    parser.add_argument('--data_path', type=str, help='dataset path')

    parser.add_argument('--model', default='ResNet12', choices=['ResNet12', 'ResNet18'])
    parser.add_argument('--method', default='stl_deepbdc', choices=['stl_deepbdc', 'good_embed'])

    parser.add_argument('--val', default='meta', choices=['meta', 'last'], help='validation method')
    parser.add_argument('--val_n_episode', default=1000, type=int, help='number of episode in meta validation')
    parser.add_argument('--val_n_way', default=5, type=int, help='class num to classify in meta validation')
    parser.add_argument('--n_shot', default=1, type=int, help='number of labeled data in each class, same as n_support during meta validation')
    parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class during meta validation')

    parser.add_argument('--extra_dir', default='ema_distill_nossl_margin', help='record additional information')

    parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in training')
    parser.add_argument('--save_freq', default=50, type=int, help='saving model .pth file frequency')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    parser.add_argument('--teacher_path', default='', help='teacher model .tar file path')
    parser.add_argument('--reduce_dim', default=640, type=int, help='the output dimensions of BDC dimensionality reduction layer')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--penalty_C', type=str, default=0.1, help='logistic penalty 1shot 0.1     5shot 2')
    
    parser.add_argument('--test_n_way', default=5, type=int)

    params = parser.parse_args()

    num_gpu = set_gpu(params)
    set_seed(params.seed)

    if params.val == 'last':
        val_file = None
    elif params.val == 'meta':
        val_file = 'val'

    json_file_read = False
    if params.dataset == 'mini_imagenet':
        base_file = 'train'
        params.num_classes = 64
    elif params.dataset == 'cub':
        base_file = 'base.json'
        val_file = 'val.json'
        json_file_read = True
        params.num_classes = 200
    elif params.dataset == 'tiered_imagenet':
        base_file = 'train'
        params.num_classes = 351
    else:
        ValueError('dataset error')


    base_datamgr = SimpleDataManagerSsl(params.data_path, params.image_size, batch_size=params.batch_size, json_read=json_file_read)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)


    dm = DataManager(params)
    trainloader,valloader, testloader = dm.return_dataloaders()

    # if params.val == 'meta':
    #     test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
    #     val_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, json_read=json_file_read, **test_few_shot_params)
    #     val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    # else:
    #     val_loader = None

    model = BaselineTrain(params, model_dict[params.model], params.num_classes)
    model_t = BaselineTrain(params, model_dict[params.model], params.num_classes)
    model_ema = BaselineTrain(params, model_dict[params.model], params.num_classes)

    model = model.cuda()
    model_t = model_t.cuda()
    model_ema = model_ema.cuda()

    for name, child in model_ema.named_children():
        for param in child.parameters():
            param.requires_grad = False

    for name, child in model_t.named_children():
        for param in child.parameters():
            param.requires_grad = False

    # model save path
    params.checkpoint_dir = './checkpoints/%s/%s_%s' % (params.dataset, params.model, params.method)
    params.checkpoint_dir += '_distill'
    params.checkpoint_dir += '_born{}/'.format(params.trial)
    params.checkpoint_dir += params.extra_dir
    print(params.checkpoint_dir)
    print(params)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # teacher model load
    modelfile = os.path.join(params.teacher_path)
    tmp = torch.load(modelfile)
    state = tmp['state']
    model_t.load_state_dict(state)

    

    model = train_distill(params, base_loader, valloader,testloader, model, model_t, model_ema,params.epoch)
