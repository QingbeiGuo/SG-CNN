#!/usr/bin/python
# -*- coding: UTF-8 -*-

#pytorch -0.2.1
#python -3.6.2
#torchvision - 0.1.9

import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import os
import math
from sklearn.cluster import KMeans  

from models.densenet.densenet import densenet201
import dataset.dataset_imagenet
#import kmeans

##############################################################################################################
class GroupingFineTuner_DenseNet:
	def __init__(self, train_path, test_path, model):
	    self.args = args
	    self.learningrate = self.args.learning_rate
	    self.learning_rate_decay = self.args.learning_rate_decay
	    self.weight_decay = self.args.weight_decay
	    self.train_path = self.args.train_path
	    self.test_path = self.args.test_path
	    self.compress_percentage = np.arange(10,100,10)        #self.args.compress_percentage  #compress_percentage
	    self.retrain_num = self.args.retrain_num                  #retrain_num
	    self.final_retrain_num = self.args.final_retrain_num      #final_retrain_num
	    self.group_num = self.args.group_num                      #group_num

	    self.train_data_loader = dataset.dataset_imagenet.train_loader(self.train_path)
	    self.test_data_loader = dataset.dataset_imagenet.test_loader(self.test_path)

	    self.model = model
	    self.criterion = torch.nn.CrossEntropyLoss()

	    self.accuracys1 = []
	    self.accuracys5 = []

	    self.criterion.cuda()
	    self.model.cuda()

	    for param in self.model.parameters():
	        param.requires_grad = True

	    self.model.train()

##############################################################################################################
	def group(self):
        #for a pre-training deep model
		#self.test(0)

		if os.path.isfile("i_group_a70") and os.path.isfile("model_group_a70"):
		    self.model = torch.load("model_group_a70")
		    print("model_group resume:", self.model)
		    i_ = torch.load("i_group_a70")
		    print("i resume:", i_)

		    self.accuracys1 = torch.load("accuracys1_group_a70")
		    self.accuracys5 = torch.load("accuracys5_group_a70")
		    print("accuracys1_group resume:", self.accuracys1)
		    print("accuracys5_group resume:", self.accuracys5)
		else:
		    i_ = 0

        #for a recovering deep model
		#self.test(0)

		print("self.total_num_maskedconvlayers()", self.total_num_maskedconvlayers())
		print("self.total_num_maskedlinearlayers()", self.total_num_maskedlinearlayers())
		for i,cp in enumerate(self.compress_percentage):
		    print("i, cp", i, cp)
		    if (i <= i_) and (i_ >= 0):
		        cp_ = cp
		        print("cp_", cp_)
		        continue

		    self.model = torch.load("model_training_" + str(cp_))
		    self.test(0)

		    for j in list(range(self.total_num_maskedconvlayers())):
		        print("j-conv", j)
		        centroids, centroids_bin, data_to_centroids = self.cluster_Euclidean_distance(i, j, self.group_num, cp, ltype = 0)
		        #mask pruning
		        self.prune(j, np.array(centroids_bin), np.array(data_to_centroids), ltype = 0)
		        #self.test(0)

		    if cp <= 60:
		        for j in list(range(self.total_num_maskedlinearlayers())):
		            print("j-fc", j)
		            centroids, centroids_bin, data_to_centroids = self.cluster_Euclidean_distance(i, j, self.group_num, cp, ltype = 1)
		            #mask pruning
		            self.prune(j, np.array(centroids_bin), np.array(data_to_centroids), ltype = 1)
		            #self.test(0)

		    self.accuracys1 = []
		    self.accuracys5 = []

		    self.test()

		    torch.save(i, "i_group_b" + str(cp))
		    torch.save(self.model, "model_group_b" + str(cp))
		    torch.save(self.accuracys1, "accuracys1_group_b" + str(cp))
		    torch.save(self.accuracys5, "accuracys5_group_b" + str(cp))

        #retraining
		    self.retrain(epoches=self.retrain_num, cp = cp)

		    torch.save(i, "i_group_a" + str(cp))
		    torch.save(self.model, "model_group_a" + str(cp))
		    torch.save(self.accuracys1, "accuracys1_group_a" + str(cp))
		    torch.save(self.accuracys5, "accuracys5_group_a" + str(cp))

		    cp_ = cp
		    print("cp_", cp_)
#		#retraining
#		self.retrain(epoches=self.final_retrain_num)
#
#		torch.save(self.model, "model_group")
#		torch.save(self.accuracys1, "accuracys1_group")
#		torch.save(self.accuracys5, "accuracys5_group")

##############################################################################################################
	def finetune(self):
         #for a pre-training deep model
		#self.test(0)

		for i in [80, 70]:
		    if os.path.isfile("model_group_b" + str(i)):
		        self.model = torch.load("model_group_b" + str(i))
		        self.model = torch.nn.DataParallel(self.model.module).cuda()
		        print("model_group resume:", self.model)

		        self.accuracys1 = []
		        self.accuracys5 = []

		        print("i resume:", i)

            #for a recovering deep model
		        self.test(0)

		        #retraining
		        self.retrain(epoches=self.retrain_num, cp = i)

		        torch.save(i, "i_group")
		        torch.save(self.model, "model_group")
		        torch.save(self.accuracys1, "accuracys1_group" + str(i))
		        torch.save(self.accuracys5, "accuracys5_group" + str(i))

##############################################################################################################
	def train(self, epoches = -1, batches = -1):
		self.test()

		if os.path.isfile("layeres_training") and os.path.isfile("model_training"):
		    self.model = torch.load("model_training")
		    print("model_training resume:", self.model)
		    list_layeres = torch.load("layeres_training")
		    print("list_layeres resume:", list_layeres)

		    self.accuracys1 = torch.load("accuracys1_trainning")
		    self.accuracys5 = torch.load("accuracys5_trainning")
		    print("accuracys1_trainning resume:", self.accuracys1)
		    print("accuracys5_trainning resume:", self.accuracys5)
		else:
		    list_layeres = list(range(epoches))

		list_ = list_layeres[:]
		for i in list_layeres[:]:
		    print("Epoch: ", i)

		    optimizer = optim.SGD(self.model.parameters(), lr=self.learningrate, momentum=0.9, weight_decay=self.weight_decay)
		    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

		    #self.train_epoch(i, batches, optimizer, scheduler)
		    self.train_epoch(i, batches, optimizer)
		    self.test()

		    self.adjust_learning_rate(i)

		    torch.save(list_, "layeres_training")
		    torch.save(self.model, "model_training")
		    torch.save(self.accuracys1, "accuracys1_trainning")
		    torch.save(self.accuracys5, "accuracys5_trainning")

##############################################################################################################
	def total_num_maskedconvlayers(self):
		convlayers = 0
		#for module in self.model.module.features.children():
		for module in self.model.modules():
		    if module.__str__().startswith('MaskedConv2d'):
		        convlayers = convlayers + 1
		return convlayers

	def importance_vector_layer_filters(self, layer_index):
		convlayers = 0
		#for module in self.model.module.features.children():
		for module in self.model.modules():
		    if module.__str__().startswith('MaskedConv2d'):
		        if convlayers == layer_index:
		            weight_size = module.weight.size()  #[out_channels, in_channels, W, H]

		            for i in range(weight_size[0]):
		                #masks = module._mask[i,:,:,:].data
		                #print("masks",masks)
		                #kernels = module.weight[i,:,:,:].data
		                #print("kernels",kernels)

		                masked_kernels = module.weight[i,:,:,:].data * module._mask[i,:,:,:].data
		                #print("masked_kernels",masked_kernels)

		                importance_vector = torch.sum(torch.sum(torch.abs(masked_kernels[:,:,:]), dim=1), dim=1)
		                #print("importance_vector",importance_vector)
		                if i == 0:
		                    #normal_vector = self.normalization(importance_vector)
		                    normal_vector = importance_vector
		                else:
		                    #normal_vector = torch.cat((normal_vector,self.normalization(importance_vector)), 0) 
		                    normal_vector = torch.cat((normal_vector,importance_vector), 0) 
		                #print("normal_vector",normal_vector)

		            #print("normal_vector",normal_vector)
		            #print("normal_vector size",normal_vector.size())

		            normal_vectors = normal_vector.view(weight_size[0], -1)				    
		            #print("normal_matrix",normal_matrix)
		            #print("normal_matrix size",normal_matrix.size())

		        convlayers = convlayers + 1
		return normal_vectors

	def mask_layer_filters(self, layer_index):
		convlayers = 0
		#for module in self.model.module.features.children():
		for module in self.model.modules():
		    if module.__str__().startswith('MaskedConv2d'):
		        if convlayers == layer_index:
		            mask = torch.sum(torch.sum(module._mask[:,:,:,:].data, 2), 2).cpu()/(module.kernel_size[0]*module.kernel_size[1])  #[out_channels, in_channels, W, H]
		            #print("mask",mask)

		        convlayers = convlayers + 1
		return mask

##############################################################################################################
	def total_num_maskedlinearlayers(self):
		linearlayers = 0
		#for module in self.model.module.classifier.children():
		for module in self.model.modules():
		    if module.__str__().startswith('MaskedLinear'):
		        linearlayers = linearlayers + 1
		return linearlayers

	def importance_vector_layer_connects(self, layer_index):
		linearlayers = 0
		#for module in self.model.module.classifier.children():
		for module in self.model.modules():
		    if module.__str__().startswith('MaskedLinear'):
		        if linearlayers == layer_index:
		            weight_size = module.weight.size()  #[out_features, in_features]

		            for i in range(weight_size[0]):
		                #masks = module._mask[i,:].data
		                #print("masks",masks)
		                #kernels = module.weight[i,:].data
		                #print("kernels",kernels)

		                masked_kernels = module.weight[i,:].data * module._mask[i,:].data
		                #print("masked_kernels",masked_kernels)

		                importance_vector = torch.abs(masked_kernels[:])
		                #print("importance_vector",importance_vector)
		                if i == 0:
		                    #normal_vector = self.normalization(importance_vector)
		                    normal_vector = importance_vector
		                else:
		                    #normal_vector = torch.cat((normal_vector,self.normalization(importance_vector)), 0) 
		                    normal_vector = torch.cat((normal_vector,importance_vector), 0) 
		                #print("normal_vector",normal_vector)

		            #print("normal_vector",normal_vector)
		            #print("normal_vector size",normal_vector.size())

		            normal_vectors = normal_vector.view(weight_size[0], -1)				    
		            #print("normal_matrix",normal_matrix)
		            #print("normal_matrix size",normal_matrix.size())

		        linearlayers = linearlayers + 1
		return normal_vectors

	def mask_layer_connects(self, layer_index):
		linearlayers = 0
		#for module in self.model.module.classifier.children():
		for module in self.model.modules():
		    if module.__str__().startswith('MaskedLinear'):
		        if linearlayers == layer_index:
		            mask = module._mask[:,:].data.cpu()  #[out_features, in_features]
		            #print("mask",mask)

		        linearlayers = linearlayers + 1
		return mask

##############################################################################################################
	def normalization(self, importance_vector):
		return importance_vector / importance_vector.sum()

	def cluster_Euclidean_distance(self, compress_index, layer_index, num_clusters, compress_percentage, ltype = 0):
		if ltype == 0:
		    importance_vectors = self.importance_vector_layer_filters(layer_index)
		else:
		    importance_vectors = self.importance_vector_layer_connects(layer_index)
		#print("importance_vectors", importance_vectors)
		#print("importance_vectors size", len(importance_vectors))

		estimator = KMeans(n_clusters=num_clusters, max_iter=300, n_init=10, init = 'random').fit(importance_vectors.cpu())
		centroids = estimator.cluster_centers_.tolist()
		data_to_centroids = estimator.labels_.tolist()
		'''kme = kmeans.kmeans_euclid(importance_vectors,importance_vectors[:num_clusters],0)
		centroids = kme[0]    #[[], [], [], ...]
		data_to_centroids = kme[1]    #[]'''
		print("centroids",centroids)
		print("data_to_centroids",data_to_centroids)

		cluster_count = [(lambda inx:data_to_centroids.count(inx))(inx) for inx, c in enumerate(centroids)]
		print("cluster_count",cluster_count)

		if ltype == 0:
		    mask_vectors = self.mask_layer_filters(layer_index)
		else:
		    mask_vectors = self.mask_layer_connects(layer_index)
		print("mask_vectors",mask_vectors)

		centroids_flatten = []
		for inx, d in enumerate(data_to_centroids):
		    centroids_flatten.extend(np.multiply(np.array(centroids[d]),np.array(mask_vectors[inx,:])).tolist())
		#print("centroids_flatten",centroids_flatten)

		threshold = np.percentile(centroids_flatten, compress_percentage)
		print("threshold", threshold)

		centroids_bin = []
		for inx, c in enumerate(centroids):
		    if inx in data_to_centroids:
		        c = np.multiply(np.array(c),np.array(mask_vectors[data_to_centroids.index(inx),:])).tolist()
		        #print("c", c)
		        centroids_bin.append(list(np.where(np.array(c) > threshold, 1, 0)))
		    else:
		        centroids_bin.append([])
		print("centroids_bin",centroids_bin)

		centroids_bin_count = [(lambda c:c.count(0))(c) for c in centroids_bin]
		print("centroids_bin_count",centroids_bin_count)
		print("cluster_count",cluster_count)
		print("centroids_bin_perc",np.multiply(np.array(cluster_count),np.array(centroids_bin_count)).sum()/len(centroids_flatten))

		return centroids, centroids_bin, data_to_centroids

##############################################################################################################
	def prune(self, layer_index, centroids_bin, data_to_centroids, ltype = 0):
		if ltype == 0:
		    self.model.module.set_conv_mask(layer_index, centroids_bin, data_to_centroids)
		else:
		    self.model.module.set_linear_mask(layer_index, centroids_bin, data_to_centroids)

##############################################################################################################
	def retrain(self, epoches = -1, batches = -1, cp = -1):
		self.learningrate = self.args.learning_rate

		accuracy = 0
		for i in range(epoches):
			print("Epoch: ", i)

			optimizer = optim.SGD(self.model.parameters(), lr=self.learningrate, momentum=0.9, weight_decay=self.weight_decay)
			#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

			#self.train_epoch(i, batches, optimizer, scheduler)
			self.train_epoch(i, batches, optimizer)
			cor1, cor5 = self.test()
			#save the best model
			if cor1 > accuracy:
			    torch.save(self.model, "model_training_" + str(cp))
			    accuracy = cor1

			torch.save(i, "i_group_ing")
			torch.save(self.model, "model_group_ing")
			torch.save(self.accuracys1, "accuracys1_group_ing")
			torch.save(self.accuracys5, "accuracys5_group_ing")

			self.adjust_learning_rate(i)

	def train_epoch(self, epoch, batches, optimizer = None, scheduler = None):
		for step, (batch, label) in enumerate(self.train_data_loader):
		    if (step == batches):
		        break
		    self.train_batch(epoch, step, batch, label, optimizer, scheduler)

	def train_batch(self, epoch, step, batch, label, optimizer = None, scheduler = None):
        ### Compute output
		batch,label = Variable(batch.cuda()),Variable(label.cuda())                   #Tensor->Variable
		output = self.model(batch)
		loss_ = self.criterion(output, label)

        ### Add group lasso loss
		lasso_loss = 0.0
		if args.group_lasso_lambda > 0:
		    for module in self.model.modules():
		    #for module in self.model.features.children():
		        if module.__str__().startswith('MaskedConv2d'):
		            #weight = module.weight * module._mask
		            #weight = weight.pow(2)
		            weight = module.weight.pow(2)

		            lasso_loss = lasso_loss + weight.sum(1).sum(1).sum(1).sqrt().sum()           

		    for module in self.model.modules():
		    #for module in self.model.classifier.children():
		        if module.__str__().startswith('MaskedLinear'):
		            #weight = module.weight * module._mask
		            #weight = weight.pow(2)
		            weight = module.weight.pow(2)

		            lasso_loss = lasso_loss + weight.sum(1).sqrt().sum()           

		loss = loss_ + args.group_lasso_lambda * lasso_loss

		if step % self.args.print_freq == 0:
		    #print("Epoch-step: ", epoch, "-", step, ":", loss.data.cpu().numpy(), ",", loss_.data.cpu().numpy(), lasso_loss.data.cpu().numpy())
		    print("Epoch-step: ", epoch, "-", step, ":", loss.data.cpu().numpy(), ",", loss_.data.cpu().numpy())

        ### Compute gradient and do SGD step
		self.model.zero_grad()
		loss.backward()
		optimizer.step()                                                              #update parameters
		#scheduler.step(loss)                                                              #update parameters

	def test(self, flag = -1):
		self.model.eval()

		#correct = 0
		correct1 = 0
		correct5 = 0
		total = 0

		print("Testing...")
		for i, (batch, label) in enumerate(self.test_data_loader):
			  batch,label = Variable(batch.cuda()),Variable(label.cuda())              #Tensor->Variable
			  output = self.model(batch)
			  #pred = output.data.max(1)[1]
			  #correct += pred.cpu().eq(label).sum()
			  cor1, cor5 = accuracy(output.data, label, topk=(1, 5))                   # measure accuracy top1 and top5
			  correct1 += cor1
			  correct5 += cor5
			  total += label.size(0)

		if flag == -1:
		    self.accuracys1.append(float(correct1) / total)
		    self.accuracys5.append(float(correct5) / total)

		print("learningrate", self.learningrate)
		print("Accuracy Top1:", float(correct1) / total)
		print("Accuracy Top5:", float(correct5) / total)

		self.model.train()                                                              

		return float(correct1) / total, float(correct5) / total

	def adjust_learning_rate(self, epoch):
        #manually
		if self.args.learning_rate_decay == 0:
		    if epoch in [30, 60]:
		    #if epoch in []:
		        self.learningrate = self.learningrate/10;
        #exponentially
		elif self.args.learning_rate_decay == 1:
		    num_epochs = 100
		    lr_start = 0.1
		    #print("lr_start = "+str(self.lr_start))
		    lr_fin = 0.001
		    #print("lr_fin = "+str(self.lr_fin))
		    lr_decay = (lr_fin/lr_start)**(1./num_epochs)
		    #print("lr_decay = "+str(self.lr_decay))

		    self.learningrate = self.learningrate * lr_decay

def accuracy(output, target, topk=(1,)):                                               
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)                                          
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))                               

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)                                 
        res.append(correct_k)
    return res

##############################################################################################################
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--arch', '--a', default='DenseNet', help='model architecture: (default: DenseNet)')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run')
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.01, help = 'initial learning rate')
    parser.add_argument('--learning_rate_decay', '--lr_decay', type=int, default=0, help = 'maually[0] or exponentially[1] decaying learning rate')
    parser.add_argument('--weight_decay', '--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument('--group_num', '--group_num', type=int, default=16, help = 'group number')
    parser.add_argument('--retrain_num', '--retrain_num', type=int, default=90, help = 'retraining number')
    parser.add_argument('--final_retrain_num', '--final_retrain_num', type=int, default=0, help = 'final retraining number')
    parser.add_argument('--compress_percentage', '--compress_percentage', type=int, default=10, help = 'compress percentage')
    #parser.add_argument('--group_lasso_lambda', '--group_lasso_lambda', type=float, default=1e-4, help = 'group lasso lambda')
    parser.add_argument('--group_lasso_lambda', '--group_lasso_lambda', type=float, default=0, help = 'group lasso lambda')
    parser.add_argument('--print_freq', '--p', type=int, default=20, help = 'print frequency (default:20)')
    parser.add_argument('--train_path',type=str, default='/data/qingbeiguo/work/Datasets/ImageNet/ILSVRC2012/ILSVRC2012_img_train/', help = 'train dataset path')
    parser.add_argument('--test_path', type=str, default='/data/qingbeiguo/work/Datasets/ImageNet/ILSVRC2012/ILSVRC2012_img_val_subfolders/', help = 'test dataset path')
    parser.set_defaults(train=False)
    parser.set_defaults(group=False)
    parser.set_defaults(finetune=True)
    args = parser.parse_args()

    return args

##############################################################################################################
if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    args = get_args()
    print("args:", args)

    model = models.densenet201(pretrained=True)
    print("model_training:", model)
    torch.save(model.state_dict(), "model_densenet201.pth")

#    model = torch.load("model_training_70").module
#    print("model_training:", model)
#    torch.save(model.state_dict(), "model_training_70.pth")

    '''densenet = models.densenet201(pretrained=True).cuda()
    print("densenet:", densenet)
    model = densenet201(pretrained=False)
    print("densenet201:", model)

    pretrained_dict = densenet.state_dict()
    #print("pretrained_dict", pretrained_dict)
    model_dict = model.state_dict()

    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    #print("model_dict", model_dict)
    print("model_training:", model)
    torch.save(model, "model")'''

#    model = densenet201(pretrained=False)
#    model = torch.load("model").cuda()
#    #print("model_training:", model)
#
#    model = torch.nn.DataParallel(model).cuda()
#
#    fine_tuner = GroupingFineTuner_DenseNet(args.train_path, args.test_path, model)
#
#    if args.train:
#        fine_tuner.train(epoches = args.epochs)
#        torch.save(model, "model_training_final")
#        print("model_training_final:", model)
#
#    elif args.group:
#        fine_tuner.group()
#        torch.save(fine_tuner.model, "model_group_final")
#        print("model_group_final:", fine_tuner.model)
#
#    elif args.finetune:
#        fine_tuner.finetune()
