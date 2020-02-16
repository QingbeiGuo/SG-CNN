import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

from collections import OrderedDict
import numpy as np
import os

from models.densenet.densenet import densenet121

def summary(model, input_size, batch_size=-1, device="cpu"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                if isinstance(module, nn.Conv2d) and module.__str__().startswith('MaskedConv2d'):
                    params1 = 0
                    mask = torch.sum(torch.sum(module._mask[:,:,:,:].data, 2), 2).cpu()/(module.kernel_size[0]*module.kernel_size[1])  #[out_channels, in_channels, W, H]
                    #print("mask conv",mask)
                    out_channels, in_channels, W, H = module.weight.size()  #[out_channels, in_channels, W, H]
                    #print("out_channels, in_channels, W, H conv", out_channels, in_channels, W, H)

                    params1 += mask.sum().numpy()*W*H
                    #params += params1
                    #print("params weight conv", params1)

                    #BN
                    mask = torch.sum(torch.sum(module._mask[:,:,:,:].data, 2), 2).cpu()/(module.kernel_size[0]*module.kernel_size[1])  #[out_channels, in_channels, W, H]
                    mask = torch.sum(mask, 1).cpu().numpy()  #[out_channels]
                    #print("mask bias",mask)

                    params1 += np.sum(mask>0)*2
                    params += params1
                    #print("params bias", params1)
                    
                elif isinstance(module, nn.Conv2d) and (not module.__str__().startswith('MaskedConv2d')):
                    params1 = 0
                    params1 += torch.prod(torch.LongTensor(list(module.weight.size()))).numpy()
                    params += params1
                    #print("params conv", params1)
                    
                elif isinstance(module, nn.Linear):
                    params1 = 0
                    out_channels, in_channels = module.weight.size()  #[out_channels, in_channels]
                    #print("out_channels, in_channels linear", out_channels, in_channels)

                    params1 = out_channels * in_channels
                    params += params1
                    #print("params weight linear", params1)
    
                summary[m_key]["trainable"] = module.weight.requires_grad

            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                if isinstance(module, nn.Conv2d) and module.__str__().startswith('MaskedConv2d'):
                    params1 = 0
                    mask = torch.sum(torch.sum(module._mask[:,:,:,:].data, 2), 2).cpu()/(module.kernel_size[0]*module.kernel_size[1])  #[out_channels, in_channels, W, H]
                    mask = torch.sum(mask, 1).cpu().numpy()  #[out_channels]
                    #print("mask bias",mask)

                    params1 += np.sum(mask>0)
                    params += params1
                    #print("params bias", params1)
                elif isinstance(module, nn.Conv2d) and (not module.__str__().startswith('MaskedConv2d')):
                    params1 = 0
                    params1 += torch.prod(torch.LongTensor(list(module.bias.size()))).numpy()
                    params += params1
                    #print("params bias", params1)
                elif isinstance(module, nn.Linear):
                    params1 = 0
                    params1 += torch.prod(torch.LongTensor(list(module.bias.size()))).numpy()
                    params += params1

            summary[m_key]["nb_params"] = params

            flops = 0
            if hasattr(module, "weight") and module.weight is not None:
                if isinstance(module, nn.Conv2d) and module.__str__().startswith('MaskedConv2d'):
                    flops1 = 0
                    mask = torch.sum(torch.sum(module._mask[:,:,:,:].data, 2), 2).cpu()/(module.kernel_size[0]*module.kernel_size[1])  #[out_channels, in_channels, W, H]
                    mask = torch.sum(mask, 1).cpu().numpy()  #[out_channels]
                    #print("mask conv",mask)
                    _, _, output_height, output_width = output.size()
                    #print("output_height, output_width conv", output_height, output_width)
                    out_channels, in_channels, W, H = module.weight.size()  #[out_channels, in_channels, W, H]
                    #print("out_channels, in_channels, W, H conv", out_channels, in_channels, W, H)

                    for i in mask:
                        flops1 += output_height*output_width*i*W*H
                    #flops += flops1
                    #print("flops weight conv", flops1)

                    #BN
                    mask = torch.sum(torch.sum(module._mask[:,:,:,:].data, 2), 2).cpu()/(module.kernel_size[0]*module.kernel_size[1])  #[out_channels, in_channels, W, H]
                    mask = torch.sum(mask, 1).cpu().numpy()  #[out_channels, in_channels]
                    #print("mask bias",mask)

                    flops1 += np.sum(mask>0)
                    flops += flops1
                    #print("flops bias", flops1)

                elif isinstance(module, nn.Conv2d) and (not module.__str__().startswith('MaskedConv2d')):
                    flops1 = 0
                    _, _, output_height, output_width = output.size()
                    output_channel, input_channel, kernel_height, kernel_width = module.weight.size()
                    flops1 = output_channel * output_height * output_width * input_channel * kernel_height * kernel_width
                    flops += flops1
                    #print("flops conv", flops1)

                elif isinstance(module, nn.Linear):
                    flops1 = 0
                    out_channels, in_channels = module.weight.size()  #[out_channels, in_channels]
                    #print("out_channels, in_channels linear", out_channels, in_channels)

                    flops1 += out_channels * in_channels
                    flops += flops1
                    #print("flops weight linear", flops1)

                summary[m_key]['weight'] = list(module.weight.size())
            else:
                summary[m_key]['weight'] = 'None'

            if hasattr(module, "bias") and module.bias is not None:
                if isinstance(module, nn.Conv2d) and module.__str__().startswith('MaskedConv2d'):
                    flops1 = 0
                    mask = torch.sum(torch.sum(module._mask[:,:,:,:].data, 2), 2).cpu()/(module.kernel_size[0]*module.kernel_size[1])  #[out_channels, in_channels, W, H]
                    mask = torch.sum(mask, 1).cpu().numpy()  #[out_channels, in_channels]
                    #print("mask bias",mask)

                    flops1 += np.sum(mask>0)
                    flops += flops1
                    #print("flops bias", flops1)
                elif isinstance(module, nn.Conv2d) and (not module.__str__().startswith('MaskedConv2d')):
                    flops1 = 0
                    flops1 = module.bias.numel()
                    flops += flops1
                    #print("flops bias", flops1)

                elif isinstance(module, nn.Linear):
                    flops1 = 0
                    flops1 = module.bias.numel()
                    flops += flops1
                    #print("flops bias", flops1)

            summary[m_key]["flops"] = flops

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("---------------------------------------------------------------------------------------------------------------------------")
    line_new = "{:>20}  {:>25}   {:>25} {:>15} {:>15} {:>15}".format("Layer (type)", "Input Shape", "Output Shape", "Weight", "Param #", "FLOPs #")
    print(line_new)
    print("===========================================================================================================================")
    total_params = 0
    total_flops = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25}  {:>25} {:>15} {:>15} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            str(summary[layer]["weight"]),
            "{0:,}".format(summary[layer]["nb_params"]),
            "{0:,}".format(summary[layer]["flops"]),
        )
        total_params += summary[layer]["nb_params"]
        total_flops += summary[layer]["flops"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("===========================================================================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Total flops: {0:,}".format(total_flops))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("---------------------------------------------------------------------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("---------------------------------------------------------------------------------------------------------------------------")
    # return summary

##############################################################################################################
if __name__ == '__main__':

	os.environ["CUDA_VISIBLE_DEVICES"] = "2"

	model = densenet121() #Total params: 23,272,266 #Total FLOPS: 62,751,882 #Accuracy: 85.61_98.95
	#model = torch.load("model_training").cpu()
	model = torch.load("model_training_final").cpu()
	#model = model.module
	print("model:", model)

	#summary(model, (1, 28, 28), device="cpu")
	#summary(model, (3, 32, 32), device="cpu")
	summary(model, (3, 224, 224), device="cpu")