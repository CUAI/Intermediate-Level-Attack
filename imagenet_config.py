import sys
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import pretrainedmodels
from attacks import wrap_attack_imagenet, ifgsm, momentum_ifgsm, ILA, Transferable_Adversarial_Perturbations


model_configs = {
    "ResNet18": ("ResNet18", models.resnet18),
    "DenseNet121": ("DenseNet121", models.densenet121),
    "SqueezeNet1.0": ("SqueezeNet1.0", models.squeezenet1_0),
    "alexnet": ("alexnet", models.alexnet),
    'Inc-v3': ('Inc-v3',pretrainedmodels.__dict__['inceptionv3']),
    'IncRes-v2': ('IncRes-v2',pretrainedmodels.__dict__['inceptionresnetv2']),
    'Inc-v4': ('Inc-v4',pretrainedmodels.__dict__['inceptionv4']),
}

def attack_name(attack):
    return attack.__name__  
         
TAP_params ={'learning_rate': 0.008, 'epsilon': 0.06,  'lam' : 0.005, 'alpha' : 0.5, 's' : 3, 'yita' : 0.01} 
attack_configs ={ 
    attack_name(ifgsm): wrap_attack_imagenet(ifgsm, {'learning_rate': 0.008, 'epsilon': 0.03}),
    attack_name(momentum_ifgsm): wrap_attack_imagenet(momentum_ifgsm, {'learning_rate': 0.018, 'epsilon': 0.03, 'decay': 0.9}),
    attack_name(Transferable_Adversarial_Perturbations): wrap_attack_imagenet(Transferable_Adversarial_Perturbations, TAP_params),
}


use_projection = True

ILA_params = {
    attack_name(ifgsm): {'niters': 10, 'learning_rate': 0.01, 'epsilon':0.03, 'coeff': 0.8, 'dataset': 'imagenet'}, 
    attack_name(momentum_ifgsm): {'niters': 10, 'learning_rate': 0.018, 'epsilon':0.03, 'coeff': 0.8, 'dataset': 'imagenet'},
    attack_name(Transferable_Adversarial_Perturbations): {'niters': 10, 'learning_rate': 0.01, 'epsilon':0.06, 'coeff': 5.0, 'dataset': 'imagenet'},
}



def get_source_layers(model_name, model):
    if model_name == 'ResNet18':
        # exclude relu, maxpool
        return list(enumerate(map(lambda name: (name, model._modules.get(name)), ['conv1', 'bn1', 'layer1', 'layer2','layer3','layer4','fc'])))
    
    elif model_name == 'DenseNet121':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get('features')._modules.get(name)), ['conv0', 'denseblock1', 'transition1', 'denseblock2', 'transition2', 'denseblock3', 'transition3', 'denseblock4', 'norm5']))
        layer_list.append(('classifier', model._modules.get('classifier')))
        return list(enumerate(layer_list))
                                             
    elif model_name == 'SqueezeNet1.0':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: ('layer '+name, model._modules.get('features')._modules.get(name)), ['0','3','4','5','7','8','9','10','12']))
        layer_list.append(('classifier', model._modules.get('classifier')._modules.get('1')))
        return list(enumerate(layer_list))
    
    elif model_name == 'alexnet':
        # exclude avgpool
        layer_list = list(map(lambda name: ('layer '+name, model._modules.get('features')._modules.get(name)), ['0','3','6','8','10']))
        layer_list += list(map(lambda name: ('layer '+name, model._modules.get('classifier')._modules.get(name)), ['1','4','6']))
        return list(enumerate(layer_list))
    
    elif model_name == 'IncRes-v2':
        # exclude relu, maxpool
        return list(enumerate(map(lambda name: (name, model._modules.get(name)), ['conv2d_1a', 'conv2d_2a', 'conv2d_2b', 'maxpool_3a', 'conv2d_3b', 'conv2d_4a', 'maxpool_5a', 'mixed_5b', 'repeat', 'mixed_6a','repeat_1', 'mixed_7a', 'repeat_2', 'block8', 'conv2d_7b', 'avgpool_1a', 'last_linear'])))

    elif model_name == 'Inc-v4':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get('features')._modules.get(name)), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']))
        return list(enumerate(layer_list))
                                             
    elif model_name == 'Inc-v3':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get(name)), ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c']))
        return list(enumerate(layer_list))
    
    else:
        # model is not supported
        assert False
    