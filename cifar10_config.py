from attacks import wrap_attack, wrap_cw_linf, ifgsm, momentum_ifgsm, fgsm, deepfool, CW_Linf, Transferable_Adversarial_Perturbations, ILA
from cifar10models import *

data_preprocess = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

def model_name(model):
    return model.__name__

model_configs = {
    model_name(ResNet18): (ResNet18, '/share/cuvl/weights/cifar10/resnet18_epoch_347_acc_94.77.pth'),
    model_name(DenseNet121): (DenseNet121,'/share/cuvl/weights/cifar10/densenet121_epoch_315_acc_95.61.pth'),
    model_name(GoogLeNet): (GoogLeNet, '/share/cuvl/weights/cifar10/googlenet_epoch_227_acc_94.86.pth'),
    model_name(SENet18): (SENet18, '/share/cuvl/weights/cifar10/senet18_epoch_279_acc_94.59.pth')
}


def attack_name(attack):
    return attack.__name__  
         
attack_configs ={ 
    attack_name(ifgsm): wrap_attack(ifgsm, {'learning_rate': 0.002, 'epsilon': 0.03}),
    attack_name(momentum_ifgsm): wrap_attack(momentum_ifgsm, {'learning_rate': 0.002, 'epsilon': 0.03, 'decay': 0.9}),
    attack_name(fgsm): wrap_attack(fgsm, {}),
    attack_name(deepfool): wrap_attack(deepfool, {}), 
    attack_name(CW_Linf): wrap_cw_linf(CW_Linf, {'targeted': False} ),
    attack_name(Transferable_Adversarial_Perturbations): wrap_attack(Transferable_Adversarial_Perturbations, {'learning_rate': 0.002, 'epsilon': 0.03,  'lam' : 0.005, 'alpha' : 0.5, 's' : 3, 'yita' : 0.01} ),
}


use_projection = True

ILA_params = {
    attack_name(ifgsm): {'niters': 10, 'learning_rate': 0.006, 'epsilon':0.03, 'coeff': 5.0}, 
    attack_name(momentum_ifgsm): {'niters': 10, 'learning_rate': 0.006, 'epsilon':0.03, 'coeff': 5.0},
    attack_name(fgsm):  {'niters': 10, 'learning_rate': 0.006, 'epsilon':0.03, 'coeff': 5.0}, 
    attack_name(deepfool): {'niters': 10, 'learning_rate': 1.0, 'epsilon':0.03, 'coeff': 5.0},
    attack_name(CW_Linf): {'niters': 10, 'learning_rate': 0.006, 'epsilon':0.03, 'coeff': 5.0},
    attack_name(Transferable_Adversarial_Perturbations): {'niters': 10, 'learning_rate': 0.006, 'epsilon':0.03, 'coeff': 5.0},
}

source_layers = {
    model_name(ResNet18): list(enumerate(ResNet18()._modules.keys())), 
    model_name(DenseNet121): list(enumerate(DenseNet121()._modules.keys())), 
    model_name(GoogLeNet): list(enumerate(GoogLeNet()._modules.keys())), 
    model_name(SENet18):list(enumerate(SENet18()._modules.keys())), 
}
                



 