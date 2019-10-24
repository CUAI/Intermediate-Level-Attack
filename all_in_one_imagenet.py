import argparse
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from imagenet_config import *
from attacks import wrap_attack_imagenet, momentum_ifgsm, Transferable_Adversarial_Perturbations, ILA, ifgsm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_models', nargs='+', help='<Required> source models', required=True)
    parser.add_argument('--transfer_models', nargs='+', help='<Required> transfer models', required=True)
    parser.add_argument('--attacks', nargs='+', help='<Required> base attacks', required=True)
    parser.add_argument('--num_batches', type=int, help='<Required> number of batches', required=True)
    parser.add_argument('--batch_size', type=int, help='<Required> batch size', required=True)
    parser.add_argument('--out_name', help='<Required> out file name', required=True)
    parser.add_argument('--use_Inc_model', action='store_true', help='<Required> use Inception models group')
    args = parser.parse_args()
    return args

def log(out_df, source_model_name, target_model_name, batch_index, layer_index, layer_name, fool_method, with_ILA,  fool_rate, acc_after_attack, original_acc):
    return out_df.append({
        'source_model':source_model_name, 
        'target_model':target_model_name,
        'batch_index':batch_index,
        'layer_index':layer_index, 
        'layer_name':layer_name, 
        'fool_method':fool_method, 
        'with_ILA':with_ILA,  
        'fool_rate':fool_rate, 
        'acc_after_attack':acc_after_attack, 
        'original_acc':original_acc},ignore_index=True)


def get_data(batch_size, use_Inc_model = False):
    
    if use_Inc_model:
        transform_test = transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),  
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
                      ])
    else:
        transform_test = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),  
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          ])   
    
    testset = torchvision.datasets.ImageFolder(root='/share/cuvl/datasets/imagenet/val', 
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, 
                                             num_workers=8, pin_memory=True)
    return testloader

def get_fool_adv_orig(model, adversarial_xs, originals, labels):
    total = adversarial_xs.size(0)
    correct_orig = 0
    correct_adv = 0
    fooled = 0

    advs, ims, lbls = adversarial_xs.cuda(), originals.cuda(), labels.cuda()
    outputs_adv = model(advs)
    outputs_orig = model(ims)
    _, predicted_adv = torch.max(outputs_adv.data, 1)
    _, predicted_orig = torch.max(outputs_orig.data, 1)

    correct_adv += (predicted_adv == lbls).sum()
    correct_orig += (predicted_orig == lbls).sum()
    fooled += (predicted_adv != predicted_orig).sum()
    return [100.0 * float(fooled.item())/total, 100.0 * float(correct_adv.item())/total, 100.0 * float(correct_orig.item())/total]


def test_adv_examples_across_models(transfer_models, adversarial_xs, originals, labels, use_Inc_model):
    accum = []
    for name, net_class in transfer_models:
        if use_Inc_model:
            net = net_class(num_classes=1000, pretrained='imagenet').cuda()
        else:
            net = net_class(pretrained=True).cuda()
        net.eval()
        res = get_fool_adv_orig(net, adversarial_xs, originals, labels)
        res.append(name)
        accum.append(res)
    return accum


def complete_loop(sample_num, batch_size, attacks, source_models, transfer_models, out_name, use_Inc_model):
    labels_file = open('labels', 'r').readlines()
    out_df = pd.DataFrame(columns=['source_model','target_model','batch_index','layer_index', 'layer_name', 'fool_method', 'with_ILA',  'fool_rate', 'acc_after_attack', 'original_acc'])
    testloader = get_data(batch_size, use_Inc_model)
    for source_model_name, model_class in source_models:
        if use_Inc_model:
            model = model_class(num_classes=1000, pretrained='imagenet').cuda()
        else:
            model = model_class(pretrained=True).cuda()
        model.eval()
        for attack_name, attack in attacks:
            print('using source model {0} attack {1}'.format(source_model_name, attack_name))

            for batch_i, data in enumerate(testloader, 0):

                if batch_i%100 == 0: 
                    print("batch" , batch_i)
                    save_to_csv(out_df, out_name)
                if batch_i == sample_num:
                    break
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                
                #### baseline 

                ### generate
                adversarial_xs = attack(model, images, labels, niters=20, use_Inc_model=use_Inc_model)

                ### eval
                transfer_list = test_adv_examples_across_models(transfer_models, adversarial_xs, images, labels, use_Inc_model)
                for target_fool_rate, target_acc_attack, target_acc_original, transfer_model_name in transfer_list:
                    out_df = log(out_df,source_model_name, transfer_model_name, 
                                 batch_i, np.nan, "", attack_name, False, 
                                 target_fool_rate, target_acc_attack, target_acc_original)

                #### ILA
                
                ### generate
                ## step1: reference 
                ILA_input_xs = attack(model, images, labels, niters=10, use_Inc_model=use_Inc_model) 

                ## step2: ILA target at different layers
                for layer_ind, (layer_name, layer) in get_source_layers(source_model_name, model):
                    ILA_adversarial_xs = ILA(model, images, X_attack=ILA_input_xs, y=labels, feature_layer=layer, use_Inc_model=use_Inc_model, **(ILA_params[attack_name]))

                    ### eval
                    ILA_transfer_list = test_adv_examples_across_models(transfer_models, ILA_adversarial_xs, images, labels, use_Inc_model)
                    for target_fool_rate, target_acc_attack, target_acc_original, transfer_model_name in ILA_transfer_list:
                        out_df = log(out_df,source_model_name, transfer_model_name, batch_i, layer_ind, layer_name, attack_name, True, target_fool_rate, target_acc_attack, target_acc_original)
            
            save_to_csv(out_df, out_name)    
            
          
def save_to_csv(out_df, out_name):
     #save csv
    out_df.to_csv(out_name, sep=',', encoding='utf-8')
    

if __name__ == "__main__":
    args = get_args()
    attacks = list(map(lambda attack_name: (attack_name, attack_configs[attack_name]), args.attacks))
    source_models = list(map(lambda model_name: model_configs[model_name], args.source_models))
    transfer_models = list(map(lambda model_name: model_configs[model_name], args.transfer_models))
    
    complete_loop(args.num_batches, args.batch_size, attacks, source_models, transfer_models, args.out_name, args.use_Inc_model);
    


















