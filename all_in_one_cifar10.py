import argparse
from attacks import wrap_attack, wrap_cw_linf, ifgsm, momentum_ifgsm, deepfool, CW_Linf, Transferable_Adversarial_Perturbations, ILA
from cifar10models import *
from cifar10_config import *
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_models', nargs='+', help='<Required> source models', required=True)
    parser.add_argument('--transfer_models', nargs='+', help='<Required> transfer models', required=True)
    parser.add_argument('--attacks', nargs='+', help='<Required> base attacks', required=True)
    parser.add_argument('--num_batches', type=int, help='<Required> number of batches', required=True)
    parser.add_argument('--batch_size', type=int, help='<Required> batch size', required=True)
    parser.add_argument('--out_name', help='<Required> out file name', required=True)
    args = parser.parse_args()
    return args

def log(out_df, source_model, source_model_file, target_model, target_model_file, batch_index, layer_index, layer_name, fool_method, with_ILA,  fool_rate, acc_after_attack, original_acc):
    return out_df.append({
        'source_model':model_name(source_model), 
        'source_model_file': source_model_file,
        'target_model':model_name(target_model),
        'target_model_file': target_model_file,
        'batch_index':batch_index,
        'layer_index':layer_index, 
        'layer_name':layer_name, 
        'fool_method':fool_method, 
        'with_ILA':with_ILA,  
        'fool_rate':fool_rate, 
        'acc_after_attack':acc_after_attack, 
        'original_acc':original_acc},ignore_index=True)


def get_data(batch_size, mean, stddev):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, stddev)])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader

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


def test_adv_examples_across_models(transfer_models, adversarial_xs, originals, labels):
    accum = []
    for (network, weights_path) in transfer_models:
        net = network().cuda()
        net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
        net.eval()
        res = get_fool_adv_orig(net, adversarial_xs, originals, labels)
        res.append(weights_path)
        accum.append(res)
    return accum


def complete_loop(sample_num, batch_size, attacks, source_models, transfer_models, out_name):
    out_df = pd.DataFrame(columns=['source_model', 'source_model_file', 'target_model','target_model_file', 'batch_index','layer_index', 'layer_name', 'fool_method', 'with_ILA',  'fool_rate', 'acc_after_attack', 'original_acc'])

    trainloader, testloader = get_data(batch_size, *data_preprocess)
    for model_class, source_weight_path in source_models:
        model = model_class().cuda()
        model.load_state_dict(torch.load(source_weight_path))
        model.eval()
        dic = model._modules
        for attack_name, attack in attacks:
            print('using source model {0} attack {1}'.format(model_name(model_class), attack_name))
            iterator = tqdm(enumerate(testloader, 0))
            for batch_i, data in iterator:
                if batch_i == sample_num:
                    iterator.close()
                    break
                images, labels = data
                images, labels = images.cuda(), labels.cuda() 


                #### baseline 

                ### generate
                adversarial_xs = attack(model, images, labels, niters= 20)
                 
                ### eval
                transfer_list = test_adv_examples_across_models(transfer_models, adversarial_xs, images, labels)
                for i, (target_fool_rate, target_acc_attack, target_acc_original, target_weight_path) in enumerate(transfer_list):
                    out_df = log(out_df,model_class, source_weight_path,transfer_models[i][0], 
                                 target_weight_path, batch_i, np.nan, "", attack_name, False, 
                                 target_fool_rate, target_acc_attack, target_acc_original)


                #### ILA
                
                ### generate
                ## step1: reference 
                ILA_input_xs = attack(model, images, labels, niters= 10)

                ## step2: ILA target at different layers
                for layer_ind, layer_name in source_layers[model_name(model_class)]:
                    ILA_adversarial_xs = ILA(model, images, X_attack=ILA_input_xs, y=labels, feature_layer=model._modules.get(layer_name), **(ILA_params[attack_name]))
                    
                    ### eval
                    ILA_transfer_list = test_adv_examples_across_models(transfer_models, ILA_adversarial_xs, images, labels)
                    for i, (fooling_ratio, accuracy_perturbed, accuracy_original, attacked_model_path) in enumerate(ILA_transfer_list):
                        out_df = log(out_df,model_class,attacked_model_path, transfer_models[i][0], source_weight_path, batch_i, layer_ind, layer_name, attack_name, True, fooling_ratio, accuracy_perturbed, accuracy_original)


            #save csv
            out_df.to_csv(out_name, sep=',', encoding='utf-8')


if __name__ == "__main__":
    args = get_args()
    attacks = list(map(lambda attack_name: (attack_name, attack_configs[attack_name]), args.attacks))
    source_models = list(map(lambda model_name: model_configs[model_name], args.source_models))
    transfer_models = list(map(lambda model_name: model_configs[model_name], args.transfer_models))
   
    complete_loop(args.num_batches, args.batch_size, attacks, source_models, transfer_models, args.out_name);



















