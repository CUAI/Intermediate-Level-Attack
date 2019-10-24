import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import argparse

from all_in_one_imagenet import get_source_layers
from attacks import ifgsm, ILA

# Training settings
parser = argparse.ArgumentParser(description='Demo for ILA on ImageNet')
parser.add_argument('--modeltype', type=str, default='resnet18', help='ResNet18 | DenseNet121 | alexnet | SqueezeNet1.0')
parser.add_argument('--layerindex', type=int, default='4', help='layer index to emphasize with ila projection')
parser.add_argument('--imagepath', type=str, default='eft_image_test.jpg', help='path to image to test')
parser.add_argument('--outpath', type=str, default='adv_example.jpg', help='path for output image')
parser.add_argument('--imagelabel', type=int, default=27, help='imagenet label (0-999)')
parser.add_argument('--niters_baseline', type=int, default=20, help='number of iterations of baseline attack')
parser.add_argument('--niters_ila', type=int, default=10, help='number of iterations of ILA')
parser.add_argument('--epsilon', type=float, default=0.03, help='epsilon on 0..1 range, 0.03 corresponds to ~8 in the imagenet scale')
opt = parser.parse_args()

# load pretrained model to attack
def load_model(model_name):
    if model_name == 'ResNet18':
        return torchvision.models.resnet18(pretrained=True).cuda()
    elif model_name == 'DenseNet121':
        return torchvision.models.densenet121(pretrained=True).cuda()
    elif model_name == 'alexnet':
        return torchvision.models.alexnet(pretrained=True).cuda()
    elif model_name == 'SqueezeNet1.0':
        return torchvision.models.squeezenet1_0(pretrained=True).cuda()
    else:
        print('Not supported model')

# load source model
model = load_model(opt.modeltype)

# load transfer models
all_model_names = ['ResNet18', 'DenseNet121', 'alexnet', 'SqueezeNet1.0']
transfer_model_names = [x for x in all_model_names if x != opt.modeltype]
transfer_models = [load_model(x) for x in transfer_model_names]

print('Loaded model...')

# pre-process input image
mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform_resize = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, stddev)])
img_pil = Image.open(opt.imagepath)
img_pil_resize = transform_resize(img_pil.copy())
img = transform_norm(img_pil_resize.copy()).cuda().unsqueeze(0)
lbl = torch.tensor([opt.imagelabel]).cuda()

# run attack on source model
img_ifgsm = ifgsm(model, img, lbl, niters=opt.niters_baseline, dataset='imagenet')

source_layers = get_source_layers(opt.modeltype, model)
ifgsm_guide = ifgsm(model, img, lbl, learning_rate=0.008, epsilon=opt.epsilon, niters=opt.niters_ila, dataset='imagenet')
img_ila = ILA(model, img, ifgsm_guide, lbl, source_layers[opt.layerindex][1][1], learning_rate=0.01, epsilon=opt.epsilon, niters=opt.niters_ila, dataset='imagenet')

# get labels for source
model.eval()
orig_pred_label, ifgsm_pred_label, ila_pred_label = model(img).max(dim=1)[1].item(), model(img_ifgsm).max(dim=1)[1].item(), model(img_ila).max(dim=1)[1].item()

# get labels for transfer
transfer_labs = []
for mod in transfer_models:
    mod.eval()
    o, f, i = mod(img).max(dim=1)[1].item(), mod(img_ifgsm).max(dim=1)[1].item(), mod(img_ila).max(dim=1)[1].item()
    transfer_labs.append((o, f, i))

print('Ran attacks...')

# display results
imagenet_labels = eval(open('imagenet_labels.txt').read())

print(f'True label: {opt.imagelabel} ({imagenet_labels[opt.imagelabel]})')

print(f'{opt.modeltype} (source model)')
print(f'Prediction on original: {orig_pred_label} ({imagenet_labels[orig_pred_label]})')
print(f'Prediction on I-FGSM: {ifgsm_pred_label} ({imagenet_labels[ifgsm_pred_label]})')
print(f'Prediction on ILA: {ila_pred_label} ({imagenet_labels[ila_pred_label]})')
print()

print('---Transfer Results Follow---')

for j, name in enumerate(transfer_model_names):
    o, f, i = transfer_labs[j]
    print(f'{name} (transfer model)')
    print(f'Prediction on original: {o} ({imagenet_labels[o]})')
    print(f'Prediction on I-FGSM: {f} ({imagenet_labels[f]})')
    print(f'Prediction on ILA: {i} ({imagenet_labels[i]})')
    print()

# helpers
class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean).cuda()
        std = torch.as_tensor(std).cuda()
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

# save output image
invTrans = NormalizeInverse(mean, stddev)
to_pil = transforms.ToPILImage()
out_img = np.array(to_pil(invTrans(img_ila[0]).clamp(0,1).cpu())).clip(0, 255).astype(np.uint8)

print(f'Image L-inf modification: {np.abs(out_img.astype(np.int32) - np.array(img_pil_resize).astype(np.int32)).max()}')
Image.fromarray(out_img).save(opt.outpath)

print('Saved image.')
