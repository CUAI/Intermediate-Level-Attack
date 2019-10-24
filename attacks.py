#!/usr/bin/env python3
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.autograd import Variable
from scipy.misc import imread, imresize, imsave


# helpers
def avg(l):
    return sum(l) / len(l)


def sow_images(images):
    """Sow batch of torch images (Bx3xWxH) into a grid PIL image (BWxHx3)

    Args:
        images: batch of torch images.

    Returns:
        The grid of images, as a numpy array in PIL format.
    """
    images = torchvision.utils.make_grid(
        images
    )  # sow our batch of images e.g. (4x3x32x32) into a grid e.g. (3x32x128)
    
    mean_arr, stddev_arr = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    # denormalize
    for c in range(3):
        images[c, :, :] = images[c, :, :] * stddev_arr[c] + mean_arr[c]

    images = images.cpu().numpy()  # go from Tensor to numpy array
    # switch channel order back from
    # torch Tensor to PIL image: going from 3x32x128 - to 32x128x3
    images = np.transpose(images, (1, 2, 0))
    return images


# normalization (L-inf norm projection) code for output delta
def normalize_and_scale_imagenet(delta_im, epsilon, use_Inc_model):
    """Normalize and scale imagenet perturbation according to epsilon Linf norm

    Args:
        delta_im: perturbation on imagenet images
        epsilon: Linf norm

    Returns:
        The re-normalized perturbation
    """
     
    if use_Inc_model:
        stddev_arr = [0.5, 0.5, 0.5]
    else:
        stddev_arr = [0.229, 0.224, 0.225]
    
    for ci in range(3):
        mag_in_scaled = epsilon / stddev_arr[ci]
        delta_im[:,ci] = delta_im[:,ci].clone().clamp(-mag_in_scaled, mag_in_scaled)

    return delta_im


def renormalization(X, X_pert, epsilon, dataset="cifar10", use_Inc_model = False):
    """Normalize and scale perturbations according to epsilon Linf norm

    Args:
        X: original images
        X_pert: adversarial examples corresponding to X
        epsilon: Linf norm
        dataset: dataset images are from, 'cifar10' | 'imagenet'

    Returns:
        The re-normalized perturbation
    """
    # make sure you don't modify the original image beyond epsilon, also clamp
    if dataset == "cifar10":
        eps_added = (X_pert.detach() - X.clone()).clamp(-epsilon, epsilon) + X.clone()
        # clamp
        return eps_added.clamp(-1.0, 1.0)
    elif dataset == "imagenet":
        eps_added = normalize_and_scale_imagenet(X_pert.detach() - X.clone(), epsilon, use_Inc_model) + X.clone()
        # clamp
        mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        for i in range(3):
            min_clamp = (0 - mean[i]) / stddev[i]
            max_clamp = (1 - mean[i]) / stddev[i]
            eps_added[:,i] = eps_added[:,i].clone().clamp(min_clamp, max_clamp)
        return eps_added


def ifgsm(
    model,
    X,
    y,
    niters=10,
    epsilon=0.03,
    visualize=False,
    learning_rate=0.005,
    display=4,
    defense_model=False,
    setting="regular",
    dataset="cifar10",
    use_Inc_model = False,
):
    """Perform ifgsm attack with respect to model on images X with labels y

    Args:
        model: torch model with respect to which attacks will be computed
        X: batch of torch images
        y: labels corresponding to the batch of images
        niters: number of iterations of ifgsm to perform
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1
        visualize: whether you want to visualize the perturbations or not
        learning_rate: learning rate of ifgsm
        display: number of images to display in visualization
        defense_model: set to true if you are using a defended model,
        e.g. ResNet18Defended, instead of the usual ResNet18
        setting: 'regular' is usual ifgsm, 'll' is least-likely ifgsm, and
        'nonleaking' is non-label-leaking ifgsm
        dataset: dataset the images are from, 'cifar10' | 'imagenet'

    Returns:
        The batch of adversarial examples corresponding to the original images
    """
    model.eval()
    out = None
    if defense_model:
        out = model(X)[0]
    else:
        out = model(X)
    y_ll = out.min(1)[1]  # least likely model output
    y_ml = out.max(1)[1]  # model label

    X_pert = X.clone()
    X_pert.requires_grad = True
    for i in range(niters):
        output_perturbed = None
        if defense_model:
            output_perturbed = model(X_pert)[0]
        else:
            output_perturbed = model(X_pert)

        y_used = y
        ll_factor = 1
        if setting == "ll":
            y_used = y_ll
            ll_factor = -1
        elif setting == "noleaking":
            y_used = y_ml

        loss = nn.CrossEntropyLoss()(output_perturbed, y_used)
        loss.backward()
        pert = ll_factor * learning_rate * X_pert.grad.detach().sign()

        # perform visualization
        if visualize is True and i == niters - 1:
            np_image = sow_images(X[:display].detach())
            np_delta = sow_images(pert[:display].detach())
            np_recons = sow_images(
                (X_pert.detach() + pert.detach()).clamp(-1, 1)[:display]
            )

            fig = plt.figure(figsize=(8, 8))
            fig.add_subplot(3, 1, 1)
            plt.axis("off")
            plt.imshow(np_recons)
            fig.add_subplot(3, 1, 2)
            plt.axis("off")
            plt.imshow(np_image)
            fig.add_subplot(3, 1, 3)
            plt.axis("off")
            plt.imshow(np_delta)
            plt.show()
        # end visualization

        # add perturbation
        X_pert = X_pert.detach() + pert
        X_pert.requires_grad = True

        # make sure we don't modify the original image beyond epsilon and clamp
        X_pert = renormalization(X, X_pert, epsilon, dataset=dataset, use_Inc_model=use_Inc_model)
        X_pert.requires_grad = True

    return X_pert


def fgsm(model, X, y, epsilon=0.01, **args):
    """Perform cifar 10 fgsm attack with respect to model on images X with labels y

    Args:
        model: torch model with respect to which attacks will be computed
        X: batch of torch images
        y: labels corresponding to the batch of images
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1

    Returns:
        The batch of adversarial examples corresponding to the original images
    """
    if dataset != "cifar10":
        raise "fgsm as now does not support " + dataset

    X_pert = X.clone()
    X_pert.requires_grad = True
    output_perturbed = model(X_pert)
    loss = nn.CrossEntropyLoss()(output_perturbed, y)
    loss.backward()

    pert = epsilon * X_pert.grad.detach().sign()
    X_pert = X_pert.detach() + pert
    X_pert = X_pert.detach().clamp(X.min(), X.max())
    return X_pert


def momentum_ifgsm(
    model,
    X,
    y,
    niters=10,
    epsilon=0.03,
    visualize=False,
    learning_rate=0.005,
    decay=0.9,
    dataset="cifar10",
    use_Inc_model = False,
):
    """Perform momentum ifgsm attack with respect to model on images
    X with labels y

    Args:
        model: torch model with respect to which attacks will be computed
        X: batch of torch images
        y: labels corresponding to the batch of images
        niters: number of iterations of ifgsm to perform
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1
        visualize: whether you want to visualize the perturbations or not
        learning_rate: learning rate of ifgsm
        decay: decay of momentum in the momentum ifgsm attack
        dataset: dataset the images are from, 'cifar10' | 'imagenet'

    Returns:
        The batch of adversarial examples corresponding to the original images
    """
    X_pert = X.clone()
    X_pert.requires_grad = True

    momentum = 0
    for _ in range(niters):
        output_perturbed = model(X_pert)
        loss = nn.CrossEntropyLoss()(output_perturbed, y)
        loss.backward()

        momentum = decay * momentum + X_pert.grad / torch.sum(torch.abs(X_pert.grad))
        pert = learning_rate * momentum.sign()

        # add perturbation
        X_pert = X_pert.detach() + pert
        X_pert.requires_grad = True

        # make sure we don't modify the original image beyond epsilon
        X_pert = renormalization(X, X_pert, epsilon, dataset=dataset, use_Inc_model=use_Inc_model)
        X_pert.requires_grad = True

    return X_pert


def deepfool_single(net, image, num_classes=10, overshoot=0.02, max_iter=20):
    """Perform cifar10 deepfool attack with respect to net on image,
    pushing towards easiest target class as defined in the deepfool paper

    Args:
        net: torch model with respect to which attacks will be computed
        image: single image
        num_classes: number of classes net classifies
        overshoot: overshoot parameter of deepfool attack
        max_iter: maximum number of iterations of deepfool attack

    Returns:
        The adversarial examples corresponding to image
    """
    import copy
    from torch.autograd.gradcheck import zero_gradients
    from torch.autograd import Variable

    f_image = (
        net.forward(Variable(image[None, :, :, :], requires_grad=True))
        .data.cpu()
        .numpy()
        .flatten()
    )
    out = (np.array(f_image)).flatten().argsort()[::-1]

    out = out[0:num_classes]
    label = out[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, out[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, out[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, out[k]] - fs[0, out[0]]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot
    return pert_image



def deepfool(model, images, labels, num_classes=10, niters=50, epsilon=0.03, dataset="cifar10"):
    """Perform cifar 10 deepfool attack with respect to net on images with specified
    labels, pushing towards easiest target class as defined in the
    deepfool paper

    Args:
        model: torch model with respect to which attacks will be computed
        images: batch of torch images
        labels: groundtruth labels of corresponding images
        num_classes: number of classes net classifies
        niters: number of iterations
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1

    Returns:
        The batch of adversarial examples corresponding to the original images
    """
    if dataset != "cifar10":
        raise "deepfool as now does not support " + dataset
    r = torch.zeros(images.size()).cuda()
    for i in range(images.size(0)):
        r[i] = deepfool_single(model, images[i], num_classes, max_iter=niters)

    adv = (r - images).clamp(-epsilon, epsilon) + images
    adv = adv.clamp(-1.0, 1.0)
    return adv


# CW

# cifar only, -1..1 image range
def tanh(x):
    return torch.nn.Tanh()(x)


def return_max(x_tensor, y):
    if x_tensor.data > y:
        return x_tensor.data
    else:
        return y


def torch_arctanh(x, eps=1e-6):
    x *= 1.0 - eps
    return (torch.log((1 + x) / (1 - x))) * 0.5


class CW_Linf:
    def __init__(
        self,
        targeted=True,
        learning_rate=5e-3,
        max_iterations=1000,
        abort_early=True,
        initial_const=1e-5,
        largest_const=2e1,
        reduce_const=False,
        decrease_factor=0.9,
        const_factor=2.0,
        num_classes=10,
    ):
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.DECREASE_FACTOR = decrease_factor
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor
        self.num_classes = num_classes
        self.cuda = True

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False

    def gradient_descent(self, model):
        def compare(x, y):
            if self.TARGETED:
                return x == y
            else:
                return x != y

        # TODO replace hardcode
        shape = (1, 3, 32, 32)

        def doit(oimgs, labs, starts_temp, tt, CONST):
            # convert to tanh space
            imgs = torch_arctanh(oimgs * 1.999999)
            starts = torch_arctanh(starts_temp * 1.999999)

            # initial tau
            tau = tt
            timg = imgs
            tlab = labs
            const = CONST

            # changing tlab to one hot
            target_onehot = torch.zeros(
                (1, self.num_classes), requires_grad=True, device="cuda"
            )
            tlab = target_onehot.scatter(1, tlab.unsqueeze(1), 1.0)
            # iterate through constants, try to get highest one
            while CONST < self.LARGEST_CONST:
                # setup the modifier variable,
                # this is the variable we are optimizing over
                modifier = torch.zeros(shape, requires_grad=True, device="cuda")
                optimizer = optim.Adam([modifier], lr=self.LEARNING_RATE)

                # starting point for simg
                simg = starts.clone()

                for _ in range(self.MAX_ITERATIONS):
                    newimg = tanh(modifier + simg) / 2

                    output = model(newimg)
                    orig_output = model(tanh(timg) / 2)  # assumes -0.5..0.5

                    real = torch.mean((tlab) * output)
                    other = torch.max((1 - tlab) * output - (tlab * 10000))

                    if self.TARGETED:
                        # if targetted, optimize for making the other class most likely
                        loss1 = torch.max(other - real, torch.zeros_like(real))
                    else:
                        # if untargeted, optimize for making this class least likely.
                        loss1 = torch.max(real - other, torch.zeros_like(real))

                    loss2 = torch.sum(
                        torch.max(
                            torch.abs(newimg - tanh(timg) / 2) - tau,
                            torch.zeros_like(newimg),
                        )
                    )
                    loss = const * loss1 + loss2

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    # it worked
                    if loss < 0.0001 * CONST and self.ABORT_EARLY:
                        works = compare(torch.argmax(output), torch.argmax(tlab))
                        if works:
                            return output, orig_output, newimg, CONST

                C = CONST * self.const_factor
                CONST = C

        return doit

    def attack(self, model, imgs, targets):
        """
        Perform the L_inf attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        i = 0
        for img, target in zip(imgs, targets):
            orig_lab = target.cpu().data.numpy()
            target_lab = random.choice([j for j in range(10) if j != orig_lab])
            x = self.attack_single(
                model, img.unsqueeze(0), torch.tensor([target_lab]).cuda()
            )
            print(target.data, torch.max(model(x), 1)[1])
            r.extend(x)
            i += 1
        return r

    def attack_single(self, model, img, target):
        """
        Run the attack on a single image and label
        """
        prev = img.clone()
        tau = 1.0
        const = self.INITIAL_CONST

        while tau > 1.0 / 256:
            # try to solve given this tau value
            res = self.gradient_descent(model)(
                img.clone(), target, prev.clone(), tau, const
            )
            if res is None:
                return prev

            scores, origscores, nimg, const = res

            if self.REDUCE_CONST:
                const = torch.div(const, 2)

            # the attack succeeded, reduce tau and try again
            actualtau = torch.max(torch.abs(nimg - img))

            if actualtau < tau:
                tau = actualtau

            prev = nimg.clone()
            t = tau * self.DECREASE_FACTOR
            tau = t
        return prev


# function to wrap/instantiate and call CW_linf
def wrap_cw_linf(attack, params):
    """Perform cifar10 cw linf attack using model on "images" with "labels"

    Args:
        model: torch model with respect to which attacks will be computed
        images: batch of torch images
        labels: labels corresponding to the batch of images
        niters: number of iterations of cw to perform
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1

    Returns:
        The batch of adversarial examples corresponding to the original images
    """
    def wrap_f(model, images, labels, niters, use_Inc_model=False):
        def net_hacked(x):
            return model(x * 2)

        i2 = images.clone() / 2  # CW optimization happens on -0.5..0.5 range
        cw = CW_Linf(max_iterations=niters)
        res_unclip = torch.stack(cw.attack(net_hacked, i2, labels))
        res = (res_unclip - i2).detach().clamp(-epsilon / 2, epsilon / 2) + i2
        r = res.clamp(-0.5, 0.5)
        return torch.mul(r, 2).cuda()  # return normal -1..1 range adversarial example
    return wrap_f


def wrap_attack(attack, params, dataset='cifar10'):
    """
    A wrap for attack functions
    attack: an attack function
    params: kwargs for the attack func
    """
    def wrap_f(model, images, labels, niters, use_Inc_model=False):
        # attack should process by batch
        return attack(model, images, labels, niters=niters, dataset=dataset, use_Inc_model=use_Inc_model, **params)

    return wrap_f


def wrap_attack_imagenet(attack, params):
    return wrap_attack(attack, params, dataset='imagenet')


# TAP (transferable adversairal perturbation ECCV 2018)
class Transferable_Adversarial_Perturbations_Loss(torch.nn.Module):
    def __init__(self):
        super(Transferable_Adversarial_Perturbations_Loss, self).__init__()

    def forward(
        self,
        X,
        X_pert,
        original_mids,
        new_mids,
        y,
        output_perturbed,
        lam,
        alpha,
        s,
        yita,
    ):

        l1 = nn.CrossEntropyLoss()(output_perturbed, y)

        l2 = 0
        for i, new_mid in enumerate(new_mids):
            a = torch.sign(original_mids[i]) * torch.pow(
                torch.abs(original_mids[i]), alpha
            )
            b = torch.sign(new_mid) * torch.pow(torch.abs(new_mid), alpha)
            l2 += lam * (a - b).norm() ** 2

        l3 = yita * torch.abs(nn.AvgPool2d(s)(X - X_pert)).sum()

        return l1 + l2 + l3


mid_outputs = []


def Transferable_Adversarial_Perturbations(
    model,
    X,
    y,
    niters=10,
    epsilon=0.03,
    lam=0.005,
    alpha=0.5,
    s=3,
    yita=0.01,
    learning_rate=0.006,
    dataset="cifar10",
    use_Inc_model = False,
):
    """Perform cifar10 TAP attack using model on images X with labels y

    Args:
        model: torch model with respect to which attacks will be computed
        X: batch of torch images
        y: labels corresponding to the batch of images
        niters: number of iterations of TAP to perform
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1
        lam: lambda parameter of TAP
        alpha: alpha parameter of TAP
        s: s parameter of TAP
        yita: yita parameter of TAP
        learning_rate: learning rate of TAP attack

    Returns:
        The batch of adversarial examples corresponding to the original images
    """
    feature_layers = list(model._modules.keys())
    global mid_outputs
    X = X.detach()
    X_pert = torch.zeros(X.size()).cuda()
    X_pert.copy_(X).detach()
    X_pert.requires_grad = True

    def get_mid_output(m, i, o):
        global mid_outputs
        mid_outputs.append(o)

    hs = []
    for layer_name in feature_layers:
        hs.append(model._modules.get(layer_name).register_forward_hook(get_mid_output))

    out = model(X)
        
    mid_originals = []
    for mid_output in mid_outputs:
        mid_original = torch.zeros(mid_output.size()).cuda()
        mid_originals.append(mid_original.copy_(mid_output))

    mid_outputs = []

    for _ in range(niters):
        output_perturbed = model(X_pert)
        # generate adversarial example by max middle
        # layer pertubation in the direction of increasing loss
        mid_originals_ = []
        for mid_original in mid_originals:
            mid_originals_.append(mid_original.detach())

        loss = Transferable_Adversarial_Perturbations_Loss()(
            X,
            X_pert,
            mid_originals_,
            mid_outputs,
            y,
            output_perturbed,
            lam,
            alpha,
            s,
            yita,
        )
        loss.backward()
        pert = learning_rate * X_pert.grad.detach().sign()

        # minimize loss
        X_pert = X_pert.detach() + pert
        X_pert.requires_grad = True

        # make sure we don't modify the original image beyond epsilon
        X_pert = renormalization(X, X_pert, epsilon, dataset=dataset, use_Inc_model=use_Inc_model)
        X_pert.requires_grad = True

        mid_outputs = []

    for h in hs:
        h.remove()
    return X_pert


# ILA attack

# square sum of dot product
class Proj_Loss(torch.nn.Module):
    def __init__(self):
        super(Proj_Loss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)
        x_norm = x / x.norm()

        proj_loss = torch.mm(y, x_norm.transpose(0, 1)) / x.norm()
        return proj_loss


# square sum of dot product
class Mid_layer_target_Loss(torch.nn.Module):
    def __init__(self):
        super(Mid_layer_target_Loss, self).__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1, -1)
        y = (new_mid - original_mid).view(1, -1)

        x_norm = x / x.norm()
        if (y == 0).all():
            y_norm = y
        else:
            y_norm = y / y.norm()
        angle_loss = torch.mm(x_norm, y_norm.transpose(0, 1))
        magnitude_gain = y.norm() / x.norm()
        return angle_loss + magnitude_gain * coeff


"""Return: perturbed x"""
mid_output = None


def ILA(
    model,
    X,
    X_attack,
    y,
    feature_layer,
    niters=10,
    epsilon=0.01,
    coeff=1.0,
    learning_rate=1,
    dataset="cifar10",
    use_Inc_model = False,
    with_projection=True,
):
    """Perform ILA attack with respect to model on images X with labels y

    Args:
        with_projection: boolean, specifies whether projection should happen
        in the attack
        model: torch model with respect to which attacks will be computed
        X: batch of torch images
        X_attack: starting adversarial examples of ILA that will be modified
        to become more transferable
        y: labels corresponding to the batch of images
        feature_layer: layer of model to project on in ILA attack
        niters: number of iterations of the attack to perform
        epsilon: Linf norm of resulting perturbation; scale of images is -1..1
        coeff: coefficient of magnitude loss in ILA attack
        visualize: whether you want to visualize the perturbations or not
        learning_rate: learning rate of the attack
        dataset: dataset the images are from, 'cifar10' | 'imagenet'

    Returns:
        The batch of modified adversarial examples, examples have been
        augmented from X_attack to become more transferable
    """
    X = X.detach()
    X_pert = torch.zeros(X.size()).cuda()
    X_pert.copy_(X).detach()
    X_pert.requires_grad = True

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o

    h = feature_layer.register_forward_hook(get_mid_output)

    out = model(X)
    mid_original = torch.zeros(mid_output.size()).cuda()
    mid_original.copy_(mid_output)

    out = model(X_attack)
    mid_attack_original = torch.zeros(mid_output.size()).cuda()
    mid_attack_original.copy_(mid_output)

    for _ in range(niters):
        output_perturbed = model(X_pert)

        # generate adversarial example by max middle layer pertubation
        # in the direction of increasing loss
        if with_projection:
            loss = Proj_Loss()(
                mid_attack_original.detach(), mid_output, mid_original.detach(), coeff
            )
        else:
            loss = Mid_layer_target_Loss()(
                mid_attack_original.detach(), mid_output, mid_original.detach(), coeff
            )

        loss.backward()
        pert = learning_rate * X_pert.grad.detach().sign()

        # minimize loss
        X_pert = X_pert.detach() + pert
        X_pert.requires_grad = True

        # make sure we don't modify the original image beyond epsilon
        X_pert = renormalization(X, X_pert, epsilon, dataset=dataset, use_Inc_model=use_Inc_model)
        X_pert.requires_grad = True

    h.remove()
    return X_pert


def input_diversity(X, p, image_width, image_resize):
    rnd = torch.randint(image_width, image_resize, ())
    rescaled = nn.functional.interpolate(X, [rnd, rnd])
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem,())
    pad_right = w_rem - pad_left
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [image_width, image_width])
    return padded if torch.rand(()) < p else X


def DI_2_fgsm(
    model, 
    X, 
    y, 
    niters=10, 
    epsilon=0.01, 
    p = 0.5, 
    image_width=32, 
    image_resize=36, 
    visualize=False, 
    learning_rate=0.005,
    dataset="cifar10",
    use_Inc_model = False,
):
    X_pert = X.clone()
    X_pert.requires_grad = True

    for _ in range(niters):
        output_perturbed = model(input_diversity(X_pert, p, image_width, image_resize))
        loss = nn.CrossEntropyLoss()(output_perturbed, y)
        loss.backward()
        pert = learning_rate * X_pert.grad.detach().sign()
        # add perturbation
        X_pert = X_pert.detach() + pert
        X_pert.requires_grad = True

        # make sure we don't modify the original image beyond epsilon
        X_pert = renormalization(X, X_pert, epsilon, dataset=dataset, use_Inc_model=use_Inc_model)
        X_pert.requires_grad = True

    return X_pert

TMP_PATH = '/home/qh53/Intermediate-Level-Attack-test'


def DI_2_fgsm_tf(model, batch_i, y, niters=10, epsilon=0.01, p = 0.5, image_width=299, image_resize=330, learning_rate=0.005, dataset="cifar10",
    use_Inc_model = False,):
    
    images = []
    batch_size = 8
    for i in range(batch_size):
        image = imread(TMP_PATH + '/tmp_{}_iter={}_momentum=0/{}_{}.png'.format(model, niters, batch_i, i), mode='RGB')
        images.append(image)
    
    return torch.tensor(images).permute(0, 3, 1, 2).cuda().float() / 255.0 * 2.0 - 1.0
