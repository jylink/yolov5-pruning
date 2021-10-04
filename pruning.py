import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
import copy
import matplotlib.pyplot as plt
from models.yolo import Model
from utils.torch_utils import intersect_dicts, is_parallel


def load_model(weights):
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    model = Model(ckpt['model'].yaml).to(device)  # create
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])  # intersect
    model.load_state_dict(state_dict, strict=False)  # load
    print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    assert len(state_dict) == len(model.state_dict())

    model.float()
    model.model[-1].export = True
    return model

def bn_analyze(prunable_modules, save_path=None):
    bn_val = []
    max_val = []
    for layer_to_prune in prunable_modules:
        # select a layer
        weight = layer_to_prune.weight.data.detach().cpu().numpy()
        max_val.append(max(weight))
        bn_val.extend(weight)
    bn_val = np.abs(bn_val)
    max_val = np.abs(max_val)
    bn_val = sorted(bn_val)
    max_val = sorted(max_val)
    plt.hist(bn_val, bins=101, align="mid")
    if save_path is not None:
        if os.path.isfile(save_path):
            os.remove(save_path)
        plt.savefig(save_path)
    return bn_val, max_val

def channel_prune(ori_model, example_inputs, output_transform, pruned_prob=0.3, thres=None):
    model = copy.deepcopy(ori_model)
    model.cpu().eval()

    prunable_module_type = (nn.BatchNorm2d)

    prunable_modules = []
    for i, m in enumerate(model.modules()):
        if isinstance(m, prunable_module_type):
            prunable_modules.append(m)

    ori_size = tp.utils.count_params(model)
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs, output_transform=output_transform)

    bn_val, _ = bn_analyze(prunable_modules, os.path.splitext(opt.save_path)[0] + "_before_pruning.jpg")
    if thres is None:
        print('Recalculating thresh')
        thres_pos = int(pruned_prob * len(bn_val))
        thres_pos = min(thres_pos, len(bn_val)-1)
        thres_pos = max(thres_pos, 0)
        thres = bn_val[thres_pos]
    print("Min val is %f, Max val is %f, Thres is %f" % (bn_val[0], bn_val[-1], thres))

    for layer_to_prune in prunable_modules:
        # select a layer
        weight = layer_to_prune.weight.data.detach().cpu().numpy()
        prune_fn = tp.prune_batchnorm
        L1_norm = np.abs(weight)

        pos = np.array([i for i in range(len(L1_norm))])
        pruned_idx_mask = L1_norm < thres
        prun_index = pos[pruned_idx_mask].tolist()
        if len(prun_index) == len(L1_norm):
            del prun_index[np.argmax(L1_norm)]

        plan = DG.get_pruning_plan(layer_to_prune, tp.prune_batchnorm, prun_index)
        plan.exec()

    bn_analyze(prunable_modules, os.path.splitext(opt.save_path)[0] + "_after_pruning.jpg")

    model.train()
    ori_model.train()
    with torch.no_grad():
        out = model(example_inputs)
        out2 = ori_model(example_inputs)
        if output_transform:
            out = output_transform(out)
            out2 = output_transform(out2)
        print("  Params: %s => %s" % (ori_size, tp.utils.count_params(model)))
        if isinstance(out, (list, tuple)):
            for o, o2 in zip(out, out2):
                print("  Output: ", o.shape)
                assert o.shape == o2.shape, f'{o.shape} {o2.shape}'
        else:
            print("  Output: ", out.shape)
            assert out.shape == out2.shape, f'{o.shape} {o2.shape}'
        print("------------------------------------------------------\n")
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='')
    parser.add_argument('--save_path', default="", type=str, help='')
    parser.add_argument('-p', '--prob', default=0.5, type=float, help='pruning prob')
    parser.add_argument('-t', '--thres', default=0, type=float, help='pruning thres')
    parser.add_argument('--shape', nargs='+', type=int, default=[1, 3, 640, 640])
    opt = parser.parse_args()

    weights = opt.weights
    if not opt.save_path.endswith('.pt'):
        save_dir = opt.save_path if os.path.isdir(opt.save_path) else os.path.dirname(os.path.abspath(weights)) 
        save_name = os.path.splitext(os.path.basename(weights))[0] + '_pruned.pt'
        opt.save_path = os.path.join(save_dir, save_name)
        
    device = torch.device('cpu')
    model = load_model(weights)

    example_inputs = torch.zeros(opt.shape, dtype=torch.float32).to(device)
    output_transform = None
    # for prob in [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    if opt.thres != 0:
        thres = opt.thres
        prob = None
    else:
        thres = None
        prob = opt.prob

    pruned_model = channel_prune(model, example_inputs=example_inputs,
                                 output_transform=output_transform, pruned_prob=prob, thres=thres)
    pruned_model.model[-1].export = False

    ckpt = {
        'model': copy.deepcopy(pruned_model.module if is_parallel(pruned_model) else pruned_model).half(),
        'optimizer': None,
        'epoch': -1,
        }
    torch.save(ckpt, opt.save_path)
    print("Saved", opt.save_path)