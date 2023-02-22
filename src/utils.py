import torch
from torch.nn import functional as F
from torch import nn
from pathlib import Path
import os
import clearml
import numpy as np
from clearml import Task

def save_onnx(model, save_path, state_size: int):
    try: 
        # model = raw_model.clone()
        # model = model.load_state_dict(torch.load(
        #             list(Path(os.getcwd()).glob('last*'))[0]
        #             )['state_dict'])
        conv2onnx(model.actor, Path(save_path) / 'last-model.onnx', size=(state_size, 1))
        output_model = clearml.OutputModel()
        output_model.update_weights(str(Path(save_path) / 'last-model.onnx'))
    except IndexError:
        print('Best model not found, skipping torch save')
    

def conv2onnx(model, outpath, size):
    
    w,h = size
    model.eval()
    device = next(model.hidden_layers.parameters()).device
    x = torch.randn((h, w)).to(device)

    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        outpath,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable lenght axes
            "output": {0: "batch_size"},
        },
    )


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' in rads. The angle can be positive or negative
    """
    def unit_vector(vector):
        """ Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_v1, angle_v2 = np.arctan2([v1_u[1],v2_u[1]], [v1_u[0], v2_u[0]])
    
    delta = angle_v1 - angle_v2
    
    if delta > np.pi:
        delta = delta - 2*np.pi
    elif delta < -np.pi:
        delta = 2*np.pi + delta
    
    return -delta

def normalized_angle(v1, v2):
    # Receive 2 vectors (the order is important) and return the the range (-1,1)

    return angle_between(v1,v2)/np.pi

def upload_data(data):
    task = Task.current_task()
    task.upload_artifact(name='data', artifact_object=data)

def norm_grad(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def weights_init(m):
    # for every Linear layer in a model..
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)