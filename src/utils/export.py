import torch, os, logging

def save_to_onnx(path: str, model: torch.nn.Module, tensor_x: torch.Tensor):
    """Exports a PyTorch model to ONNX format.

    Args:
        path (str): The directory were the .onnx file will be saved
        model (torch.nn.Module): The model to export
        tensor_x (torch.Tensor): A data sample used to train the model
    """
    os.makedirs(path, exist_ok = True)
    onnx_program = torch.onnx.export(
        model, 
        (tensor_x,), 
        dynamo=True,
        export_params=True,        
        opset_version=18,          
        do_constant_folding=True,  
        input_names = ['input'],                          
        output_names = ['output'],
    )
    onnx_program.save(os.path.join(path, 'model.onnx'));
    logging.info(f'Model successfully exported to ONNX format in {path}')