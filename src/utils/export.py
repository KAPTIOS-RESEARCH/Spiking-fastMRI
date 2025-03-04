def export_to_onnx(model, input_shape):
    dummy_input = torch.randn(input_shape)

    torch.onnx.export(
        model, 
        dummy_input, 
        "cunet.onnx",
        opset_version=11,
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )