import os
import argparse
import torch
import onnx
import onnxruntime

try:
    import pycuda.autoinit
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    print("Warning: pycuda and/or tensorrt not available. TensorRT conversion will be skipped.")
    TENSORRT_AVAILABLE = False

from train import get_model

def export_onnx(model, dummy_input, onnx_path):
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}},
                      opset_version=11)


def build_engine(onnx_path, engine_path, dummy_input_shape):
    if not TENSORRT_AVAILABLE:
        print("TensorRT not available. Skipping engine build.")
        return None
        
    try:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in parser.errors:
                    print(error)
                return None
        
        # Create builder config (new TensorRT API)
        config = builder.create_builder_config()
        
        # Set memory pool limit (replaces max_workspace_size)
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        except AttributeError:
            # Fallback for older TensorRT versions
            try:
                config.max_workspace_size = 1 << 30
            except AttributeError:
                print("Warning: Could not set workspace size")
        
        # Create optimization profile
        profile = builder.create_optimization_profile()
        input_shape = (1, dummy_input_shape[1]) if len(dummy_input_shape) == 2 else (1,) + dummy_input_shape[1:]
        profile.set_shape('input', input_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)
        
        # Build engine with new API
        try:
            engine = builder.build_serialized_network(network, config)
            if engine is None:
                print("Failed to build TensorRT engine")
                return None
            
            with open(engine_path, 'wb') as f:
                f.write(engine)
            print(f"TensorRT engine saved to {engine_path}")
            return True
            
        except AttributeError:
            # Fallback for older TensorRT versions
            engine = builder.build_engine(network, config)
            if engine is None:
                print("Failed to build TensorRT engine")
                return None
            
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            print(f"TensorRT engine saved to {engine_path}")
            return engine
            
    except Exception as e:
        print(f"TensorRT engine build failed: {e}")
        return None


def main(args):
    if not TENSORRT_AVAILABLE:
        print("TensorRT/PyCUDA not available. Only ONNX export will be performed.")
    
    model = get_model(args.model, args.window_size)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu'))
    model.eval()

    dummy = torch.randn(1, args.window_size*2)
    onnx_path = os.path.join(args.out_dir, f'{args.model}.onnx')
    engine_path = os.path.join(args.out_dir, f'{args.model}.trt')
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Always export ONNX
    export_onnx(model, dummy, onnx_path)
    print(f'Exported ONNX to {onnx_path}')
    
    # Only build TensorRT engine if available
    if TENSORRT_AVAILABLE:
        engine = build_engine(onnx_path, engine_path, dummy.shape)
        if engine:
            print(f'Built TensorRT engine at {engine_path}')
        else:
            print('Failed to build TensorRT engine')
    else:
        print('Skipping TensorRT engine build - dependencies not available')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['ae','ff'], help='Model to convert (onnx supports linear models)')
    parser.add_argument('--weights-path', type=str, required=True)
    parser.add_argument('--window-size', type=int, default=1024)
    parser.add_argument('--out-dir', type=str, default='engines')
    args = parser.parse_args()
    main(args)
