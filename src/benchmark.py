import os
import time
import argparse
import numpy as np
import torch

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    print("Warning: pycuda and/or tensorrt not available. TensorRT benchmarking will be skipped.")
    TENSORRT_AVAILABLE = False

from data_loader import get_dataloaders
from train import get_model

def measure_pytorch(model, inputs, device):
    model.to(device)
    model.eval()
    times = []
    with torch.no_grad():
        for x in inputs:
            x = x.to(device)
            start = time.time()
            _ = model(x)[0]
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append(end-start)
    return np.mean(times), len(inputs)/np.sum(times)

def measure_trt(engine_path, inputs_np):
    if not TENSORRT_AVAILABLE:
        print("TensorRT not available. Skipping TensorRT benchmarking.")
        return 0.0, 0.0
        
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    times = []
    bindings = []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        alloc = cuda.mem_alloc(size * dtype().nbytes)
        bindings.append(int(alloc))
    stream = cuda.Stream()
    for inp in inputs_np:
        start = time.time()
        # Set binding
        context.execute_async_v2(bindings, stream.handle, None)
        stream.synchronize()
        end = time.time()
        times.append(end-start)
    return np.mean(times), len(inputs_np)/np.sum(times)


def benchmark(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader = get_dataloaders(args.clean, args.jammed,
                                     window_size=args.window_size,
                                     batch_size=1)
    model = get_model(args.model, args.window_size).to(device)
    model.load_state_dict(torch.load(args.model_path))

    # Prepare inputs list
    inputs = [x for x, _ in val_loader]
    inp_flat = [x.view(x.size(0), -1) if 'ae' in args.model or args.model=='ff' else x for x in inputs]

    pt_lat, pt_through = measure_pytorch(model, inp_flat, device)
    print(f'PyTorch {args.model} - Latency: {pt_lat*1000:.2f} ms, Throughput: {pt_through:.2f} samples/s')

    if TENSORRT_AVAILABLE and hasattr(args, 'engine_path') and os.path.exists(args.engine_path):
        trt_lat, trt_through = measure_trt(args.engine_path, [i.numpy() for i in inp_flat])
        print(f'TensorRT {args.model} - Latency: {trt_lat*1000:.2f} ms, Throughput: {trt_through:.2f} samples/s')
        print(f'Speedup: {pt_lat/trt_lat:.2f}x')
    else:
        print('TensorRT benchmarking skipped - engine not available or TensorRT not installed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['ae','ff'], help='Model to benchmark')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--engine-path', type=str, required=True)
    parser.add_argument('--clean', type=str, default='clean_5g_dataset.h5')
    parser.add_argument('--jammed', type=str, default='jammed_5g_dataset.h5')
    parser.add_argument('--window-size', type=int, default=128)
    args = parser.parse_args()
    benchmark(args)
