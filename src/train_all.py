import os
import argparse
import torch
from types import SimpleNamespace
from train import train, get_model
from evaluate import evaluate

try:
    from convert_tensorrt import main as convert_main
    from benchmark import benchmark
    TENSORRT_AVAILABLE = True
except ImportError:
    print("Warning: TensorRT/PyCUDA not available. Skipping TensorRT operations.")
    TENSORRT_AVAILABLE = False

def run_pipeline(model_name, args):
    print(f"\n=== Processing model: {model_name} ===")
    # Directories
    weight_dir = os.path.join(args.out_dir, 'weights')
    engine_dir = os.path.join(args.out_dir, 'engines')
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(engine_dir, exist_ok=True)

    # Train
    train_args = SimpleNamespace(
        model=model_name,
        clean=args.clean,
        jammed=args.jammed,
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        use_psd=args.use_psd,
        out_dir=weight_dir
    )
    train(train_args)

    # Evaluate
    best_path = os.path.join(weight_dir, f"{model_name}_best.pth")
    eval_args = SimpleNamespace(
        model=model_name,
        model_path=best_path,
        clean=args.clean,
        jammed=args.jammed,
        window_size=args.window_size,
        batch_size=args.batch_size
    )
    evaluate(eval_args)

    # Convert to TensorRT (only for supported)
    if TENSORRT_AVAILABLE and model_name in ['ae', 'ff']:
        convert_args = SimpleNamespace(
            model=model_name,
            weights_path=best_path,
            window_size=args.window_size,
            out_dir=engine_dir
        )
        convert_main(convert_args)
        engine_path = os.path.join(engine_dir, f"{model_name}.trt")
        # Benchmark
        bench_args = SimpleNamespace(
            model=model_name,
            model_path=best_path,
            engine_path=engine_path,
            clean=args.clean,
            jammed=args.jammed,
            window_size=args.window_size
        )
        benchmark(bench_args)
    else:
        if not TENSORRT_AVAILABLE:
            print(f"Skipping TensorRT conversion for {model_name} - TensorRT/PyCUDA not available.")
        else:
            print(f"Skipping TensorRT conversion for {model_name} - model not supported for TensorRT conversion.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full pipeline for all models')
    parser.add_argument('--models', nargs='+', default=['ae','cnn_ae','lstm_ae','resnet_ae','aae','ff'],
                        help='List of models to process')
    parser.add_argument('--clean', type=str, default='clean_5g_dataset.h5')
    parser.add_argument('--jammed', type=str, default='jammed_5g_dataset.h5')
    parser.add_argument('--window-size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use-psd', action='store_true', help='Include PSD features')
    parser.add_argument('--out-dir', type=str, default='output')
    args = parser.parse_args()
    for m in args.models:
        run_pipeline(m, args)
