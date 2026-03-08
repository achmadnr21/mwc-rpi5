"""
Export YOLO11n (COCO) to ONNX for RPi5 bird detection benchmarking.

Downloads the official YOLO11n pretrained weights (COCO 80 classes, includes
"bird" = class 14), exports to ONNX with opset 12 and 320×320 input, and
copies the result to markaswalet-stream/yolo11n_bird.onnx.

Requirements:
    pip install ultralytics onnx onnxsim

Usage:
    python export_yolo11n_bird.py

For optional INT8 quantization (faster on RPi5, ~50% less RAM):
    python export_yolo11n_bird.py --quantize
"""

import os
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantize', action='store_true',
                        help='Apply INT8 static quantization via onnxruntime')
    parser.add_argument('--imgsz', type=int, default=320,
                        help='Input image size (default: 320 for RPi5 speed)')
    args = parser.parse_args()

    from ultralytics import YOLO

    print('=== Downloading YOLO11n pretrained (COCO 80 classes) ===')
    model = YOLO('yolo11n.pt')   # auto-downloads ~5.7 MB

    print(f'\n=== Exporting to ONNX (imgsz={args.imgsz}, opset=12) ===')
    export_path = model.export(
        format='onnx',
        imgsz=args.imgsz,
        simplify=True,   # onnxsim — removes redundant nodes
        opset=12,        # OpenCV DNN supports opset ≤ 12 well
        dynamic=False,   # static batch=1 for RPi5
    )
    print(f'Exported: {export_path}')

    if args.quantize:
        _apply_int8_quantization(export_path)

    deploy_dir  = os.path.join(os.path.dirname(__file__), 'markaswalet-stream')
    deploy_path = os.path.join(deploy_dir, 'yolo11n_bird.onnx')
    shutil.copy(export_path, deploy_path)
    print(f'\nDeployed to: {deploy_path}')

    _print_benchmark_note(deploy_path)


def _apply_int8_quantization(onnx_path: str) -> str:
    """Static INT8 quantisation via onnxruntime-tools.

    Reduces model size ~75% and speeds up CPU inference on RPi5 ARM Cortex-A76.
    Requires: pip install onnxruntime onnxruntime-tools
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print('[WARN] onnxruntime-tools not installed — skipping quantization.')
        print('       pip install onnxruntime onnxruntime-tools')
        return onnx_path

    out_path = onnx_path.replace('.onnx', '_int8.onnx')
    print(f'\n=== Applying INT8 dynamic quantization → {out_path} ===')
    quantize_dynamic(
        model_input=onnx_path,
        model_output=out_path,
        weight_type=QuantType.QUInt8,
    )
    print(f'INT8 model saved: {out_path}')
    return out_path


def _print_benchmark_note(deploy_path: str):
    size_mb = os.path.getsize(deploy_path) / (1024 * 1024)
    print(f"""
=== Benchmark Notes ===
Model path : {deploy_path}
Size       : {size_mb:.1f} MB
Input      : 320×320 BGR → normalised [0,1]
Bird class : COCO class 14 ("bird")
Expected inference on RPi5 (Cortex-A76 @ 2.4 GHz): ~80-150 ms/frame

To enable in the pipeline, set in config or dashboard:
    use_yolo        = true
    yolo_model_type = "yolo11n_coco"

Compare against custom YOLOv8n (mAP50=90.7% on swiftlet dataset):
  - YOLO11n COCO may detect non-swiftlet birds (false positives in swiftlet houses)
  - YOLO11n COCO may miss small/fast swiftlets (trained on larger bird instances)
  - Use both side-by-side to measure precision / recall on your footage
""")


if __name__ == '__main__':
    main()
