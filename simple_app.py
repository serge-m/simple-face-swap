import cv2
import argparse
import os
import sys
import numpy as np
import insightface
import urllib.request
from dataclasses import dataclass, field
from typing import List, Any, Optional, Union
from tqdm import tqdm

# --- 1. Configuration Data Class ---
@dataclass
class Config:
    source_path: str = ""
    target_path: str = "0"
    execution_providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    headless: bool = True
    models_dir: str = "models"

config = Config()

# --- 2. Utilities ---
def conditional_download(download_file_path: str, url: str) -> None:
    directory = os.path.dirname(download_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(download_file_path):
        print(f"Downloading {url}...")
        request = urllib.request.urlopen(url)
        total = int(request.headers.get("Content-Length", 0))
        with tqdm(total=total, desc="Downloading", unit="B", unit_scale=True, unit_divisor=1024) as progress:
            urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

# --- 3. Face Analyser (Re-created) ---
class FaceAnalyser:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis(name='buffalo_l', providers=config.execution_providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_one_face(self, frame: np.ndarray) -> Any:
        faces = self.app.get(frame)
        if not faces:
            return None
        # Return the face with the smallest x-coordinate (left-most)
        return min(faces, key=lambda x: x.bbox[0])

# --- 4. Face Swapper (Incorporated & Simplified) ---
class FaceSwapper:
    def __init__(self):
        self.model = None
        self.model_name = "inswapper_128_fp16.onnx"

    def pre_check(self) -> bool:
        model_path = os.path.join(config.models_dir, self.model_name)
        conditional_download(model_path, "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx")
        return True

    def initialize(self) -> bool:
        model_path = os.path.join(config.models_dir, self.model_name)
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return False
        
        print(f"Loading Face Swapper model from {model_path}...")
        try:
            self.model = insightface.model_zoo.get_model(model_path, providers=config.execution_providers)
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def process_frame(self, source_face: Any, target_face: Any, temp_frame: np.ndarray) -> np.ndarray:
        if self.model is None:
            return temp_frame
        return self.model.get(temp_frame, target_face, source_face, paste_back=True)

# --- 5. Frame Processing Abstraction ---
class FrameSource:
    def __init__(self, source: Union[str, int]):
        self.source = source
        self.cap = None

    def __enter__(self):
        # Try to convert to int for camera index
        try:
            src = int(self.source)
        except ValueError:
            src = self.source
            
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open source: {self.source}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.cap:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

def parse_args():
    parser = argparse.ArgumentParser(description="Deep-Live-Cam Simple App")
    parser.add_argument('-s', '--source', required=True, help='Path to source image')
    parser.add_argument('-t', '--target', default='0', help='Target camera index or video file path (default: 0)')
    parser.add_argument('--provider', default='openvino', choices=['cpu', 'cuda', 'openvino', 'directml', 'coreml'], help='Execution provider (default: openvino)')
    return parser.parse_args()

def validate_args(args):
    if not os.path.exists(args.source):
        print(f"Error: Source file {args.source} does not exist.")
        sys.exit(1)

def run():
    args = parse_args()
    validate_args(args)
    
    # Setup Config
    config.source_path = args.source
    config.target_path = args.target
    
    providers_map = {
        'cpu': 'CPUExecutionProvider',
        'cuda': 'CUDAExecutionProvider',
        'openvino': 'OpenVINOExecutionProvider',
        'directml': 'DmlExecutionProvider',
        'coreml': 'CoreMLExecutionProvider'
    }
    
    selected_provider = providers_map.get(args.provider, 'CPUExecutionProvider')
    config.execution_providers = [selected_provider]
    # Add CPU fallback
    if 'CPUExecutionProvider' not in config.execution_providers:
        config.execution_providers.append('CPUExecutionProvider')

    # Initialize Components
    analyser = FaceAnalyser()
    swapper = FaceSwapper()
        
    # Load Source
    source_frame = cv2.imread(config.source_path)
    if source_frame is None:
        print("Error: Failed to load source image.")
        return
        
    print("Analyzing source face...")
    source_face = analyser.get_one_face(source_frame)
    if not source_face:
        print("Error: No face detected in source image.")
        return
        
    # Initialize Model
    print(f"Initializing Face Swapper with {args.provider}...")
    if not swapper.pre_check():
        print("Model download check failed.")
        return
        
    if not swapper.initialize():
        print("Model initialization failed.")
        return
        
    # Processing Loop
    print(f"Starting processing on target: {config.target_path}. Press 'q' to quit.")
    
    try:
        with FrameSource(config.target_path) as src:
            while True:
                frame = src.get_frame()
                if frame is None:
                    print("End of stream.")
                    break
                
                # Process
                try:
                    # Detect face in target frame
                    target_face = analyser.get_one_face(frame)
                    if target_face:
                        processed = swapper.process_frame(source_face, target_face, frame)
                    else:
                        processed = frame
                except Exception as e:
                    print(f"Processing error: {e}")
                    processed = frame
                    
                # Visualization
                if frame.shape != processed.shape:
                    processed = cv2.resize(processed, (frame.shape[1], frame.shape[0]))
                    
                combined = cv2.hconcat([frame, processed])
                
                cv2.imshow('Deep-Live-Cam Simple', combined)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Stopped by user.")
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()