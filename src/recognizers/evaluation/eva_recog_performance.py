"""
Gesture Recognition Performance Evaluation Script

This script evaluates the performance and efficiency of the gesture recognition model.
It measures various metrics including latency, FPS, CPU/GPU usage, and memory consumption.

Outputs:
    - Visual reports (plots) saved to results/plots/
    - Text report printed to console and saved to results/performance_report.txt
    - Detailed metrics saved to results/metrics.json
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import os
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import psutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.detectors.hand_detector import HandDetector
from src.recognizers.models.gesture_recognizer import GestureRecognizer
from src.recognizers.utils.performance_metric import (
    PerformanceMetrics,
    measure_model_load_time,
    measure_inference_latency,
    calculate_fps,
    calculate_fps_over_time,
    get_memory_usage,
    get_cpu_usage,
    get_gpu_usage,
    compute_statistics,
    check_performance_thresholds,
    PerformanceTracker
)

# Set matplotlib style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class GesturePerformanceEvaluator:
    """Evaluates performance metrics for gesture recognition system"""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize performance evaluator
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.metrics = PerformanceMetrics()
        self.current_run_dir: Optional[Path] = None
        self.metrics_cache: Dict[str, Any] = {}
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.detector = None
        self.recognizer = None
        
    def load_models(self) -> Dict[str, float]:
        """
        Load hand detector and gesture recognizer models
        
        Returns:
            Dictionary with load times for each model
        """
        print("\n" + "="*80)
        print("LOADING MODELS")
        print("="*80)
        
        # Measure detector load time
        with PerformanceTracker("HandDetector Load"):
            detector_load_time, self.detector = measure_model_load_time(
                HandDetector,
                max_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        
        # Measure recognizer load time
        with PerformanceTracker("GestureRecognizer Load"):
            recognizer_load_time, self.recognizer = measure_model_load_time(
                GestureRecognizer
            )
        
        load_times = {
            'detector_load_ms': detector_load_time,
            'recognizer_load_ms': recognizer_load_time,
            'total_load_ms': detector_load_time + recognizer_load_time
        }
        
        print(f"\n✓ Detector loaded in {detector_load_time:.2f}ms")
        print(f"✓ Recognizer loaded in {recognizer_load_time:.2f}ms")
        print(f"✓ Total load time: {load_times['total_load_ms']:.2f}ms")
        
        return load_times
    
    def run_inference_benchmark(self, num_frames: int = 100, 
                                use_camera: bool = False,
                                video_path: str = None) -> Dict[str, any]:
        """
        Run inference benchmark on video frames
        
        Args:
            num_frames: Number of frames to process
            use_camera: Whether to use webcam
            video_path: Path to video file (if not using camera)
            
        Returns:
            Dictionary with benchmark results
        """
        print("\n" + "="*80)
        print("RUNNING INFERENCE BENCHMARK")
        print("="*80)
        print(f"Frames to process: {num_frames}")
        
        # Reset metrics
        self.metrics.reset()

        # Prime CPU counters so the first sampled value is meaningful
        try:
            self.metrics.process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        
        # Open video source
        if use_camera:
            cap = cv2.VideoCapture(0)
            print("Using webcam as video source")
        elif video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            print(f"Using video file: {video_path}")
        else:
            # Generate synthetic frames
            print("Generating synthetic frames for testing")
            cap = None
        
        frame_count = 0
        gestures_detected = []
        
        print("\nProcessing frames...")
        start_benchmark = time.time()
        
        try:
            while frame_count < num_frames:
                # Get frame
                if cap is not None:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"End of video reached at frame {frame_count}")
                        break
                else:
                    # Generate synthetic frame (640x480 RGB)
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # Record timestamp
                timestamp = time.time()
                self.metrics.timestamps.append(timestamp)
                
                # Measure inference latency
                latency, result = self._process_single_frame(frame)
                
                # Record metrics
                self.metrics.latencies.append(latency)
                gestures_detected.append(result)
                
                # Record system metrics (every 10 frames to reduce overhead)
                if frame_count % 10 == 0:
                    memory = get_memory_usage()
                    cpu = get_cpu_usage(interval=None)
                    gpu = get_gpu_usage()
                    
                    self.metrics.memory_usage.append(memory['rss'])
                    self.metrics.cpu_usage.append(cpu['process_percent'])
                    
                    if gpu['available']:
                        self.metrics.gpu_usage.append(gpu['gpu_utilization'])
                
                frame_count += 1
                
                # Progress indicator
                if frame_count % 20 == 0:
                    print(f"  Processed {frame_count}/{num_frames} frames...")
        
        finally:
            if cap is not None:
                cap.release()
        
        end_benchmark = time.time()
        total_time = end_benchmark - start_benchmark
        
        print(f"\n✓ Processed {frame_count} frames in {total_time:.2f}s")
        print(f"✓ Average throughput: {frame_count/total_time:.2f} FPS")
        
        # Compile results
        results = {
            'frames_processed': frame_count,
            'total_time_s': total_time,
            'gestures_detected': gestures_detected,
            'latencies': self.metrics.latencies,
            'timestamps': self.metrics.timestamps,
            'memory_usage': self.metrics.memory_usage,
            'cpu_usage': self.metrics.cpu_usage,
            'gpu_usage': self.metrics.gpu_usage
        }
        
        return results
    
    def _process_single_frame(self, frame: np.ndarray) -> Tuple[float, str]:
        """
        Process a single frame and return latency + result
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (latency_ms, gesture_name)
        """
        start = time.perf_counter()
        
        # Detect hands
        frame = self.detector.find_hands(frame, draw=False)
        landmark_list = self.detector.get_position(frame, hand_no=0)
        
        # Recognize gesture
        gesture = 'none'
        if landmark_list:
            fingers = self.detector.fingers_up(landmark_list)
            gesture = self.recognizer.recognize_gesture(fingers, landmark_list)
        
        end = time.perf_counter()
        latency = (end - start) * 1000
        
        return latency, gesture
    
    def compute_metrics(self, benchmark_results: Dict[str, any], 
                       load_times: Dict[str, float]) -> Dict[str, any]:
        """
        Compute comprehensive performance metrics
        
        Args:
            benchmark_results: Results from inference benchmark
            load_times: Model loading times
            
        Returns:
            Dictionary with all computed metrics
        """
        print("\n" + "="*80)
        print("COMPUTING METRICS")
        print("="*80)
        
        # Latency statistics
        latency_stats = compute_statistics(benchmark_results['latencies'])
        
        # FPS calculations
        avg_fps = calculate_fps(benchmark_results['latencies'])
        fps_over_time = calculate_fps_over_time(benchmark_results['timestamps'], window_size=30)
        
        # Memory statistics
        memory_stats = compute_statistics(benchmark_results['memory_usage'])
        
        # CPU statistics
        cpu_stats = compute_statistics(benchmark_results['cpu_usage'])
        
        # GPU statistics
        gpu_stats = {}
        if benchmark_results['gpu_usage']:
            gpu_stats = compute_statistics(benchmark_results['gpu_usage'])
        
        # Compile all metrics
        metrics = {
            'load_time': load_times,
            'latency': latency_stats,
            'fps': {
                'average': avg_fps,
                'over_time': fps_over_time,
                'min': min(fps_over_time) if fps_over_time else 0,
                'max': max(fps_over_time) if fps_over_time else 0
            },
            'memory': memory_stats,
            'cpu': cpu_stats,
            'gpu': gpu_stats if gpu_stats else {'available': False},
            'frames_processed': benchmark_results['frames_processed'],
            'total_time': benchmark_results['total_time_s']
        }
        self.metrics_cache = metrics
        
        print("✓ Metrics computed successfully")
        
        return metrics
    
    def generate_visual_report(self, benchmark_results: Dict[str, any], 
                              metrics: Dict[str, any],
                              save_dir: Optional[Path] = None):
        """
        Generate visual reports with plots
        
        Args:
            benchmark_results: Results from inference benchmark
            metrics: Computed metrics
        """
        print("\n" + "="*80)
        print("GENERATING VISUAL REPORTS")
        print("="*80)
        
        # Resolve target directory for plots (keep alongside metrics file when available)
        target_dir = save_dir or self.current_run_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Latency Histogram
        ax1 = plt.subplot(2, 3, 1)
        latencies = benchmark_results['latencies']
        ax1.hist(latencies, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(metrics['latency']['mean'], color='red', linestyle='--', 
                   label=f"Mean: {metrics['latency']['mean']:.2f}ms")
        ax1.axvline(metrics['latency']['p95'], color='orange', linestyle='--',
                   label=f"P95: {metrics['latency']['p95']:.2f}ms")
        ax1.set_xlabel('Latency (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Inference Latency Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. FPS over Time
        ax2 = plt.subplot(2, 3, 2)
        fps_values = metrics['fps']['over_time']
        if fps_values:
            ax2.plot(fps_values, color='green', linewidth=1.5)
            ax2.axhline(metrics['fps']['average'], color='red', linestyle='--',
                       label=f"Average: {metrics['fps']['average']:.2f} FPS")
            ax2.fill_between(range(len(fps_values)), fps_values, alpha=0.3, color='green')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('FPS')
        ax2.set_title('FPS Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory Usage over Time
        ax3 = plt.subplot(2, 3, 3)
        memory_usage = benchmark_results['memory_usage']
        if memory_usage:
            ax3.plot(memory_usage, color='purple', linewidth=1.5)
            ax3.axhline(metrics['memory']['mean'], color='red', linestyle='--',
                       label=f"Mean: {metrics['memory']['mean']:.2f}MB")
            ax3.fill_between(range(len(memory_usage)), memory_usage, alpha=0.3, color='purple')
        ax3.set_xlabel('Sample')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_title('Memory Usage Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. CPU Usage over Time
        ax4 = plt.subplot(2, 3, 4)
        cpu_usage = benchmark_results['cpu_usage']
        if cpu_usage:
            ax4.plot(cpu_usage, color='orange', linewidth=1.5)
            ax4.axhline(metrics['cpu']['mean'], color='red', linestyle='--',
                       label=f"Mean: {metrics['cpu']['mean']:.2f}%")
            ax4.fill_between(range(len(cpu_usage)), cpu_usage, alpha=0.3, color='orange')
        ax4.set_xlabel('Sample')
        ax4.set_ylabel('CPU Usage (%)')
        ax4.set_title('CPU Utilization Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Latency Box Plot
        ax5 = plt.subplot(2, 3, 5)
        ax5.boxplot([latencies], labels=['Latency'])
        ax5.set_ylabel('Latency (ms)')
        ax5.set_title('Latency Box Plot')
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary Statistics Table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_data = [
            ['Metric', 'Value'],
            ['', ''],
            ['Avg Latency', f"{metrics['latency']['mean']:.2f} ms"],
            ['P95 Latency', f"{metrics['latency']['p95']:.2f} ms"],
            ['P99 Latency', f"{metrics['latency']['p99']:.2f} ms"],
            ['Average FPS', f"{metrics['fps']['average']:.2f}"],
            ['Max Memory', f"{metrics['memory']['max']:.2f} MB"],
            ['Avg CPU', f"{metrics['cpu']['mean']:.2f} %"],
            ['Frames', f"{metrics['frames_processed']}"],
            ['Total Time', f"{metrics['total_time']:.2f} s"]
        ]
        
        table = ax6.table(cellText=summary_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = target_dir / "performance_report.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visual report saved to: {plot_path}")
        
        plt.close()
        
        # Generate individual detailed plots
        self._generate_detailed_plots(benchmark_results, metrics, target_dir)
    
    def _generate_detailed_plots(self, benchmark_results: Dict[str, any],
                                 metrics: Dict[str, any],
                                 save_dir: Optional[Path] = None):
        """Generate additional detailed plots"""

        target_dir = save_dir or self.current_run_dir 
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Latency vs Frame Number
        fig, ax = plt.subplots(figsize=(12, 6))
        latencies = benchmark_results['latencies']
        ax.plot(latencies, color='blue', alpha=0.6, linewidth=0.8)
        ax.axhline(metrics['latency']['mean'], color='red', linestyle='--',
                  label=f"Mean: {metrics['latency']['mean']:.2f}ms")
        ax.axhline(metrics['latency']['p95'], color='orange', linestyle='--',
                  label=f"P95: {metrics['latency']['p95']:.2f}ms")
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency per Frame')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(target_dir / "latency_per_frame.png", dpi=200)
        plt.close()
        
        print(f"✓ Detailed plots saved to: {target_dir}")
    
    def generate_text_report(self, metrics: Dict[str, any], 
                            thresholds: Dict[str, float] = None,
                            save_dir: Optional[Path] = None) -> str:
        """
        Generate text report with metrics
        
        Args:
            metrics: Computed metrics
            thresholds: Performance thresholds for warnings
            
        Returns:
            Text report string
        """
        print("\n" + "="*80)
        print("GENERATING TEXT REPORT")
        print("="*80)
        
        # Default thresholds
        if thresholds is None:
            thresholds = {
                'max_latency_ms': 50.0,
                'min_fps': 20.0,
                'max_memory_mb': 1000.0,
                'max_cpu_percent': 80.0
            }
        
        # Check thresholds
        threshold_results = check_performance_thresholds(metrics, **thresholds)
        
        # Build report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("GESTURE RECOGNITION PERFORMANCE EVALUATION REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Model Load Time
        report_lines.append("MODEL LOAD TIME")
        report_lines.append("-" * 80)
        report_lines.append(f"  Detector Load Time:    {metrics['load_time']['detector_load_ms']:>10.2f} ms")
        report_lines.append(f"  Recognizer Load Time:  {metrics['load_time']['recognizer_load_ms']:>10.2f} ms")
        report_lines.append(f"  Total Load Time:       {metrics['load_time']['total_load_ms']:>10.2f} ms")
        report_lines.append("")
        
        # Inference Latency
        report_lines.append("INFERENCE LATENCY")
        report_lines.append("-" * 80)
        report_lines.append(f"  Mean:      {metrics['latency']['mean']:>10.2f} ms")
        report_lines.append(f"  Median:    {metrics['latency']['median']:>10.2f} ms")
        report_lines.append(f"  Std Dev:   {metrics['latency']['std']:>10.2f} ms")
        report_lines.append(f"  Min:       {metrics['latency']['min']:>10.2f} ms")
        report_lines.append(f"  Max:       {metrics['latency']['max']:>10.2f} ms")
        report_lines.append(f"  P50:       {metrics['latency']['p50']:>10.2f} ms")
        report_lines.append(f"  P95:       {metrics['latency']['p95']:>10.2f} ms")
        report_lines.append(f"  P99:       {metrics['latency']['p99']:>10.2f} ms")
        report_lines.append("")
        
        # FPS
        report_lines.append("FRAMES PER SECOND (FPS)")
        report_lines.append("-" * 80)
        report_lines.append(f"  Average FPS:  {metrics['fps']['average']:>10.2f}")
        report_lines.append(f"  Min FPS:      {metrics['fps']['min']:>10.2f}")
        report_lines.append(f"  Max FPS:      {metrics['fps']['max']:>10.2f}")
        report_lines.append("")
        
        # Memory Usage
        report_lines.append("MEMORY USAGE")
        report_lines.append("-" * 80)
        report_lines.append(f"  Mean:      {metrics['memory']['mean']:>10.2f} MB")
        report_lines.append(f"  Min:       {metrics['memory']['min']:>10.2f} MB")
        report_lines.append(f"  Max:       {metrics['memory']['max']:>10.2f} MB")
        report_lines.append(f"  Std Dev:   {metrics['memory']['std']:>10.2f} MB")
        report_lines.append("")
        
        # CPU Usage
        report_lines.append("CPU UTILIZATION")
        report_lines.append("-" * 80)
        report_lines.append(f"  Mean:      {metrics['cpu']['mean']:>10.2f} %")
        report_lines.append(f"  Min:       {metrics['cpu']['min']:>10.2f} %")
        report_lines.append(f"  Max:       {metrics['cpu']['max']:>10.2f} %")
        report_lines.append(f"  Std Dev:   {metrics['cpu']['std']:>10.2f} %")
        report_lines.append("")
        
        # GPU Usage
        if metrics['gpu'].get('available', False):
            report_lines.append("GPU UTILIZATION")
            report_lines.append("-" * 80)
            report_lines.append(f"  Mean:      {metrics['gpu']['mean']:>10.2f} %")
            report_lines.append(f"  Min:       {metrics['gpu']['min']:>10.2f} %")
            report_lines.append(f"  Max:       {metrics['gpu']['max']:>10.2f} %")
            report_lines.append("")
        else:
            report_lines.append("GPU UTILIZATION")
            report_lines.append("-" * 80)
            report_lines.append("  GPU not available or not detected")
            report_lines.append("")
        
        # Processing Summary
        report_lines.append("PROCESSING SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"  Frames Processed:  {metrics['frames_processed']:>10}")
        report_lines.append(f"  Total Time:        {metrics['total_time']:>10.2f} s")
        report_lines.append(f"  Throughput:        {metrics['frames_processed']/metrics['total_time']:>10.2f} FPS")
        report_lines.append("")
        
        # Threshold Warnings
        report_lines.append("PERFORMANCE ANALYSIS")
        report_lines.append("-" * 80)
        
        if threshold_results['errors']:
            report_lines.append("  ERRORS:")
            for error in threshold_results['errors']:
                report_lines.append(f"    ✗ {error}")
            report_lines.append("")
        
        if threshold_results['warnings']:
            report_lines.append("  WARNINGS:")
            for warning in threshold_results['warnings']:
                report_lines.append(f"    ⚠ {warning}")
            report_lines.append("")
        
        if not threshold_results['errors'] and not threshold_results['warnings']:
            report_lines.append("  ✓ All metrics within acceptable thresholds")
            report_lines.append("")
        
        if threshold_results['info']:
            report_lines.append("  INFO:")
            for info in threshold_results['info']:
                report_lines.append(f"    ℹ {info}")
            report_lines.append("")
        
        report_lines.append("="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        report = "\n".join(report_lines)
        
        # Save to file
        target_dir = save_dir or self.current_run_dir or self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        report_path = target_dir / "performance_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Text report saved to: {report_path}")
        
        return report
    
    def save_metrics_json(self, metrics: Dict[str, any]):
        """
        Save metrics to JSON file
        
        Args:
            metrics: Computed metrics dictionary
        """
        # Convert any numpy types to Python native types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        metrics_clean = convert_types(metrics)
        
        json_path = self.output_dir / "metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics_clean, f, indent=2)
        
        print(f"✓ Metrics JSON saved to: {json_path}")

    def save_metrics_to_file(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        filepath: Optional[Union[str, Path]] = None,
        format: str = 'json',
        run_id: Optional[str] = None,
        version: Optional[str] = None,
        create_timestamped_folder: bool = True
    ) -> str:
        """
        Save performance metrics with metadata in txt, json, or csv formats.
        """
        if metrics is None:
            metrics = self.metrics_cache

        if not metrics:
            print("No metrics available to save.")
            return ""

        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_hash = hashlib.md5(str(metrics.get('latency', {})).encode()).hexdigest()[:8]
            run_id = f"{timestamp}_{metrics_hash}"

        if version is None:
            version = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if create_timestamped_folder:
            now = datetime.now()
            timestamped_folder = now.strftime("%M_%H_%d_%m_%Y")
            save_dir = self.output_dir / timestamped_folder
        else:
            save_dir = self.output_dir
            timestamped_folder = None

        save_dir.mkdir(parents=True, exist_ok=True)
        self.current_run_dir = save_dir

        if filepath is None:
            filepath = save_dir / f"performance_metrics_{run_id}.{format}"
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            'run_id': run_id,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'format': format,
            'timestamped_folder': timestamped_folder
        }

        def convert_types(obj: Any) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        if format == 'txt':
            lines = []
            lines.append("="*70)
            lines.append(" PERFORMANCE METRICS REPORT ".center(70, '='))
            lines.append("="*70)
            lines.append("")
            lines.append("METADATA")
            lines.append("-"*70)
            lines.append(f"Run ID:       {metadata['run_id']}")
            lines.append(f"Version:      {metadata['version']}")
            lines.append(f"Timestamp:    {metadata['timestamp']}")
            if metadata['timestamped_folder']:
                lines.append(f"Folder:       {metadata['timestamped_folder']}")
            lines.append("")
            lines.append("LOAD TIME (ms)")
            lines.append("-"*70)
            lines.append(f"  Detector:   {metrics['load_time']['detector_load_ms']:.2f}")
            lines.append(f"  Recognizer: {metrics['load_time']['recognizer_load_ms']:.2f}")
            lines.append(f"  Total:      {metrics['load_time']['total_load_ms']:.2f}")
            lines.append("")
            lines.append("LATENCY (ms)")
            lines.append("-"*70)
            for key in ['mean','median','std','min','max','p50','p95','p99']:
                if key in metrics['latency']:
                    lines.append(f"  {key.upper():<6}: {metrics['latency'][key]:.2f}")
            lines.append("")
            lines.append("FPS")
            lines.append("-"*70)
            lines.append(f"  Average: {metrics['fps']['average']:.2f}")
            lines.append(f"  Min:     {metrics['fps']['min']:.2f}")
            lines.append(f"  Max:     {metrics['fps']['max']:.2f}")
            lines.append("")
            lines.append("MEMORY (MB)")
            lines.append("-"*70)
            for key in ['mean','min','max','std','p50','p95','p99']:
                if key in metrics['memory']:
                    lines.append(f"  {key.upper():<6}: {metrics['memory'][key]:.2f}")
            lines.append("")
            lines.append("CPU (%)")
            lines.append("-"*70)
            for key in ['mean','min','max','std','p50','p95','p99']:
                if key in metrics['cpu']:
                    lines.append(f"  {key.upper():<6}: {metrics['cpu'][key]:.2f}")
            lines.append("")
            lines.append("GPU (%)")
            lines.append("-"*70)
            if metrics['gpu'].get('available', False):
                for key in ['mean','min','max','std','p50','p95','p99']:
                    if key in metrics['gpu']:
                        lines.append(f"  {key.upper():<6}: {metrics['gpu'][key]:.2f}")
            else:
                lines.append("  GPU not available or not detected")
            lines.append("")
            lines.append("SUMMARY")
            lines.append("-"*70)
            lines.append(f"  Frames Processed: {metrics['frames_processed']}")
            lines.append(f"  Total Time (s):   {metrics['total_time']:.2f}")
            with open(filepath, 'w') as f:
                f.write("\n".join(lines))
            print(f"Metrics report saved to: {filepath}")

        elif format == 'json':
            serializable_metrics = convert_types(metrics)
            output_data = {
                'metadata': metadata,
                'metrics': serializable_metrics
            }
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Metrics report saved to: {filepath}")

        elif format == 'csv':
            rows = []
            for section in ['load_time', 'latency', 'memory', 'cpu', 'gpu']:
                if section in metrics:
                    for key, value in metrics[section].items():
                        rows.append([f"{section}.{key}", convert_types(value)])
            rows.append(["frames_processed", metrics['frames_processed']])
            rows.append(["total_time", metrics['total_time']])
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerows(rows)
            print(f"Metrics report saved to: {filepath}")

        else:
            raise ValueError("Unsupported format: use 'txt', 'json', or 'csv'.")

        return str(filepath)


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("GESTURE RECOGNITION PERFORMANCE EVALUATION")
    print("="*80)
    print("This script will evaluate the performance of the gesture recognition system")
    print("="*80 + "\n")
    
    # Initialize evaluator (save all artifacts to docs path under repo)
    repo_root = Path(__file__).resolve().parent.parent.parent
    target_dir = repo_root / "docs" / "evaluation_result" / "recognizer_evaluator"
    evaluator = GesturePerformanceEvaluator(output_dir=str(target_dir))
    
    # Load models
    load_times = evaluator.load_models()
    
    # Run benchmark
    benchmark_results = evaluator.run_inference_benchmark(
        num_frames=100,
        use_camera=False,
        video_path=None
    )
    
    # Compute metrics
    metrics = evaluator.compute_metrics(benchmark_results, load_times)
    
    # Save metrics snapshot with metadata (sets run directory)
    metrics_path = evaluator.save_metrics_to_file(metrics, format="json")

    # Generate visual report in same run directory
    evaluator.generate_visual_report(benchmark_results, metrics, save_dir=evaluator.current_run_dir)

    # Generate text report in same run directory
    report = evaluator.generate_text_report(metrics, save_dir=evaluator.current_run_dir)
    
    # Print report to console
    print("\n" + report)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {evaluator.current_run_dir or evaluator.output_dir}")
    print(f"  - Visual report: {(evaluator.current_run_dir) / 'performance_report.png'}")
    print(f"  - Text report: {(evaluator.current_run_dir or evaluator.output_dir) / 'performance_report.txt'}")
    print(f"  - Metrics JSON: {metrics_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
