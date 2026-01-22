#!/usr/bin/env python3
"""
Quick Start Script for Performance Evaluation

This script provides a simple interface to run performance evaluation
with sensible defaults.

Usage:
    python quick_start_evaluation.py                    # Run full evaluation
    python quick_start_evaluation.py --frames 200       # Use 200 frames
    python quick_start_evaluation.py --camera           # Use webcam
    python quick_start_evaluation.py --help             # Show all options
"""

import sys
import argparse
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Quick start performance evaluation for gesture recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run with defaults (100 frames, synthetic)
  %(prog)s --frames 200                 # Test with 200 frames
  %(prog)s --camera                     # Use webcam input
  %(prog)s --video path/to/video.mp4    # Use video file
  %(prog)s --results-dir my_results/    # Custom output directory
        """
    )
    
    parser.add_argument(
        '--frames',
        type=int,
        default=100,
        help='Number of frames to process (default: 100)'
    )
    
    parser.add_argument(
        '--camera',
        action='store_true',
        help='Use webcam as input source'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Path to video file to use as input'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='src/recognizers/evaluation/results',
        help='Directory to save results (default: src/recognizers/evaluation/results)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots (faster)'
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only show summary (less verbose)'
    )
    
    args = parser.parse_args()
    
    # Import here to show help faster
    from src.recognizers.evaluation.eva_recog_performance import GesturePerformanceEvaluator
    
    print("\n" + "="*80)
    print("GESTURE RECOGNITION PERFORMANCE EVALUATION - QUICK START")
    print("="*80)
    print(f"Configuration:")
    print(f"  Frames:        {args.frames}")
    print(f"  Input Source:  {'Webcam' if args.camera else 'Video' if args.video else 'Synthetic'}")
    print(f"  Results Dir:   {args.results_dir}")
    print(f"  Generate Plots: {not args.no_plots}")
    print("="*80 + "\n")
    
    # Confirm if using camera
    if args.camera:
        response = input("This will use your webcam. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Initialize evaluator
    evaluator = GesturePerformanceEvaluator(output_dir=args.results_dir)
    
    try:
        # Load models
        load_times = evaluator.load_models()
        
        # Run benchmark
        benchmark_results = evaluator.run_inference_benchmark(
            num_frames=args.frames,
            use_camera=args.camera,
            video_path=args.video
        )
        
        # Compute metrics
        metrics = evaluator.compute_metrics(benchmark_results, load_times)
        
        # Generate reports
        if not args.no_plots:
            evaluator.generate_visual_report(benchmark_results, metrics)
        
        report = evaluator.generate_text_report(metrics)
        
        # Save metrics
        evaluator.save_metrics_json(metrics)
        
        # Print summary
        if args.summary_only:
            print("\n" + "="*80)
            print("QUICK SUMMARY")
            print("="*80)
            print(f"Average Latency: {metrics['latency']['mean']:.2f}ms")
            print(f"Average FPS:     {metrics['fps']['average']:.2f}")
            print(f"Max Memory:      {metrics['memory']['max']:.2f}MB")
            print(f"Avg CPU:         {metrics['cpu']['mean']:.2f}%")
            print("="*80)
        else:
            print("\n" + report)
        
        print("\n" + "="*80)
        print("✓ EVALUATION COMPLETE")
        print("="*80)
        print(f"Results saved to: {evaluator.output_dir.absolute()}")
        
        if not args.no_plots:
            print(f"\nView visual report:")
            print(f"  {evaluator.plots_dir}/performance_report.png")
        
        print(f"\nView detailed report:")
        print(f"  {evaluator.output_dir}/performance_report.txt")
        print(f"\nMetrics data:")
        print(f"  {evaluator.output_dir}/metrics.json")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
